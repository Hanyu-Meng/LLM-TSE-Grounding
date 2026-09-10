"""Dataset for the three Phase-2 training cases, built from tokenized VB-DEMAND.

Each item is a 2-row grid (row 0 = CV3 audio stream, row 1 = Qwen text stream)
laid out as ``[PROMPT | INPUT | ANSWER]`` (mirrors EchoMind's construction):

  case "t2t"               input = noisy CV3 tokens (audio row);  answer = clean tokens
  case "audio2token"       input = whisper(noisy)+text prompt;    answer = clean tokens
  case "audio2token_text"  input = whisper(noisy)+text prompt;    answer = clean tokens + transcript

Audio-row ids are stored layer-shifted (`vocab.layershift`) for the combined
embedding; **labels stay raw** (0-based in each slice). The prompt+input regions
are masked (-100); only the answer region is supervised.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.store import load_tokens, manifest_to_dict, read_manifest
from ..data.vbdemand import load_transcripts
from .vocab import SEVocabConfig

IGNORE_INDEX = -100

DEFAULT_PROMPTS = {
    "t2t": "",
    "t2t_text": "",                       # noisy tokens -> clean tokens + transcript
    "audio2token": "Enhance the noisy speech.",
    "audio2token_text": "Enhance the noisy speech and transcribe it.",
    "at2t": "Enhance the noisy speech.",        # whisper(noisy) + enhanced token -> clean token
    "at2t_text": "Enhance the noisy speech and transcribe it.",
    "e1n_t2t": "",          # noisy CV3 token + (E1-)enhanced CV3 token -> clean token
    "e1n_t2t_text": "",
}
# tasks whose input is the noisy CV3 token stream (vs whisper audio)
_TOKEN_INPUT = ("t2t", "t2t_text")
# tasks that fuse whisper(noisy) features + an (enhanced) CV3 token stream as input
_AUDIO_TOKEN_INPUT = ("at2t", "at2t_text")
# tasks that fuse two discrete token streams (original noisy + enhanced) as input
_DUAL_TOKEN_INPUT = ("e1n_t2t", "e1n_t2t_text")
# tasks that also supervise the text (transcript) output stream
_TEXT_OUTPUT = ("t2t_text", "audio2token_text", "at2t_text", "e1n_t2t_text")


class SEDataset(Dataset):
    def __init__(
        self,
        tokens_root: str,
        split: str,
        vocab: SEVocabConfig,
        tokenizer,
        task_type: str = "t2t",
        prompt: Optional[str] = None,
        data_root: Optional[str] = None,
        txt_subdir: Optional[str] = None,
        whisper_n_mels: int = 128,
        whisper_ds_rate: int = 5,
        cv3_root: Optional[str] = None,
        limit: Optional[int] = None,
        noisy_wav_root: Optional[str] = None,
    ) -> None:
        assert task_type in DEFAULT_PROMPTS, f"unknown task_type {task_type}"
        self.tokens_root = tokens_root
        self.split = split
        self.v = vocab
        self.tok = tokenizer
        self.task_type = task_type
        self.prompt = DEFAULT_PROMPTS[task_type] if prompt is None else prompt
        self.n_mels = whisper_n_mels
        self.ds_rate = whisper_ds_rate
        self.cv3_root = cv3_root
        # for at2t*: whisper mel comes from the original noisy wav (the enhanced-token
        # manifest has no wav_path); path = <noisy_wav_root>/<split>/noisy/wav/<utt>.flac
        self.noisy_wav_root = noisy_wav_root
        # for e1n_t2t*: the original noisy CV3 tokens come from a second token root
        # (tokens_root itself holds the enhanced tokens); path built per utt below.
        self.noisy_tok_root = noisy_wav_root  # same data_tokens_synth root holds noisy/tokens

        clean_rows = read_manifest(os.path.join(tokens_root, split, "clean", "manifest.jsonl"))
        self.clean = {r["utt_id"]: r for r in clean_rows}
        self.noisy = manifest_to_dict(
            read_manifest(os.path.join(tokens_root, split, "noisy", "manifest.jsonl"))
        )
        self.utts = [u for u in (r["utt_id"] for r in clean_rows) if u in self.noisy]
        if limit is not None:
            self.utts = self.utts[:limit]

        self.transcripts: Dict[str, str] = {}
        if task_type in _TEXT_OUTPUT:
            if data_root and txt_subdir:  # VB-DEMAND: separate .txt dir
                self.transcripts = load_transcripts(os.path.join(data_root, txt_subdir))
            if not self.transcripts:      # fallback: 'text' field in the clean manifest (libri)
                self.transcripts = {r["utt_id"]: r["text"] for r in clean_rows if r.get("text")}

    def __len__(self) -> int:
        return len(self.utts)

    # ------------------------------------------------------------------ #
    def token_lengths(self) -> List[int]:
        """Per-utt length proxy (noisy CV3 token count) for length bucketing.

        Token count is proportional to audio duration (25 Hz), so it also
        tracks whisper mel frames for the at2t tasks. Values are read from
        the .npy headers once and cached to a json next to the manifest.
        """
        import json as _json

        cache = os.path.join(self.tokens_root, self.split, "noisy", ".lengths_cache.json")
        lens_by_utt: Dict[str, int] = {}
        if os.path.exists(cache):
            try:
                lens_by_utt = _json.load(open(cache))
            except Exception:
                lens_by_utt = {}
        missing = [u for u in self.utts if u not in lens_by_utt]
        if missing:
            from ..data.store import _subset_base

            for u in missing:
                try:
                    row = self.noisy[u]
                    arr = np.load(
                        os.path.join(_subset_base(self.tokens_root, row),
                                     row["token_path"]), mmap_mode="r")
                    lens_by_utt[u] = int(arr.shape[0])
                except Exception:
                    lens_by_utt[u] = 0
            tmp = cache + ".tmp"
            try:  # atomic-ish write; best effort (cache only)
                _json.dump(lens_by_utt, open(tmp, "w"))
                os.replace(tmp, cache)
            except Exception:
                pass
        return [lens_by_utt.get(u, 0) for u in self.utts]

    # ------------------------------------------------------------------ #
    def _prompt_region(self) -> Dict[str, List[int]]:
        v = self.v
        ids = self.tok.encode(self.prompt) if self.prompt else []
        text_row = [v.input_t] + ids + [v.eot]
        audio_row = [v.layershift(v.pad_a)] * len(text_row)
        return {"audio": audio_row, "text": text_row}

    def _build_prefix(self, utt: str):
        """Prompt + input region (no answer). Returns rows, modality span, mel."""
        v = self.v
        pr = self._prompt_region()
        audio_mel = None
        modality_span = None
        if self.task_type in _TOKEN_INPUT:
            noisy_tok = load_tokens(self.tokens_root, self.noisy[utt]).astype(np.int64)
            L = len(noisy_tok)
            a_in = [v.layershift(v.input_a)] + [v.layershift(int(x)) for x in noisy_tok] \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * L + [v.eot, v.answer_t]
        elif self.task_type in _AUDIO_TOKEN_INPUT:
            # FUSE: whisper(noisy) feature region + enhanced-token region in one audio row
            import os as _os
            from .whisper_encoder import log_mel, mel_to_token_len
            from ..utils.audio import load_wav

            wpath = _os.path.join(self.noisy_wav_root, self.split, "noisy", "wav", f"{utt}.flac")
            wav, sr = load_wav(wpath, target_sr=16000)
            mel = log_mel(wav.squeeze(0), n_mels=self.n_mels, cv3_root=self.cv3_root)
            while mel.dim() > 2:
                mel = mel.squeeze(0)
            audio_mel = mel.transpose(0, 1).contiguous()
            Lw = mel_to_token_len(mel.shape[1], self.ds_rate)
            etok = load_tokens(self.tokens_root, self.noisy[utt]).astype(np.int64)
            Lt = len(etok)
            a_in = [v.layershift(v.input_a)] + [v.layershift(v.pad_a)] * Lw \
                + [v.layershift(int(x)) for x in etok] \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * (Lw + Lt) + [v.eot, v.answer_t]
            modality_span = (len(pr["audio"]) + 1, Lw)   # whisper region right after input_a
        elif self.task_type in _DUAL_TOKEN_INPUT:
            etok = load_tokens(self.tokens_root, self.noisy[utt]).astype(np.int64)   # enhanced
            nrow = {"split": self.split, "subset": "noisy",
                    "token_path": f"tokens/{utt}.npy", "save_format": "npy"}
            ntok = load_tokens(self.noisy_tok_root, nrow).astype(np.int64)            # original noisy
            Ln, Le = len(ntok), len(etok)
            a_in = [v.layershift(v.input_a)] + [v.layershift(int(x)) for x in ntok] \
                + [v.layershift(int(x)) for x in etok] \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * (Ln + Le) + [v.eot, v.answer_t]
        else:
            from .whisper_encoder import log_mel, mel_to_token_len
            from ..utils.audio import load_wav

            wav, sr = load_wav(self.noisy[utt]["wav_path"], target_sr=16000)
            mel = log_mel(wav.squeeze(0), n_mels=self.n_mels, cv3_root=self.cv3_root)
            while mel.dim() > 2:
                mel = mel.squeeze(0)
            audio_mel = mel.transpose(0, 1).contiguous()
            L = mel_to_token_len(mel.shape[1], self.ds_rate)
            a_in = [v.layershift(v.input_a)] + [v.layershift(v.pad_a)] * L \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * L + [v.eot, v.answer_t]
            modality_span = (len(pr["audio"]) + 1, L)
        audio_row = pr["audio"] + a_in
        text_row = pr["text"] + t_in
        return audio_row, text_row, modality_span, audio_mel

    def infer_item(self, utt: str) -> Dict:
        """Inference prefix for one utt (no answer/labels): feed to model.generate."""
        audio_row, text_row, span, audio_mel = self._build_prefix(utt)
        T = len(audio_row)
        input_ids = torch.tensor([audio_row, text_row], dtype=torch.long).unsqueeze(0)  # [1,2,T]
        attention_mask = torch.ones(1, T, dtype=torch.long)
        modality_mask = torch.zeros(1, T, dtype=torch.bool)
        if span is not None:
            s, n = span
            modality_mask[0, s:s + n] = True
        item = {"input_ids": input_ids, "attention_mask": attention_mask,
                "modality_mask": modality_mask, "utt_id": utt,
                "noisy_wav": self.noisy[utt].get("wav_path", ""),
                "clean_wav": self.clean[utt].get("wav_path", "")}  # synth: filled from libri map by eval
        if audio_mel is not None:
            item["audio_mel"] = audio_mel.unsqueeze(0)
        return item

    def __getitem__(self, idx: int) -> Dict:
        # robust to a few corrupt files (e.g. truncated flac from a crashed shard):
        # skip forward to the next loadable sample instead of killing training.
        n = len(self.utts)
        for k in range(16):
            j = (idx + k) % n
            try:
                return self._build_item(j)
            except Exception as e:  # noqa: BLE001
                if k == 0:
                    import logging
                    logging.getLogger("se_align.dataset").warning(
                        "skip bad sample %s (%s): %s", j, self.utts[j], e)
        raise RuntimeError(f"too many unreadable samples near idx {idx}")

    def _build_item(self, idx: int) -> Dict:
        v = self.v
        utt = self.utts[idx]
        clean_tok = load_tokens(self.tokens_root, self.clean[utt]).astype(np.int64)

        pr = self._prompt_region()
        a_in: List[int]
        t_in: List[int]
        audio_mel = None
        modality_span = None  # (start, length) of whisper splice within full audio row
        evidence_tok = None   # raw evidence ids, time-aligned to the answer region

        if self.task_type in _TOKEN_INPUT:
            noisy_tok = load_tokens(self.tokens_root, self.noisy[utt]).astype(np.int64)
            evidence_tok = noisy_tok
            L = len(noisy_tok)
            a_in = [v.layershift(v.input_a)] + [v.layershift(int(x)) for x in noisy_tok] \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * L + [v.eot, v.answer_t]
        elif self.task_type in _AUDIO_TOKEN_INPUT:
            # FUSE: whisper(noisy) feature region + enhanced-token region in one audio row
            import os as _os
            from .whisper_encoder import log_mel, mel_to_token_len
            from ..utils.audio import load_wav

            wpath = _os.path.join(self.noisy_wav_root, self.split, "noisy", "wav", f"{utt}.flac")
            wav, sr = load_wav(wpath, target_sr=16000)
            mel = log_mel(wav.squeeze(0), n_mels=self.n_mels, cv3_root=self.cv3_root)
            while mel.dim() > 2:
                mel = mel.squeeze(0)
            audio_mel = mel.transpose(0, 1).contiguous()
            Lw = mel_to_token_len(mel.shape[1], self.ds_rate)
            etok = load_tokens(self.tokens_root, self.noisy[utt]).astype(np.int64)
            Lt = len(etok)
            a_in = [v.layershift(v.input_a)] + [v.layershift(v.pad_a)] * Lw \
                + [v.layershift(int(x)) for x in etok] \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * (Lw + Lt) + [v.eot, v.answer_t]
            modality_span = (len(pr["audio"]) + 1, Lw)
            evidence_tok = etok
        elif self.task_type in _DUAL_TOKEN_INPUT:
            etok = load_tokens(self.tokens_root, self.noisy[utt]).astype(np.int64)   # enhanced
            nrow = {"split": self.split, "subset": "noisy",
                    "token_path": f"tokens/{utt}.npy", "save_format": "npy"}
            ntok = load_tokens(self.noisy_tok_root, nrow).astype(np.int64)            # original noisy
            evidence_tok = etok
            Ln, Le = len(ntok), len(etok)
            a_in = [v.layershift(v.input_a)] + [v.layershift(int(x)) for x in ntok] \
                + [v.layershift(int(x)) for x in etok] \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * (Ln + Le) + [v.eot, v.answer_t]
        else:
            # whisper input: pad_a placeholders that get spliced with encoder feats
            from .whisper_encoder import log_mel, mel_to_token_len
            from ..utils.audio import load_wav

            wav, sr = load_wav(self.noisy[utt]["wav_path"], target_sr=16000)
            mel = log_mel(wav.squeeze(0), n_mels=self.n_mels, cv3_root=self.cv3_root)
            while mel.dim() > 2:                                # CV3 returns [1, n_mels, frames]
                mel = mel.squeeze(0)
            audio_mel = mel.transpose(0, 1).contiguous()       # [frames, n_mels]
            L = mel_to_token_len(mel.shape[1], self.ds_rate)
            a_in = [v.layershift(v.input_a)] + [v.layershift(v.pad_a)] * L \
                + [v.layershift(v.eoa), v.layershift(v.answer_a)]
            t_in = [v.input_t] + [v.pad_t] * L + [v.eot, v.answer_t]
            modality_span = (len(pr["audio"]) + 1, L)  # after prompt + input_a marker

        # ---- answer region ----
        clean_ids = [int(x) for x in clean_tok] + [v.eoa]
        text_target: List[int] = []
        if self.task_type in _TEXT_OUTPUT:
            t = self.transcripts.get(utt, "")
            text_target = self.tok.encode(t) + [v.eot]
        A = max(len(clean_ids), len(text_target))

        # audio answer row (shifted ids for input; raw for labels)
        a_ans_in = [v.layershift(x) for x in clean_ids] + [v.layershift(v.pad_a)] * (A - len(clean_ids))
        a_ans_lab = list(clean_ids) + [IGNORE_INDEX] * (A - len(clean_ids))
        # text answer row
        if text_target:
            t_ans_in = list(text_target) + [v.pad_t] * (A - len(text_target))
            t_ans_lab = list(text_target) + [IGNORE_INDEX] * (A - len(text_target))
        else:
            t_ans_in = [v.pad_t] * A
            t_ans_lab = [IGNORE_INDEX] * A

        # ---- assemble full rows ----
        audio_row = pr["audio"] + a_in + a_ans_in
        text_row = pr["text"] + t_in + t_ans_in
        pre = len(pr["audio"]) + len(a_in)  # prompt+input length (masked)

        audio_lab = [IGNORE_INDEX] * pre + a_ans_lab
        text_lab = [IGNORE_INDEX] * pre + t_ans_lab

        input_ids = torch.tensor([audio_row, text_row], dtype=torch.long)        # [2, T]
        labels = torch.tensor([audio_lab, text_lab], dtype=torch.long)           # [2, T]
        # evidence_ids: raw evidence token aligned to the label time axis
        # (answer-region position k <-> k-th evidence token); -1 = invalid.
        ev_row = [-1] * len(audio_lab)
        if evidence_tok is not None:
            n_clean = len(clean_tok)  # exclude the trailing eoa label position
            for k in range(min(len(evidence_tok), n_clean)):
                ev_row[pre + k] = int(evidence_tok[k])
        evidence_ids = torch.tensor(ev_row, dtype=torch.long)                    # [T]
        attention_mask = torch.ones(input_ids.shape[1], dtype=torch.long)
        modality_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        if modality_span is not None:
            s, n = modality_span
            modality_mask[s:s + n] = True

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "modality_mask": modality_mask,
            "evidence_ids": evidence_ids,
            "utt_id": utt,
        }
        if audio_mel is not None:
            item["audio_mel"] = audio_mel  # [frames, n_mels]
        return item
