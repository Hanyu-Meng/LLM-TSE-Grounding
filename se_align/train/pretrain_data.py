"""Dataset for text -> semantic-token pretraining (TTS-style) on LibriSpeech.

Warms up the newly-inserted CV3 audio-token embeddings (+ optional LoRA) using a
large clean corpus, before the noisy->clean SE finetune. Self-supervised w.r.t.
the SE task (no noisy/clean pairs), so it does not add SE supervision.

Grid (2 rows [audio, text]), laid out [INPUT(text) | ANSWER(clean tokens)]:
    text row  : [input_t]  text_ids   [eot, answer_t]   pad_t ...
    audio row : [input_a]  pad_a * Lt [eoa, answer_a]   clean_tokens  eoa
Only the audio answer is supervised (predict clean CV3 tokens from the text).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.store import read_manifest
from .vocab import SEVocabConfig

IGNORE_INDEX = -100


class PretrainCorrupt2CleanDataset(Dataset):
    """Self-supervised denoising: randomly-corrupted clean CV3 tokens -> clean
    tokens + transcript. No whisper, no real noise. Corruption (random token
    substitution) mimics noisy-token errors; re-sampled each access (augmentation).

    Grid (2 rows [audio, text]), [INPUT(corrupted tokens) | ANSWER(clean tokens)]:
        audio: [input_a] corrupted_tok* [eoa, answer_a]   clean_tok  eoa
        text : [input_t] pad_t*L        [eot, answer_t]   transcript eot
    Both audio and text answers are supervised.
    """

    def __init__(self, manifest_path, tokens_root, vocab: SEVocabConfig, tokenizer,
                 corruption_ratio: float = 0.25, limit=None, max_text_tokens: int = 400):
        self.tokens_root = tokens_root
        self.v = vocab
        self.tok = tokenizer
        self.p = corruption_ratio
        self.max_text_tokens = max_text_tokens
        self.rows = read_manifest(manifest_path)
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def _corrupt(self, clean: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()  # fresh each call -> different corruption per epoch
        x = clean.copy()
        n = len(x)
        k = int(round(n * self.p))
        if k > 0:
            pos = rng.choice(n, size=k, replace=False)
            x[pos] = rng.integers(0, self.v.audio_vocabsize, size=k)  # random valid CV3 tokens
        return x

    def __getitem__(self, idx):
        v = self.v
        row = self.rows[idx]
        clean = np.load(os.path.join(self.tokens_root, row["token_path"])).astype(np.int64)
        corrupted = self._corrupt(clean)
        text_ids = self.tok.encode(row.get("text", ""))[: self.max_text_tokens]

        L = len(corrupted)
        a_in = [v.layershift(v.input_a)] + [v.layershift(int(x)) for x in corrupted] \
            + [v.layershift(v.eoa), v.layershift(v.answer_a)]
        t_in = [v.input_t] + [v.pad_t] * L + [v.eot, v.answer_t]

        clean_ids = [int(x) for x in clean] + [v.eoa]
        text_target = list(text_ids) + [v.eot]
        A = max(len(clean_ids), len(text_target))
        a_ans_in = [v.layershift(x) for x in clean_ids] + [v.layershift(v.pad_a)] * (A - len(clean_ids))
        a_ans_lab = list(clean_ids) + [IGNORE_INDEX] * (A - len(clean_ids))
        t_ans_in = list(text_target) + [v.pad_t] * (A - len(text_target))
        t_ans_lab = list(text_target) + [IGNORE_INDEX] * (A - len(text_target))

        audio_row = a_in + a_ans_in
        text_row = t_in + t_ans_in
        pre = len(a_in)
        audio_lab = [IGNORE_INDEX] * pre + a_ans_lab
        text_lab = [IGNORE_INDEX] * pre + t_ans_lab

        input_ids = torch.tensor([audio_row, text_row], dtype=torch.long)
        labels = torch.tensor([audio_lab, text_lab], dtype=torch.long)
        return {
            "input_ids": input_ids, "labels": labels,
            "attention_mask": torch.ones(input_ids.shape[1], dtype=torch.long),
            "modality_mask": torch.zeros(input_ids.shape[1], dtype=torch.bool),
            "utt_id": row["utt_id"],
        }


class PretrainTextToTokenDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tokens_root: str,
        vocab: SEVocabConfig,
        tokenizer,
        prompt: str = "",
        limit: Optional[int] = None,
        max_text_tokens: int = 400,
    ) -> None:
        self.tokens_root = tokens_root
        self.v = vocab
        self.tok = tokenizer
        self.prompt = prompt
        self.max_text_tokens = max_text_tokens
        self.rows = read_manifest(manifest_path)
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        v = self.v
        row = self.rows[idx]
        clean = np.load(os.path.join(self.tokens_root, row["token_path"])).astype(np.int64)
        text_ids = self.tok.encode(row["text"])[: self.max_text_tokens]

        # optional generic prompt region (text only)
        pa: List[int] = []
        pt: List[int] = []
        if self.prompt:
            pid = self.tok.encode(self.prompt)
            pt = [v.input_t] + pid + [v.eot]
            pa = [v.layershift(v.pad_a)] * len(pt)

        Lt = len(text_ids)
        # INPUT region: text carries the transcript, audio row is pad
        t_in = [v.input_t] + text_ids + [v.eot, v.answer_t]
        a_in = [v.layershift(v.input_a)] + [v.layershift(v.pad_a)] * Lt \
            + [v.layershift(v.eoa), v.layershift(v.answer_a)]

        # ANSWER region: audio = clean tokens (+eoa); text = pad
        clean_ids = [int(x) for x in clean] + [v.eoa]
        a_ans_in = [v.layershift(x) for x in clean_ids]
        a_ans_lab = list(clean_ids)
        t_ans = [v.pad_t] * len(clean_ids)

        audio_row = pa + a_in + a_ans_in
        text_row = pt + t_in + t_ans
        pre = len(pa) + len(a_in)
        audio_lab = [IGNORE_INDEX] * pre + a_ans_lab
        text_lab = [IGNORE_INDEX] * len(text_row)

        input_ids = torch.tensor([audio_row, text_row], dtype=torch.long)
        labels = torch.tensor([audio_lab, text_lab], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(input_ids.shape[1], dtype=torch.long),
            "modality_mask": torch.zeros(input_ids.shape[1], dtype=torch.bool),
            "utt_id": row["utt_id"],
        }
