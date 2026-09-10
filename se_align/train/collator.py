"""Batch collator: right-pads the 2-row token grids + whisper mels."""
from __future__ import annotations

from typing import Dict, List

import torch

from .vocab import SEVocabConfig

IGNORE_INDEX = -100


class SECollator:
    def __init__(self, vocab: SEVocabConfig) -> None:
        self.v = vocab

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        v = self.v
        T = max(b["input_ids"].shape[1] for b in batch)
        audio_pad = v.layershift(v.pad_a)
        text_pad = v.pad_t

        input_ids, labels, attn, modality, evidence = [], [], [], [], []
        has_ev = any("evidence_ids" in b for b in batch)
        has_mel = any("audio_mel" in b for b in batch)
        mels, mel_lens = [], []
        Fmax = max((b["audio_mel"].shape[0] for b in batch if "audio_mel" in b), default=0)
        n_mels = next((b["audio_mel"].shape[1] for b in batch if "audio_mel" in b), 0)

        for b in batch:
            t = b["input_ids"].shape[1]
            pad = T - t
            row_pad = torch.tensor([[audio_pad] * pad, [text_pad] * pad], dtype=torch.long)
            input_ids.append(torch.cat([b["input_ids"], row_pad], dim=1))
            labels.append(torch.cat(
                [b["labels"], torch.full((2, pad), IGNORE_INDEX, dtype=torch.long)], dim=1))
            attn.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            modality.append(torch.cat([b["modality_mask"], torch.zeros(pad, dtype=torch.bool)]))
            if has_ev:
                ev = b.get("evidence_ids", torch.full((t,), -1, dtype=torch.long))
                evidence.append(torch.cat([ev, torch.full((pad,), -1, dtype=torch.long)]))
            if has_mel:
                if "audio_mel" in b:
                    m = b["audio_mel"]
                    mel_lens.append(m.shape[0])
                    if m.shape[0] < Fmax:
                        m = torch.cat([m, torch.zeros(Fmax - m.shape[0], n_mels)], dim=0)
                    mels.append(m)
                else:
                    mels.append(torch.zeros(Fmax, n_mels))
                    mel_lens.append(0)

        out = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "modality_mask": torch.stack(modality),
        }
        if has_ev:
            out["evidence_ids"] = torch.stack(evidence)

        if has_mel:
            out["audio_mel"] = torch.stack(mels)
            out["audio_mel_lengths"] = torch.tensor(mel_lens, dtype=torch.long)
        return out
