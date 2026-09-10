"""Factories for the Qwen/WavLM TSE model."""

from __future__ import annotations

import torch

from ..train.vocab import SEVocabConfig
from .model import QwenWavLMTSE


def build_tse_model(qwen_path: str, wavlm_path: str, device: str = "cuda",
                    dtype: torch.dtype = torch.bfloat16,
                    freeze_wavlm: bool = True,
                    freeze_qwen: bool = False,
                    local_files_only: bool = True) -> QwenWavLMTSE:
    from transformers import AutoModelForCausalLM, WavLMModel

    llm = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    wavlm = WavLMModel.from_pretrained(
        wavlm_path,
        local_files_only=local_files_only,
    )
    model = QwenWavLMTSE(
        llm=llm,
        wavlm=wavlm,
        vocab=SEVocabConfig(),
        speaker_dim=192,
        freeze_wavlm=freeze_wavlm,
        freeze_qwen=freeze_qwen,
    )
    model.llm.to(device=device, dtype=dtype)
    model.mixture_projector.to(device=device, dtype=dtype)
    model.speaker_projector.to(device=device, dtype=dtype)
    model.wavlm.to(device=device, dtype=torch.float32)
    return model
