"""Audio IO + the single, high-quality resampler used everywhere.

All 24k<->16k (and 48k->16k) conversions go through :func:`resample` so the
whole project is consistent. Uses torchaudio's Kaldi-style sinc resampler.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


@lru_cache(maxsize=16)
def _resampler(orig_sr: int, new_sr: int) -> torchaudio.transforms.Resample:
    # High-quality sinc interpolation (Kaiser window), cached per (sr,sr) pair.
    return torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=new_sr,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        beta=14.769656459379492,
    )


def resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Resample ``[..., T]`` float waveform. No-op when rates match."""
    if orig_sr == new_sr:
        return wav
    return _resampler(int(orig_sr), int(new_sr))(wav)


def load_wav(path: str, target_sr: int | None = None, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """Load a wav as float32 ``[1, T]``; optionally resample to ``target_sr``.

    Returns ``(wav, sr)`` where ``sr`` is the (possibly resampled) rate.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    wav = torch.from_numpy(data).transpose(0, 1).contiguous()  # [C, T]
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav.contiguous(), sr


def save_wav(path: str, wav: torch.Tensor, sr: int) -> None:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    data = wav.detach().cpu().float().transpose(0, 1).numpy()  # [T, C]
    sf.write(path, data, sr)


def align_lengths(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncate two ``[..., T]`` waveforms to their common length."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]
