"""Whisper encoder wrapper for the noisy-audio input path (cases 2 & 3).

Wraps the official openai-whisper encoder (same weights EchoMind uses) and adds
``extract_variable_length_features`` (slices the positional embedding to the real
length instead of padding to 30 s), matching EchoMind's vendored AudioEncoder.

Requires ``openai-whisper`` (``pip install openai-whisper``). Only needed for the
whisper-input cases — the token->token case (case 1) does not import this.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, n_mels: int, dim: int) -> None:
        super().__init__()
        self.encoder = encoder           # whisper AudioEncoder
        self.n_mels = n_mels
        self.dim = dim                   # n_audio_state (output feature dim)

    @torch.no_grad()
    def extract_variable_length_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, n_mels, T]`` log-mel -> ``[B, T//2, dim]`` features."""
        enc = self.encoder
        x = F.gelu(enc.conv1(x))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + enc.positional_embedding[: x.shape[1]]).to(x.dtype)
        for block in enc.blocks:
            x = block(x)
        return enc.ln_post(x)


def load_whisper_encoder(name_or_path: str = "large-v3", device: str = "cpu") -> WhisperEncoderWrapper:
    """Load the whisper encoder (downloads/caches via openai-whisper if a name)."""
    import whisper

    model = whisper.load_model(name_or_path, device=device)
    enc = model.encoder
    n_mels = model.dims.n_mels
    dim = model.dims.n_audio_state
    return WhisperEncoderWrapper(enc.to(device), n_mels=n_mels, dim=dim).to(device)


def log_mel(wav16: torch.Tensor, n_mels: int = 128, cv3_root: str | None = None) -> torch.Tensor:
    """16 kHz mono waveform ``[T]`` -> whisper-style log-mel ``[n_mels, frames]``.

    Uses CosyVoice3's own ``log_mel_spectrogram`` (identical normalisation to
    whisper: clamp->log10->max-8->(x+4)/4), so no openai-whisper dependency is
    needed for feature extraction.
    """
    import sys

    if cv3_root and cv3_root not in sys.path:
        sys.path.insert(0, cv3_root)
    from cosyvoice_tokenizer.features import log_mel_spectrogram

    return log_mel_spectrogram(wav16, n_mels=n_mels, prefer_whisper=False)


def mel_to_token_len(n_mel_frames: int, ds_rate: int = 5) -> int:
    """Whisper conv2 halves the frame rate, then the projector downsamples by k."""
    return ((n_mel_frames + 1) // 2) // ds_rate
