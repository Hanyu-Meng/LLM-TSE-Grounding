"""Encoder->LLM projector (copied from EchoMind models/projector.py).

Stacks ``k`` consecutive encoder frames and projects to the LLM hidden size,
downsampling the audio feature rate by ``k`` (Whisper 50 Hz -> 10 Hz at k=5).
"""
from __future__ import annotations

import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, encoder_dim: int, llm_dim: int, ds_rate: int = 5) -> None:
        super().__init__()
        self.k = ds_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        x = x.contiguous().view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
