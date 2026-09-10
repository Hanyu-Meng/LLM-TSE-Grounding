"""Unit tests for the TSE data contract and multimodal model assembly."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from se_align.data.tse_dataset import TSECollator, TSEManifestDataset
from se_align.train.vocab import SEVocabConfig
from se_align.tse.model import QwenWavLMTSE


class FakeWavLM(torch.nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_values, attention_mask=None):
        values = input_values[:, ::2, None].repeat(1, 1, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=values)

    @staticmethod
    def _get_feat_extract_output_lengths(lengths):
        return (lengths + 1) // 2


class FakeBody(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.linear(inputs_embeds))


class FakeLLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.model = FakeBody(hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size: int):
        old = self.embedding
        self.embedding = torch.nn.Embedding(size, old.embedding_dim)
        with torch.no_grad():
            self.embedding.weight[:min(size, old.num_embeddings)].copy_(
                old.weight[:min(size, old.num_embeddings)]
            )
        return self.embedding


def test_tse_dataset_and_collator(tmp_path):
    wav_path = tmp_path / "mix.wav"
    sf.write(wav_path, np.zeros(320, dtype=np.float32), 16000)
    evidence = tmp_path / "evidence.npy"
    target = tmp_path / "target.npy"
    speaker = tmp_path / "speaker.npy"
    np.save(evidence, np.array([1, 2, 3], dtype=np.int32))
    np.save(target, np.array([4, 5], dtype=np.int32))
    np.save(speaker, np.ones(192, dtype=np.float32))
    manifest = tmp_path / "prepared.jsonl"
    manifest.write_text(json.dumps({
        "trial_id": "trial-a",
        "mixture_wav": str(wav_path),
        "evidence_token_path": str(evidence),
        "target_token_path": str(target),
        "speaker_embedding_path": str(speaker),
    }) + "\n")
    dataset = TSEManifestDataset(manifest)
    batch = TSECollator()([dataset[0]])
    assert batch["mixture_values"].shape == (1, 320)
    assert batch["evidence_tokens"].tolist() == [[1, 2, 3]]
    assert batch["target_lengths"].tolist() == [2]
    assert batch["speaker_embeddings"].shape == (1, 192)


def test_tse_forward_is_finite_and_uses_raw_fsq_vocab():
    vocab = SEVocabConfig(
        text_vocabsize=32,
        text_specialtokens=8,
        audio_vocabsize=27,
        audio_specialtokens=8,
    )
    model = QwenWavLMTSE(
        FakeLLM(vocab.total_vocabsize),
        FakeWavLM(),
        vocab=vocab,
        speaker_dim=4,
        freeze_wavlm=True,
    )
    output = model(
        mixture_values=torch.randn(2, 20),
        mixture_attention_mask=torch.tensor([[1] * 20, [1] * 16 + [0] * 4]),
        speaker_embeddings=torch.randn(2, 4),
        evidence_tokens=torch.tensor([[1, 2, 3], [4, 5, -1]]),
        evidence_lengths=torch.tensor([3, 2]),
        target_tokens=torch.tensor([[6, 7, 8], [9, 10, -1]]),
        target_lengths=torch.tensor([3, 2]),
    )
    assert output.logits.shape[-1] == 27
    assert torch.isfinite(output.logits).all()
    assert output.loss is not None and torch.isfinite(output.loss)
    assert not any(parameter.requires_grad for parameter in model.wavlm.parameters())
