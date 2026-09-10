"""Evidence-conditioned Qwen generator for target-speaker FSQ tokens."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..train.vocab import SEVocabConfig


IGNORE_INDEX = -100


@dataclass
class TSEForwardOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    labels: torch.Tensor
    sequence_lengths: torch.Tensor


class ConditionProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(values))


class QwenWavLMTSE(nn.Module):
    """Autoregressively predict target S3 tokens from three frozen conditions.

    Prefix layout:

        input_a, speaker, split, WavLM(mixture), split, evidence tokens,
        answer_a, shifted target tokens

    Only target positions are supervised. The output vocabulary is exactly the
    6561 raw CosyVoice3 S3 IDs; target length is supplied by the batch.
    """

    def __init__(self, llm: nn.Module, wavlm: nn.Module,
                 vocab: SEVocabConfig | None = None, speaker_dim: int = 192,
                 freeze_wavlm: bool = True, freeze_qwen: bool = False) -> None:
        super().__init__()
        self.vocab = vocab or SEVocabConfig()
        self.llm = llm
        self.wavlm = wavlm
        self.freeze_wavlm = bool(freeze_wavlm)
        hidden_size = int(llm.config.hidden_size)
        wavlm_dim = int(wavlm.config.hidden_size)

        current_vocab = llm.get_input_embeddings().weight.shape[0]
        if current_vocab != self.vocab.total_vocabsize:
            llm.resize_token_embeddings(self.vocab.total_vocabsize)
        self.mixture_projector = ConditionProjector(wavlm_dim, hidden_size)
        self.speaker_projector = ConditionProjector(speaker_dim, hidden_size)

        if self.freeze_wavlm:
            self.wavlm.requires_grad_(False)
            self.wavlm.eval()
        if freeze_qwen:
            self.llm.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_wavlm:
            self.wavlm.eval()
        return self

    def _audio_embeddings(self, raw_ids: torch.Tensor) -> torch.Tensor:
        combined = raw_ids + self.vocab.audio_shift
        return self.llm.get_input_embeddings()(combined)

    def _marker(self, raw_id: int, device: torch.device) -> torch.Tensor:
        value = torch.tensor([raw_id], dtype=torch.long, device=device)
        return self._audio_embeddings(value)

    def _encode_mixture(self, mixture_values: torch.Tensor,
                        mixture_attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.no_grad() if self.freeze_wavlm else nullcontext()
        with context:
            encoded = self.wavlm(
                input_values=mixture_values.float(),
                attention_mask=mixture_attention_mask,
            ).last_hidden_state
        lengths = mixture_attention_mask.sum(dim=1)
        if hasattr(self.wavlm, "_get_feat_extract_output_lengths"):
            lengths = self.wavlm._get_feat_extract_output_lengths(lengths)
        else:
            lengths = torch.full_like(lengths, encoded.shape[1])
        lengths = lengths.clamp(max=encoded.shape[1]).long()
        target_dtype = self.mixture_projector.linear.weight.dtype
        return self.mixture_projector(encoded.to(target_dtype)), lengths

    def _build_embeddings(self, mixture_features: torch.Tensor,
                          mixture_lengths: torch.Tensor,
                          speaker_embeddings: torch.Tensor,
                          evidence_tokens: torch.Tensor,
                          evidence_lengths: torch.Tensor,
                          target_tokens: torch.Tensor,
                          target_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                torch.Tensor, torch.Tensor]:
        device = mixture_features.device
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        speaker = self.speaker_projector(
            speaker_embeddings.to(self.speaker_projector.linear.weight.dtype)
        ).to(embed_dtype)
        marker_input = self._marker(self.vocab.input_a, device).to(embed_dtype)
        marker_split = self._marker(self.vocab.split, device).to(embed_dtype)
        marker_answer = self._marker(self.vocab.answer_a, device).to(embed_dtype)

        sequences: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        for index in range(mixture_features.shape[0]):
            mix_len = int(mixture_lengths[index])
            evidence_len = int(evidence_lengths[index])
            target_len = int(target_lengths[index])
            evidence = evidence_tokens[index, :evidence_len]
            target = target_tokens[index, :target_len]
            if evidence_len <= 0 or target_len <= 0:
                raise ValueError("evidence and target token sequences must be non-empty")
            evidence_emb = self._audio_embeddings(evidence).to(embed_dtype)
            teacher_ids = target[:-1]
            teacher_emb = self._audio_embeddings(teacher_ids).to(embed_dtype)
            target_prefix = torch.cat([marker_answer, teacher_emb], dim=0)
            prefix = torch.cat(
                [
                    marker_input,
                    speaker[index:index + 1],
                    marker_split,
                    mixture_features[index, :mix_len].to(embed_dtype),
                    marker_split,
                    evidence_emb,
                ],
                dim=0,
            )
            sequence = torch.cat([prefix, target_prefix], dim=0)
            labels = torch.full(
                (sequence.shape[0],), IGNORE_INDEX, dtype=torch.long, device=device
            )
            labels[prefix.shape[0]:] = target
            sequences.append(sequence)
            label_rows.append(labels)

        max_length = max(sequence.shape[0] for sequence in sequences)
        hidden_size = sequences[0].shape[-1]
        inputs = torch.zeros(
            len(sequences), max_length, hidden_size, dtype=embed_dtype, device=device
        )
        labels = torch.full(
            (len(sequences), max_length), IGNORE_INDEX, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            len(sequences), max_length, dtype=torch.long, device=device
        )
        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences], dtype=torch.long, device=device
        )
        for index, (sequence, row_labels) in enumerate(zip(sequences, label_rows)):
            length = sequence.shape[0]
            inputs[index, :length] = sequence
            labels[index, :length] = row_labels
            attention_mask[index, :length] = 1
        return inputs, attention_mask, labels, lengths

    def _build_generation_prefixes(self, mixture_features: torch.Tensor,
                                   mixture_lengths: torch.Tensor,
                                   speaker_embeddings: torch.Tensor,
                                   evidence_tokens: torch.Tensor,
                                   evidence_lengths: torch.Tensor) -> list[torch.Tensor]:
        device = mixture_features.device
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        speaker = self.speaker_projector(
            speaker_embeddings.to(self.speaker_projector.linear.weight.dtype)
        ).to(embed_dtype)
        marker_input = self._marker(self.vocab.input_a, device).to(embed_dtype)
        marker_split = self._marker(self.vocab.split, device).to(embed_dtype)
        marker_answer = self._marker(self.vocab.answer_a, device).to(embed_dtype)
        prefixes = []
        for index in range(mixture_features.shape[0]):
            mix_len = int(mixture_lengths[index])
            evidence_len = int(evidence_lengths[index])
            if evidence_len <= 0:
                raise ValueError("evidence token sequence must be non-empty")
            prefixes.append(torch.cat(
                [
                    marker_input,
                    speaker[index:index + 1],
                    marker_split,
                    mixture_features[index, :mix_len].to(embed_dtype),
                    marker_split,
                    self._audio_embeddings(
                        evidence_tokens[index, :evidence_len]
                    ).to(embed_dtype),
                    marker_answer,
                ],
                dim=0,
            ))
        return prefixes

    @staticmethod
    def _fsq_digits(device: torch.device) -> torch.Tensor:
        ids = torch.arange(6561, dtype=torch.long, device=device)
        digits = []
        for _ in range(8):
            digits.append(ids % 3)
            ids = ids // 3
        return torch.stack(digits, dim=1)

    @torch.no_grad()
    def generate(self, mixture_values: torch.Tensor,
                 mixture_attention_mask: torch.Tensor,
                 speaker_embeddings: torch.Tensor,
                 evidence_tokens: torch.Tensor,
                 evidence_lengths: torch.Tensor,
                 output_length: int | None = None,
                 csg_lambda: float = 0.0,
                 **_: object) -> torch.LongTensor:
        """Batch-1 greedy fixed-length generation in the raw 6561-ID S3 space."""
        if mixture_values.shape[0] != 1:
            raise ValueError("generate requires batch size 1; use generate_batch")
        requested = None if output_length is None else torch.tensor(
            [output_length], dtype=torch.long, device=evidence_lengths.device
        )
        return self.generate_batch(
            mixture_values=mixture_values,
            mixture_attention_mask=mixture_attention_mask,
            speaker_embeddings=speaker_embeddings,
            evidence_tokens=evidence_tokens,
            evidence_lengths=evidence_lengths,
            output_lengths=requested,
            csg_lambda=csg_lambda,
        )[0].unsqueeze(0)

    @torch.no_grad()
    def generate_batch(self, mixture_values: torch.Tensor,
                       mixture_attention_mask: torch.Tensor,
                       speaker_embeddings: torch.Tensor,
                       evidence_tokens: torch.Tensor,
                       evidence_lengths: torch.Tensor,
                       output_lengths: torch.Tensor | None = None,
                       csg_lambda: float = 0.0,
                       **_: object) -> list[torch.LongTensor]:
        """Left-padded batched UD/CSG decode with fixed per-trial lengths."""
        if csg_lambda < 0:
            raise ValueError("csg_lambda must be non-negative")
        self.eval()
        mixture_features, mixture_lengths = self._encode_mixture(
            mixture_values, mixture_attention_mask
        )
        prefixes = self._build_generation_prefixes(
            mixture_features,
            mixture_lengths,
            speaker_embeddings,
            evidence_tokens,
            evidence_lengths,
        )
        lengths = evidence_lengths.long() if output_lengths is None else output_lengths.long()
        if bool((lengths <= 0).any()):
            raise ValueError("all output lengths must be positive")
        if bool((lengths > evidence_lengths).any()):
            raise ValueError("CSG requires evidence for every generated position")
        batch_size = len(prefixes)
        max_prefix = max(prefix.shape[0] for prefix in prefixes)
        hidden_size = prefixes[0].shape[-1]
        prefix_batch = torch.zeros(
            batch_size, max_prefix, hidden_size,
            dtype=prefixes[0].dtype, device=prefixes[0].device,
        )
        attention_mask = torch.zeros(
            batch_size, max_prefix, dtype=torch.long, device=prefixes[0].device
        )
        prefix_lengths = torch.tensor(
            [prefix.shape[0] for prefix in prefixes],
            dtype=torch.long,
            device=prefixes[0].device,
        )
        for index, prefix in enumerate(prefixes):
            prefix_batch[index, max_prefix - prefix.shape[0]:] = prefix
            attention_mask[index, max_prefix - prefix.shape[0]:] = 1
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        output_weight = self.llm.get_output_embeddings().weight
        audio_weight = output_weight[
            self.vocab.audio_shift:self.vocab.audio_shift + self.vocab.audio_vocabsize
        ]
        model_output = self.llm.model(
            inputs_embeds=prefix_batch,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past = model_output.past_key_values
        hidden = model_output.last_hidden_state[:, -1]
        candidate_digits = self._fsq_digits(prefix_batch.device) if csg_lambda else None
        max_length = int(lengths.max())
        generated = torch.zeros(
            batch_size, max_length, dtype=torch.long, device=prefix_batch.device
        )

        for step in range(max_length):
            logits = hidden @ audio_weight.T
            if csg_lambda:
                evidence_ids = evidence_tokens[
                    torch.arange(batch_size, device=prefix_batch.device),
                    torch.minimum(
                        torch.full_like(lengths, step), evidence_lengths.long() - 1
                    ),
                ].long()
                reference_digits = []
                value = evidence_ids
                for _ in range(8):
                    reference_digits.append(value % 3)
                    value = value // 3
                reference = torch.stack(reference_digits, dim=1)
                distance = (
                    candidate_digits.unsqueeze(0) != reference.unsqueeze(1)
                ).sum(dim=2)
                logits = logits - float(csg_lambda) * distance.to(logits.dtype)
            next_raw = logits.argmax(dim=-1)
            generated[:, step] = next_raw
            if step + 1 == max_length:
                break
            next_embedding = self._audio_embeddings(next_raw).unsqueeze(1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        batch_size, 1, dtype=torch.long, device=prefix_batch.device
                    ),
                ],
                dim=1,
            )
            position_ids = (prefix_lengths + step).unsqueeze(1)
            model_output = self.llm.model(
                inputs_embeds=next_embedding,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )
            past = model_output.past_key_values
            hidden = model_output.last_hidden_state[:, -1]
        results = [
            generated[index, :int(lengths[index])]
            for index in range(batch_size)
        ]
        for result in results:
            if result.min() < 0 or result.max() >= self.vocab.audio_vocabsize:
                raise RuntimeError("generation produced an invalid raw S3 token")
        return results

    def forward(self, mixture_values: torch.Tensor,
                mixture_attention_mask: torch.Tensor,
                speaker_embeddings: torch.Tensor,
                evidence_tokens: torch.Tensor,
                evidence_lengths: torch.Tensor,
                target_tokens: torch.Tensor,
                target_lengths: torch.Tensor,
                **_: object) -> TSEForwardOutput:
        mixture_features, mixture_lengths = self._encode_mixture(
            mixture_values, mixture_attention_mask
        )
        inputs, attention_mask, labels, sequence_lengths = self._build_embeddings(
            mixture_features,
            mixture_lengths,
            speaker_embeddings,
            evidence_tokens,
            evidence_lengths,
            target_tokens,
            target_lengths,
        )
        hidden = self.llm.model(
            inputs_embeds=inputs, attention_mask=attention_mask
        ).last_hidden_state
        output_weight = self.llm.get_output_embeddings().weight
        audio_weight = output_weight[
            self.vocab.audio_shift:self.vocab.audio_shift + self.vocab.audio_vocabsize
        ]
        logits = hidden @ audio_weight.T
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab.audio_vocabsize),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return TSEForwardOutput(loss, logits, labels, sequence_lengths)
