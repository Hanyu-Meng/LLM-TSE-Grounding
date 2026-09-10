"""Phase-2 model: Qwen2.5-Instruct backbone + (optional) Whisper encoder, emitting
parallel CV3-audio-token and text streams.

Adapted from EchoMind ``S2SModel`` (models/s2s.py), specialised to CV3
(``code_layer = 1``) and a stock ``transformers`` Qwen2.5 backbone (resized to the
combined text+audio vocab). The three training cases are handled by one forward:

  case 1  (t2t)            : noisy CV3 tokens  -> clean CV3 tokens         (no encoder, no text loss)
  case 2  (audio2token)    : whisper(noisy)+text prompt -> clean CV3 tokens (encoder, no text loss)
  case 3  (audio2token_text): whisper(noisy)+text prompt -> clean tokens + transcript (encoder + text loss)

Mechanics (mirrors EchoMind):
  * input_ids: [B, code_layer+1, T]  -> rows [audio_0 .. , text]
  * embed all rows in one resized table, splice whisper features into the audio
    row(s) via ``modality_mask``, then average over the (code_layer+1) rows.
  * single LM head -> logits split into a text slice + one audio slice; parallel
    cross-entropy, averaged over the supervised streams.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import SEVocabConfig

logger = logging.getLogger("se_align.train.model")
IGNORE_INDEX = -100


class SEModel(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        vocab: SEVocabConfig,
        encoder: Optional[nn.Module] = None,
        encoder_projector: Optional[nn.Module] = None,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.vocab = vocab
        self.code_layer = vocab.code_layer
        self.encoder = encoder
        self.encoder_projector = encoder_projector
        self.freeze_encoder = freeze_encoder
        if encoder is not None and freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False

        # resize embedding + head to the combined vocab
        cur = self.llm.get_input_embeddings().weight.size(0)
        if cur != vocab.total_vocabsize:
            self.llm.resize_token_embeddings(vocab.total_vocabsize)
            logger.info("resized Qwen embeddings %d -> %d", cur, vocab.total_vocabsize)

    # ------------------------------------------------------------------ #
    def _embed_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
        emb = self.llm.get_input_embeddings()
        return emb(input_ids)

    def _build_inputs_embeds(self, input_ids, audio_mel, modality_mask):
        """[B, L+1, T] grid (+ optional whisper) -> averaged [B, T, D] embeds."""
        encoder_outs = None
        if audio_mel is not None and self.encoder is not None:
            if self.freeze_encoder:
                self.encoder.eval()
            # Whisper runs in fp32 (its LayerNorm casts internally); feed fp32 mel,
            # then cast features to the projector's dtype.
            enc_dtype = next(self.encoder.parameters()).dtype
            proj_dtype = self.encoder_projector.linear1.weight.dtype
            with torch.set_grad_enabled(not self.freeze_encoder):
                encoder_outs = self.encoder.extract_variable_length_features(
                    audio_mel.to(enc_dtype).permute(0, 2, 1)
                )
            encoder_outs = self.encoder_projector(encoder_outs.to(proj_dtype))

        input_ids = input_ids.clone()
        input_ids[input_ids == -1] = 0
        inputs_embeds = self._embed_tokens(input_ids)  # [B, L+1, T, D]

        if modality_mask is not None and encoder_outs is not None:
            mm = modality_mask.unsqueeze(1).repeat(1, self.code_layer, 1)  # [B, L, T]
            starts = (mm == True).float().argmax(dim=2)
            lengths = torch.clamp(mm.sum(dim=2), max=encoder_outs.shape[1]).tolist()
            pad = torch.zeros_like(inputs_embeds)
            for b in range(encoder_outs.shape[0]):
                for j in range(self.code_layer):
                    s = int(starts[b, j].item())
                    n = int(lengths[b][j])
                    pad[b, j, s:s + n] = encoder_outs[b, :n]
            inputs_embeds[:, : self.code_layer] = (
                pad[:, : self.code_layer]
                + inputs_embeds[:, : self.code_layer] * (~mm[:, :, :, None])
            )
        return inputs_embeds.mean(dim=1)  # [B, T, D]

    def set_trust_region(self, beta: float, neighbor_ids: torch.LongTensor,
                         neighbor_mask: torch.BoolTensor, reliable_only: bool = True,
                         reliable_radius: int = 2) -> None:
        """Enable the auxiliary evidence trust-region loss (training-time only).
        L = L_CE + beta * L_TR,  L_TR = -mean_t log sum_{v in B_r(z^e_t)} p_t(v).
        Buffers are registered so .to(device) moves them with the model."""
        self.tr_beta = float(beta)
        self.tr_reliable_only = bool(reliable_only)
        self.tr_reliable_radius = int(reliable_radius)
        self.register_buffer("tr_neighbor_ids", neighbor_ids, persistent=False)
        self.register_buffer("tr_neighbor_mask", neighbor_mask, persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_mel: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        evidence_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self._build_inputs_embeds(input_ids, audio_mel, modality_mask)

        pt = self.vocab.padded_text_vocabsize
        pa = self.vocab.padded_audio_vocabsize
        text_labels = audio_labels = None
        if labels is not None:
            text_labels = labels[:, self.code_layer]          # [B, T]
            audio_labels = labels[:, : self.code_layer]        # [B, L, T]

        # For tasks with no text supervision (t2t, at2t, ...) skip the 152k-wide
        # text head: run the bare decoder and project onto the audio rows of the
        # lm_head only ([B,T,6625] instead of [B,T,158625] -> big memory cut).
        skip_text_head = (
            text_labels is not None
            and not bool((text_labels[:, 1:] != IGNORE_INDEX).any())
            and hasattr(self.llm, "model") and self.llm.get_output_embeddings() is not None
        )
        if skip_text_head:
            hidden = self.llm.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            ).last_hidden_state                                # [B, T, D]
            w = self.llm.get_output_embeddings().weight        # [V, D]
            xa = [hidden @ w[pt + i * pa: pt + (i + 1) * pa].T
                  for i in range(self.code_layer)]
            xt, logits = None, None
        else:
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            logits = out.logits  # [B, T, total_vocab]
            xt = logits[..., :pt]
            xa = [logits[..., pt + i * pa: pt + (i + 1) * pa] for i in range(self.code_layer)]

        loss = None
        layer_loss = None
        tr_stats = None
        if labels is not None:
            loss, layer_loss = self._parallel_loss(xt, text_labels, xa, audio_labels)
            if (getattr(self, "tr_beta", 0.0) > 0.0 and evidence_ids is not None
                    and audio_labels is not None):
                l_tr, tr_stats = self._trust_region_loss(
                    xa[0], audio_labels[:, 0], evidence_ids)
                if l_tr is not None:
                    tr_stats["loss_ce"] = float(loss.detach())
                    loss = loss + self.tr_beta * l_tr
                    tr_stats["total_loss"] = float(loss.detach())

        return {"loss": loss, "logits": logits, "xt": xt, "xa": xa,
                "layer_loss": layer_loss, "tr_stats": tr_stats}

    def _trust_region_loss(self, xa0, audio_labels, evidence_ids):
        """Auxiliary evidence trust-region loss on the audio head.

        L_TR = -mean_t log sum_{v in B_r(z^e_t)} p_t(v) over valid (and, if
        enabled, evidence-reliable) frames. Operates on the raw-audio-id
        indexed head slice xa0 [B, T, padded_audio_vocabsize] — neighbor ids
        gather directly, no tokenizer-id mapping needed. Uses the same
        next-token shift as the CE loss (predict position t from t-1).
        """
        from .fsq_neighbors import hamming_digits
        V = self.vocab.audio_vocabsize                      # 6561
        logp = torch.log_softmax(xa0[:, :-1, :].float(), dim=-1)   # [B, T-1, pa]
        ev = evidence_ids[:, 1:]                            # [B, T-1]
        lab = audio_labels[:, 1:]
        valid = (ev >= 0) & (ev < V) & (lab >= 0) & (lab < V)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return None, None
        reliable = valid
        if getattr(self, "tr_reliable_only", False):
            ham = hamming_digits(lab.clamp(0, V - 1), ev.clamp(0, V - 1))
            reliable = valid & (ham <= self.tr_reliable_radius)
            if not bool(reliable.any()):
                return None, {"reliable_frame_ratio": 0.0}
        nbr = self.tr_neighbor_ids[ev.clamp(0, V - 1)]      # [B, T-1, K]
        nlp = torch.gather(logp, -1, nbr)
        nlp = nlp.masked_fill(~self.tr_neighbor_mask[ev.clamp(0, V - 1)], float("-inf"))
        log_mass = torch.logsumexp(nlp, dim=-1)             # [B, T-1]
        l_tr = -(log_mass[reliable]).mean()
        stats = {
            "loss_trust_region": float(l_tr.detach()),
            "trust_region_mass": float(log_mass[reliable].detach().exp().mean()),
            "reliable_frame_ratio": float(reliable.sum()) / max(n_valid, 1),
        }
        return l_tr, stats

    def _parallel_loss(self, xt, text_labels, xa, audio_labels):
        """CE per supervised stream; total = mean over active streams."""
        pt = self.vocab.padded_text_vocabsize
        pa = self.vocab.padded_audio_vocabsize
        dev = xt.device if xt is not None else xa[0].device
        active: List[torch.Tensor] = []
        layer_loss = [torch.tensor(0.0, device=dev) for _ in range(self.code_layer + 1)]

        if xt is not None and text_labels is not None and (text_labels[:, 1:] != IGNORE_INDEX).any():
            tl = F.cross_entropy(
                xt[:, :-1, :].reshape(-1, pt), text_labels[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            layer_loss[self.code_layer] = tl
            active.append(tl)

        for i in range(self.code_layer):
            ai = audio_labels[:, i]
            if (ai[:, 1:] != IGNORE_INDEX).any():
                al = F.cross_entropy(
                    xa[i][:, :-1, :].reshape(-1, pa), ai[:, 1:].reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                layer_loss[i] = al
                active.append(al)

        total = sum(active) / max(len(active), 1)
        return total, layer_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        audio_mel: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        text_output: bool = False,
        return_text: bool = False,
    ):
        """Dual-stream autoregressive decode (EchoMind s2s style).

        The audio and text streams end *independently*: each is sampled until its
        end token (eoa / eot), after which it is fed its pad token so the other
        stream stays on-distribution; we stop once both have ended. For audio-only
        tasks (``text_output=False``) the text row is pad_t throughout (its head is
        untrained), matching how those tasks were trained. Feeding a *running* text
        stream that never ends (the previous bug) corrupts the audio stream and
        causes runaway generation. Returns ``(audio_ids[raw], text_ids or None)``.
        """
        assert input_ids.shape[0] == 1, "generate supports batch size 1"
        v = self.vocab
        pt, pa = v.padded_text_vocabsize, v.padded_audio_vocabsize
        dev = input_ids.device

        embeds = self._build_inputs_embeds(input_ids, audio_mel, modality_mask)  # [1, T, D]
        out = self.llm(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]  # [1, total_vocab]

        audio_ids: List[int] = []
        text_ids: List[int] = []
        text_end = not text_output   # audio-only: text row is pad_t the whole time
        audio_end = False
        for _ in range(max_new_tokens):
            # --- text stream ---
            if not text_end:
                t_raw = int(logits[:, :pt].argmax(-1).item())
                if t_raw == v.eot:
                    text_end, t_in = True, v.pad_t
                else:
                    text_ids.append(t_raw)
                    t_in = t_raw
            else:
                t_in = v.pad_t
            # --- audio stream ---
            if not audio_end:
                a_raw = int(logits[:, pt:pt + pa].argmax(-1).item())
                if a_raw == v.eoa:
                    audio_end, a_in = True, v.pad_a
                else:
                    audio_ids.append(a_raw)
                    a_in = a_raw
            else:
                a_in = v.pad_a
            if audio_end and text_end:
                break
            # --- feed both streams back (mean of shifted-audio + text embeddings) ---
            a_emb = self._embed_tokens(torch.tensor([v.layershift(a_in)], device=dev))
            t_emb = self._embed_tokens(torch.tensor([t_in], device=dev))
            nxt = ((a_emb + t_emb) / 2).unsqueeze(1)               # [1, 1, D]
            out = self.llm(inputs_embeds=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]
        return audio_ids, (text_ids if return_text else None)

    @torch.no_grad()
    def generate_batch(
        self,
        items: List[dict],
        max_new_tokens: int = 1024,
        text_output: bool = False,
        return_text: bool = False,
    ):
        """Batched greedy dual-stream decode; exact parity with ``generate``.

        Each item: {"input_ids" [2,T], optional "audio_mel" [F,n_mels],
        "modality_mask" [T]}. Prefix embeddings are built PER ITEM (batch-1
        whisper forward -> no padding leakage, bit-identical prefixes), then
        LEFT-padded and decoded together with explicit position_ids.
        Returns ``[(audio_ids, text_ids|None), ...]`` in item order.
        """
        v = self.vocab
        pt, pa = v.padded_text_vocabsize, v.padded_audio_vocabsize
        dev = next(self.parameters()).device

        embeds = []
        for it in items:
            ii = it["input_ids"].unsqueeze(0).to(dev)
            am = it.get("audio_mel")
            mm = it.get("modality_mask")
            if am is not None:
                am = am.unsqueeze(0).to(dev)
            if mm is not None:
                mm = mm.unsqueeze(0).to(dev)
            embeds.append(self._build_inputs_embeds(ii, am, mm)[0])  # [T, D]
        B = len(embeds)
        Tm = max(e.shape[0] for e in embeds)
        x = torch.zeros(B, Tm, embeds[0].shape[-1], dtype=embeds[0].dtype, device=dev)
        mask = torch.zeros(B, Tm, dtype=torch.long, device=dev)
        for b, e in enumerate(embeds):  # left-pad: last prefix token rightmost
            x[b, Tm - e.shape[0]:] = e
            mask[b, Tm - e.shape[0]:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.llm(inputs_embeds=x, attention_mask=mask, position_ids=pos, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        lens = mask.sum(-1)                                       # true prefix lengths

        audio = [[] for _ in range(B)]
        text = [[] for _ in range(B)]
        a_end = torch.zeros(B, dtype=torch.bool, device=dev)
        t_end = torch.full((B,), not text_output, dtype=torch.bool, device=dev)
        for step in range(max_new_tokens):
            t_tok = logits[:, :pt].argmax(-1)                     # [B]
            a_tok = logits[:, pt:pt + pa].argmax(-1)              # [B]
            new_t_end = (~t_end) & (t_tok == v.eot)
            for b in torch.nonzero((~t_end) & (~new_t_end)).flatten().tolist():
                text[b].append(int(t_tok[b]))
            t_end = t_end | new_t_end
            t_in = torch.where(t_end, torch.full_like(t_tok, v.pad_t), t_tok)
            new_a_end = (~a_end) & (a_tok == v.eoa)
            for b in torch.nonzero((~a_end) & (~new_a_end)).flatten().tolist():
                audio[b].append(int(a_tok[b]))
            a_end = a_end | new_a_end
            a_in = torch.where(a_end, torch.full_like(a_tok, v.pad_a), a_tok)
            if bool((a_end & t_end).all()):
                break
            a_emb = self._embed_tokens(a_in + v.audio_shift)      # layershift, layer 0
            t_emb = self._embed_tokens(t_in)
            nxt = ((a_emb + t_emb) / 2).unsqueeze(1)              # [B, 1, D]
            mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
            out = self.llm(
                inputs_embeds=nxt, attention_mask=mask,
                position_ids=(lens + step).unsqueeze(1),
                past_key_values=past, use_cache=True,
            )
            past, logits = out.past_key_values, out.logits[:, -1]
        return [(audio[b], (text[b] if return_text else None)) for b in range(B)]
