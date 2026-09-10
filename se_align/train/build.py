"""Factory helpers: build the Qwen2.5 backbone, encoder, and SEModel."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from .model import SEModel
from .projector import EncoderProjectorConcat
from .vocab import SEVocabConfig

logger = logging.getLogger("se_align.train.build")


def build_tokenizer(llm_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(llm_path)
    return tok


def build_llm(llm_path: str, dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=dtype)


def build_model(
    llm_path: str,
    task_type: str = "t2t",
    whisper_name: str = "large-v3",
    whisper_ds_rate: int = 5,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    freeze_encoder: bool = True,
    encoder: Optional[object] = None,
    whisper_dtype: str = "fp32",
    grad_ckpt: bool = False,
) -> Tuple[SEModel, object, SEVocabConfig]:
    """Build SEModel for a task. ``encoder`` may be injected (e.g. a stub for tests)."""
    vocab = SEVocabConfig()
    tokenizer = build_tokenizer(llm_path)
    llm = build_llm(llm_path, dtype=dtype)
    llm_dim = llm.config.hidden_size

    enc = encoder
    projector = None
    if task_type in ("audio2token", "audio2token_text", "at2t", "at2t_text"):
        if enc is None:
            from .whisper_encoder import load_whisper_encoder

            enc = load_whisper_encoder(whisper_name, device=device)
        enc_dim = getattr(enc, "dim", None) or enc.encoder.ln_post.weight.shape[0]
        projector = EncoderProjectorConcat(enc_dim, llm_dim, ds_rate=whisper_ds_rate)

    model = SEModel(llm, vocab, encoder=enc, encoder_projector=projector,
                    freeze_encoder=freeze_encoder)
    # unify dtype/device (projector is created fresh in fp32; llm/encoder may differ)
    model = model.to(device=device, dtype=dtype)
    # frozen whisper encoder dtype: fp32 (default, safest) or bf16 (halves the
    # dominant per-step cost + activation memory of the whisper tasks). whisper's
    # custom LayerNorm computes in fp32 (casts input .float()), so its weights
    # must STAY fp32 -- convert everything else to bf16.
    if enc is not None:
        if whisper_dtype == "fp32":
            enc.float()
        else:
            enc.to(torch.bfloat16)
            for mod in enc.modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    mod.float()
    if grad_ckpt and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("gradient checkpointing enabled on the LLM")
    return model, tokenizer, vocab
