#!/usr/bin/env python3
"""Run one real Q-Full batch to verify trainable S3 and projector parameters."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tse.train_tse import (  # noqa: E402
    move_batch,
    optimizer_for,
    parameter_grad_norm,
    s3_grad_norm,
    s3_parameters,
    set_seed,
)
from se_align.data.tse_dataset import TSECollator, TSEManifestDataset  # noqa: E402
from se_align.tse import build_tse_model  # noqa: E402
from se_align.tse.model import IGNORE_INDEX  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def finite_nonzero(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    data = config["data"]
    models = config["models"]
    set_seed(int(training["seed"]))
    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    dataset = TSEManifestDataset(data["train_prepared_manifest"])
    shortest = min(dataset.rows, key=lambda row: int(row["target_num_tokens"]))
    dataset.rows = [shortest]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=TSECollator())
    raw_batch = next(iter(loader))
    trial_ids, batch = move_batch(raw_batch, device)

    model = build_tse_model(
        qwen_path=models["qwen_path"],
        wavlm_path=models["wavlm_path"],
        device=str(device),
        dtype=dtype,
        freeze_wavlm=True,
        freeze_qwen=False,
    )
    if training.get("gradient_checkpointing", True):
        model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = False
    optimizer_args = SimpleNamespace(
        learning_rate=float(training["learning_rate"]),
        projector_learning_rate=float(training["projector_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer = optimizer_for(model, optimizer_args)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    input_weight = model.llm.get_input_embeddings().weight
    output_weight = model.llm.get_output_embeddings().weight
    tied = input_weight.data_ptr() == output_weight.data_ptr()
    start = model.vocab.audio_shift
    stop = start + model.vocab.audio_vocabsize
    s3_before = [
        parameter.detach()[start:stop].clone()
        for parameter in s3_parameters(model)
    ]
    speaker_before = [
        parameter.detach().clone() for parameter in model.speaker_projector.parameters()
    ]
    mixture_before = [
        parameter.detach().clone() for parameter in model.mixture_projector.parameters()
    ]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("one-batch gate produced a non-finite loss")
    output.loss.backward()

    s3_norm = s3_grad_norm(model)
    speaker_norm = parameter_grad_norm(list(model.speaker_projector.parameters()))
    mixture_norm = parameter_grad_norm(list(model.mixture_projector.parameters()))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_grad_norm = parameter_grad_norm(trainable)
    s3_grad_present = all(parameter.grad is not None for parameter in s3_parameters(model))
    s3_grad_finite = all(
        torch.isfinite(parameter.grad[start:stop]).all().item()
        for parameter in s3_parameters(model)
        if parameter.grad is not None
    )
    speaker_grad_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.speaker_projector.parameters()
    )
    mixture_grad_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.mixture_projector.parameters()
    )
    torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
    optimizer.step()

    def change_norm(parameters, baselines, row_slice=None) -> float:
        squares = []
        for parameter, baseline in zip(parameters, baselines):
            current = parameter.detach()
            if row_slice is not None:
                current = current[row_slice]
            squares.append((current.float() - baseline.float()).square().sum())
        return float(torch.stack(squares).sum().sqrt().item())

    s3_change = change_norm(s3_parameters(model), s3_before, slice(start, stop))
    speaker_change = change_norm(list(model.speaker_projector.parameters()), speaker_before)
    mixture_change = change_norm(list(model.mixture_projector.parameters()), mixture_before)
    supervised = output.labels.ne(IGNORE_INDEX)
    labels = output.labels[supervised]
    token_accuracy = float(
        (output.logits.argmax(dim=-1)[supervised] == labels).float().mean().item()
    )
    wavlm_in_optimizer = any(
        id(parameter) in optimizer_ids for parameter in model.wavlm.parameters()
    )
    s3_in_optimizer = all(
        id(parameter) in optimizer_ids for parameter in s3_parameters(model)
    )
    checks = {
        "s3_requires_grad": all(p.requires_grad for p in s3_parameters(model)),
        "s3_in_optimizer": s3_in_optimizer,
        "s3_grad_present": s3_grad_present,
        "s3_grad_finite": s3_grad_finite,
        "s3_grad_nonzero": finite_nonzero(s3_norm),
        "s3_changed_after_step": finite_nonzero(s3_change),
        "speaker_grad_finite_nonzero": speaker_grad_finite and finite_nonzero(speaker_norm),
        "speaker_changed_after_step": finite_nonzero(speaker_change),
        "mixture_grad_finite_nonzero": mixture_grad_finite and finite_nonzero(mixture_norm),
        "mixture_changed_after_step": finite_nonzero(mixture_change),
        "wavlm_frozen": not any(p.requires_grad for p in model.wavlm.parameters()),
        "wavlm_excluded_from_optimizer": not wavlm_in_optimizer,
        "labels_are_raw_s3": bool(
            labels.numel() > 0
            and labels.min().item() >= 0
            and labels.max().item() < model.vocab.audio_vocabsize
        ),
    }
    passed = all(checks.values())
    result = {
        "final_pretrain_check": "PASS" if passed else "FAIL",
        "initialization": "RAW_QWEN_TO_TSE",
        "trial_ids": trial_ids,
        "loss": float(output.loss.detach().float().item()),
        "token_accuracy": token_accuracy,
        "supervised_target_tokens": int(labels.numel()),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
        "embedding_output_head_tied": tied,
        "s3_grad_norm": s3_norm,
        "speaker_projector_grad_norm": speaker_norm,
        "mixture_projector_grad_norm": mixture_norm,
        "total_grad_norm": total_grad_norm,
        "s3_weight_change": s3_change,
        "speaker_projector_weight_change": speaker_change,
        "mixture_projector_weight_change": mixture_change,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checks": checks,
        "optimizer_step_executed": True,
        "checkpoint_written": False,
        "formal_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"S3_TRAINABLE = {'YES' if passed else 'NO'}", flush=True)
    print(f"FINAL_PRETRAIN_CHECK = {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
