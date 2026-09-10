#!/usr/bin/env python3
"""Train the evidence-conditioned Qwen/WavLM TSE token generator."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from se_align.data.tse_dataset import TSECollator, TSEManifestDataset
from se_align.tse import build_tse_model
from se_align.tse.model import IGNORE_INDEX


def config_defaults(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    models = config.get("models", {})
    data = config.get("data", {})
    training = config.get("training", {})
    defaults: dict[str, object] = {
        "qwen": Path(models["qwen_path"]) if models.get("qwen_path") else None,
        "wavlm": Path(models["wavlm_path"]) if models.get("wavlm_path") else None,
        "train_manifest": (
            Path(data["train_prepared_manifest"])
            if data.get("train_prepared_manifest") else None
        ),
        "dev_manifest": (
            Path(data["dev_prepared_manifest"])
            if data.get("dev_prepared_manifest") else None
        ),
    }
    for key in (
        "seed", "epochs", "batch_size", "dev_batch_size",
        "gradient_accumulation_steps", "learning_rate",
        "projector_learning_rate", "weight_decay", "warmup_ratio",
        "max_grad_norm", "dtype", "num_workers", "gradient_checkpointing",
        "log_interval", "s3_diagnostic_interval",
    ):
        if key in training:
            defaults[key] = training[key]
    return {key: value for key, value in defaults.items() if value is not None}


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    known, _ = bootstrap.parse_known_args()
    defaults = config_defaults(known.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--wavlm", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dev-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--projector-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--s3-diagnostic-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-scheduler-on-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep model/optimizer state but start a fresh schedule for remaining epochs",
    )
    parser.add_argument("--resume-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-dev-batches", type=int)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.set_defaults(**defaults)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    required = ("train_manifest", "dev_manifest", "qwen", "wavlm")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(
            f"missing required arguments/config values: {', '.join(missing)}"
        )
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dev_batch_size": args.dev_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "projector_learning_rate": args.projector_learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "log_interval": args.log_interval,
        "s3_diagnostic_interval": args.s3_diagnostic_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {', '.join(invalid)}")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.resume_warmup_ratio < 1.0:
        raise ValueError("--resume-warmup-ratio must be in [0, 1)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.dtype == "bfloat16" and not args.device.startswith("cuda"):
        raise ValueError("use --dtype float32 when training on CPU")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: TSEManifestDataset, batch_size: int, shuffle: bool,
                num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TSECollator(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def move_batch(batch: dict, device: torch.device) -> tuple[list[str], dict]:
    trial_ids = batch.pop("trial_ids")
    values = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }
    return trial_ids, values


def optimizer_for(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    qwen_parameters = []
    projector_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("mixture_projector.", "speaker_projector.")):
            projector_parameters.append(parameter)
        else:
            qwen_parameters.append(parameter)
    if not qwen_parameters or not projector_parameters:
        raise RuntimeError("expected trainable Qwen and projector parameters")
    return AdamW(
        [
            {
                "params": qwen_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": projector_parameters,
                "lr": args.projector_learning_rate,
                "weight_decay": args.weight_decay,
            },
        ]
    )


def cosine_schedule(optimizer: AdamW, total_steps: int,
                    warmup_ratio: float) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


def token_metrics(output) -> tuple[int, int, float]:
    mask = output.labels.ne(IGNORE_INDEX)
    token_count = int(mask.sum().item())
    correct = int((output.logits.argmax(dim=-1)[mask] == output.labels[mask]).sum().item())
    return token_count, correct, float(output.loss.detach().float().item())


def parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def s3_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    candidates = [
        model.llm.get_input_embeddings().weight,
        model.llm.get_output_embeddings().weight,
    ]
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in candidates:
        if id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return unique


def s3_grad_norm(model: torch.nn.Module) -> float:
    start = model.vocab.audio_shift
    stop = start + model.vocab.audio_vocabsize
    squares = []
    for parameter in s3_parameters(model):
        if parameter.grad is not None:
            squares.append(parameter.grad[start:stop].detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def snapshot_s3_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    start = model.vocab.audio_shift
    stop = start + model.vocab.audio_vocabsize
    return [
        parameter.detach()[start:stop].clone()
        for parameter in s3_parameters(model)
    ]


def s3_weight_change(model: torch.nn.Module,
                     initial: list[torch.Tensor]) -> float:
    start = model.vocab.audio_shift
    stop = start + model.vocab.audio_vocabsize
    squares = []
    for parameter, baseline in zip(s3_parameters(model), initial):
        difference = parameter.detach()[start:stop].float() - baseline.float()
        squares.append(difference.square().sum())
    return float(torch.stack(squares).sum().sqrt().item())


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_status(path: Path | None, args: argparse.Namespace,
                 state: dict[str, object]) -> None:
    if path is None:
        return
    value = lambda key, default="NOT_AVAILABLE": state.get(key, default)
    lines = [
        "# Q-Full Raw-Qwen TSE Status",
        "",
        "Experiment:", "Q-Full [S,M,E]",
        "", "Initialization:", "Raw Qwen2.5-0.5B-Instruct",
        "", "Screen:", os.environ.get("QFULL_SCREEN_SESSION", "NOT_AVAILABLE"),
        "", "TensorBoard screen:", os.environ.get("QFULL_TB_SESSION", "NOT_AVAILABLE"),
        "", "TensorBoard port:", os.environ.get("QFULL_TB_PORT", "NOT_AVAILABLE"),
        "", "W&B:", "DISABLED",
        "", "PID:", str(os.getpid()),
        "", "Start time:", str(value("start_time")),
        "", "Command:", " ".join(sys.argv),
        "", "Seed:", str(args.seed),
        "", "Trainable params:", str(value("trainable_parameters")),
        "", "S3 trainable:", str(value("s3_trainable", "NO")),
        "", "Latest step:", str(value("global_step", 0)),
        "", "Latest train loss:", str(value("train_loss")),
        "", "Latest dev loss:", str(value("dev_loss")),
        "", "Latest token accuracy:", str(value("token_accuracy")),
        "", "Latest S3 grad norm:", str(value("s3_grad_norm")),
        "", "Latest probe target WER:", "NOT_AVAILABLE",
        "", "Latest probe speaker margin:", "NOT_AVAILABLE",
        "", "Latest probe switch rate:", "NOT_AVAILABLE",
        "", "GPU:", str(value("gpu")),
        "", "Latest checkpoint:", str(value("checkpoint")),
        "", "Status:", str(value("status", "RUNNING")),
        "",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: AdamW,
                scheduler: LambdaLR, device: torch.device, args: argparse.Namespace,
                epoch: int, global_step: int, writer: SummaryWriter,
                step_metrics_path: Path, initial_s3: list[torch.Tensor],
                status_state: dict[str, object]) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_count = len(loader)
    if args.max_train_batches is not None:
        batch_count = min(batch_count, args.max_train_batches)
    loss_sum = 0.0
    token_count = 0
    correct = 0
    started = time.monotonic()
    update_started = started
    group_loss_sum = 0.0
    group_tokens = 0
    group_correct = 0
    group_data_time = 0.0
    data_started = time.monotonic()
    speaker_parameters = [
        parameter for parameter in model.speaker_projector.parameters()
        if parameter.requires_grad
    ]
    mixture_parameters = [
        parameter for parameter in model.mixture_projector.parameters()
        if parameter.requires_grad
    ]

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= batch_count:
            break
        _, batch = move_batch(raw_batch, device)
        group_data_time += time.monotonic() - data_started
        group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
        group_size = min(args.gradient_accumulation_steps, batch_count - group_start)
        output = model(**batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {batch_index + 1}")
        current_tokens, current_correct, current_loss = token_metrics(output)
        (output.loss / group_size).backward()
        loss_sum += current_loss * current_tokens
        token_count += current_tokens
        correct += current_correct
        group_loss_sum += current_loss * current_tokens
        group_tokens += current_tokens
        group_correct += current_correct

        update = (batch_index - group_start + 1 == group_size)
        if update:
            current_s3_grad_norm = s3_grad_norm(model)
            speaker_grad_norm = parameter_grad_norm(speaker_parameters)
            mixture_grad_norm = parameter_grad_norm(mixture_parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at global step {global_step}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_time = max(time.monotonic() - update_started, 1e-6)
            step_loss = group_loss_sum / max(1, group_tokens)
            step_accuracy = group_correct / max(1, group_tokens)
            learning_rate = scheduler.get_last_lr()[0]
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": step_loss,
                "train_token_accuracy": step_accuracy,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "s3_token_grad_norm": current_s3_grad_norm,
                "speaker_projector_grad_norm": speaker_grad_norm,
                "mixture_projector_grad_norm": mixture_grad_norm,
                "step_time": step_time,
                "data_time": group_data_time,
                "steps_per_sec": 1.0 / step_time,
                "tokens_per_sec": group_tokens / step_time,
            }
            if device.type == "cuda":
                record["gpu_memory_allocated_gb"] = (
                    torch.cuda.memory_allocated(device) / (1024 ** 3)
                )
                record["gpu_memory_reserved_gb"] = (
                    torch.cuda.memory_reserved(device) / (1024 ** 3)
                )
            if global_step == 1 or global_step % args.s3_diagnostic_interval == 0:
                record["s3_token_weight_change"] = s3_weight_change(model, initial_s3)
            append_jsonl(step_metrics_path, record)
            tensorboard_tags = {
                "train_loss": "train/loss",
                "train_token_accuracy": "train/token_accuracy",
                "learning_rate": "train/learning_rate",
                "grad_norm": "train/grad_norm",
                "s3_token_grad_norm": "train/s3_token_grad_norm",
                "speaker_projector_grad_norm": "train/speaker_projector_grad_norm",
                "mixture_projector_grad_norm": "train/mixture_projector_grad_norm",
                "steps_per_sec": "train/steps_per_sec",
                "tokens_per_sec": "train/tokens_per_sec",
                "s3_token_weight_change": "train/s3_token_weight_change",
                "gpu_memory_allocated_gb": "system/gpu_memory_allocated_gb",
                "gpu_memory_reserved_gb": "system/gpu_memory_reserved_gb",
                "step_time": "system/step_time",
                "data_time": "system/data_time",
            }
            for name, tag in tensorboard_tags.items():
                if name in record:
                    writer.add_scalar(tag, record[name], global_step)
            writer.flush()
            status_state.update({
                "global_step": global_step,
                "train_loss": f"{step_loss:.6f}",
                "token_accuracy": f"{step_accuracy:.6f}",
                "s3_grad_norm": f"{current_s3_grad_norm:.6e}",
                "status": "RUNNING",
            })
            if global_step == 1 or global_step % args.log_interval == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"train epoch={epoch} update={global_step} "
                    f"batch={batch_index + 1}/{batch_count} "
                    f"loss={loss_sum / max(1, token_count):.6f} "
                    f"acc={correct / max(1, token_count):.4f} "
                    f"lr={learning_rate:.3e} grad_norm={float(grad_norm):.3e} "
                    f"s3_grad_norm={current_s3_grad_norm:.3e} "
                    f"speaker_grad_norm={speaker_grad_norm:.3e} "
                    f"mixture_grad_norm={mixture_grad_norm:.3e} "
                    f"tokens_per_s={token_count / elapsed:.1f}",
                    flush=True,
                )
                write_status(args.status_file, args, status_state)
            update_started = time.monotonic()
            group_loss_sum = 0.0
            group_tokens = 0
            group_correct = 0
            group_data_time = 0.0
        data_started = time.monotonic()
    return {
        "loss": loss_sum / max(1, token_count),
        "token_accuracy": correct / max(1, token_count),
        "tokens": float(token_count),
    }, global_step


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             max_batches: int | None) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    correct = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        _, batch = move_batch(raw_batch, device)
        output = model(**batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError(f"non-finite dev loss at batch {batch_index + 1}")
        current_tokens, current_correct, current_loss = token_metrics(output)
        loss_sum += current_loss * current_tokens
        token_count += current_tokens
        correct += current_correct
    if token_count == 0:
        raise RuntimeError("development loader produced no supervised tokens")
    return {
        "loss": loss_sum / token_count,
        "token_accuracy": correct / token_count,
        "tokens": float(token_count),
    }


def model_state_without_wavlm(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("wavlm.")
    }


def atomic_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_resume(path: Path, model: torch.nn.Module, optimizer: AdamW,
                scheduler: LambdaLR, load_scheduler: bool = True) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.startswith("wavlm.")]
    if unexpected or missing:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(checkpoint["optimizer"])
    if load_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint["best_dev_loss"]),
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.output_dir / "tensorboard"
    if args.metrics_path is None:
        args.metrics_path = args.output_dir / "step_metrics.jsonl"
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.status_file is not None:
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, default=str, indent=2)
        handle.write("\n")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"device={torch.cuda.get_device_name(device)} dtype={args.dtype}")
    else:
        print(f"device={device} dtype={args.dtype}")

    train_dataset = TSEManifestDataset(args.train_manifest, limit=args.train_limit)
    dev_dataset = TSEManifestDataset(args.dev_manifest, limit=args.dev_limit)
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers
    )
    dev_loader = make_loader(
        dev_dataset, args.dev_batch_size, False, args.num_workers
    )
    print(f"train_trials={len(train_dataset)} dev_trials={len(dev_dataset)}")

    with (args.qwen / "config.json").open("r", encoding="utf-8") as handle:
        qwen_source_config = json.load(handle)
    base_vocab_size = int(qwen_source_config["vocab_size"])
    print(f"Qwen source path={args.qwen.resolve()}")
    print(f"Qwen source vocabulary size={base_vocab_size}")
    model = build_tse_model(
        qwen_path=str(args.qwen),
        wavlm_path=str(args.wavlm),
        device=str(device),
        dtype=dtype,
        freeze_wavlm=True,
        freeze_qwen=False,
    )
    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = False
    optimizer = optimizer_for(model, args)
    input_weight = model.llm.get_input_embeddings().weight
    output_weight = model.llm.get_output_embeddings().weight
    tied_embeddings = input_weight.data_ptr() == output_weight.data_ptr()
    s3_start = model.vocab.audio_shift
    s3_stop = s3_start + model.vocab.audio_vocabsize
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    wavlm_in_optimizer = any(
        id(parameter) in optimizer_parameter_ids
        for parameter in model.wavlm.parameters()
    )
    s3_in_optimizer = all(
        id(parameter) in optimizer_parameter_ids for parameter in s3_parameters(model)
    )
    s3_trainable = all(parameter.requires_grad for parameter in s3_parameters(model))
    if not s3_trainable or not s3_in_optimizer or wavlm_in_optimizer:
        raise RuntimeError(
            "invalid optimizer membership: "
            f"s3_trainable={s3_trainable} s3_in_optimizer={s3_in_optimizer} "
            f"wavlm_in_optimizer={wavlm_in_optimizer}"
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_percentage = 100.0 * trainable_parameters / total_parameters
    print("INITIALIZATION = RAW_QWEN_TO_TSE")
    print("QWEN_BACKBONE: pretrained Qwen2.5-0.5B-Instruct")
    print(f"Qwen base embedding shape=({base_vocab_size}, {input_weight.shape[1]})")
    print(f"Qwen resized embedding shape={tuple(input_weight.shape)}")
    print(f"Qwen output head shape={tuple(output_weight.shape)}")
    print(f"Vocabulary resize={base_vocab_size}->{input_weight.shape[0]}")
    print(f"S3 raw vocabulary size={model.vocab.audio_vocabsize}")
    print(f"audio_shift={model.vocab.audio_shift}")
    print(f"S3 raw IDs=0..{model.vocab.audio_vocabsize - 1}")
    print(f"S3 combined IDs={s3_start}..{s3_stop - 1}")
    print("S3_TOKEN_EMBEDDING: NEW")
    print("S3_OUTPUT_HEAD: NEW")
    print("MIXTURE_PROJECTOR: NEW")
    print("SPEAKER_PROJECTOR: NEW")
    print(f"embedding_output_head_tied={tied_embeddings}")
    print(f"total_parameters={total_parameters}")
    print(f"trainable_parameters={trainable_parameters}")
    print(f"trainable_percentage={trainable_percentage:.4f}")
    print("trainable_strategy=entire_Qwen_plus_mixture_and_speaker_projectors")
    print(f"S3_TRAINABLE={'YES' if s3_trainable and s3_in_optimizer else 'NO'}")
    print(f"wavlm_in_optimizer={wavlm_in_optimizer}")

    metadata = {
        "initialization": "RAW_QWEN_TO_TSE",
        "qwen_source": str(args.qwen.resolve()),
        "qwen_base_embedding_shape": [base_vocab_size, input_weight.shape[1]],
        "qwen_resized_embedding_shape": list(input_weight.shape),
        "qwen_output_head_shape": list(output_weight.shape),
        "s3_raw_vocab_size": model.vocab.audio_vocabsize,
        "audio_shift": model.vocab.audio_shift,
        "s3_combined_id_range": [s3_start, s3_stop - 1],
        "embedding_output_head_tied": tied_embeddings,
        "s3_trainable": s3_trainable,
        "s3_in_optimizer": s3_in_optimizer,
        "wavlm_in_optimizer": wavlm_in_optimizer,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": trainable_percentage,
    }
    metadata_path = (
        args.status_file.parent if args.status_file is not None else args.output_dir
    ) / "initialization.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    train_batches = len(train_loader)
    if args.max_train_batches is not None:
        train_batches = min(train_batches, args.max_train_batches)
    updates_per_epoch = math.ceil(train_batches / args.gradient_accumulation_steps)
    scheduler = cosine_schedule(
        optimizer, max(1, updates_per_epoch * args.epochs), args.warmup_ratio
    )
    start_epoch = 1
    global_step = 0
    best_dev_loss = math.inf
    if args.resume is not None:
        start_epoch, global_step, best_dev_loss = load_resume(
            args.resume, model, optimizer, scheduler,
            load_scheduler=not args.reset_scheduler_on_resume,
        )
        if args.reset_scheduler_on_resume:
            optimizer.param_groups[0]["lr"] = args.learning_rate
            optimizer.param_groups[0]["initial_lr"] = args.learning_rate
            optimizer.param_groups[1]["lr"] = args.projector_learning_rate
            optimizer.param_groups[1]["initial_lr"] = args.projector_learning_rate
            remaining_epochs = args.epochs - start_epoch + 1
            if remaining_epochs <= 0:
                raise ValueError(
                    f"--epochs {args.epochs} leaves no epochs after resume epoch "
                    f"{start_epoch - 1}"
                )
            scheduler = cosine_schedule(
                optimizer,
                max(1, updates_per_epoch * remaining_epochs),
                args.resume_warmup_ratio,
            )
            print(
                "resume_schedule=RESET "
                f"remaining_epochs={remaining_epochs} "
                f"qwen_lr={args.learning_rate:.3e} "
                f"projector_lr={args.projector_learning_rate:.3e} "
                f"warmup_ratio={args.resume_warmup_ratio:.3f}"
            )
        print(
            f"resumed={args.resume} start_epoch={start_epoch} "
            f"global_step={global_step} best_dev_loss={best_dev_loss:.6f}"
        )

    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    initial_s3 = snapshot_s3_parameters(model)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    status_state: dict[str, object] = {
        "start_time": started_at,
        "trainable_parameters": trainable_parameters,
        "s3_trainable": "YES",
        "global_step": global_step,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "status": "RUNNING",
    }
    write_status(args.status_file, args, status_state)
    epoch_metrics_path = args.output_dir / "metrics.jsonl"
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, device, args, epoch,
                global_step, writer, args.metrics_path, initial_s3, status_state
            )
            dev_metrics = evaluate(model, dev_loader, device, args.max_dev_batches)
            writer.add_scalar("dev/loss", dev_metrics["loss"], global_step)
            writer.add_scalar(
                "dev/token_accuracy", dev_metrics["token_accuracy"], global_step
            )
            writer.flush()
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "dev": dev_metrics,
            }
            with epoch_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
                f"train_acc={train_metrics['token_accuracy']:.4f} "
                f"dev_loss={dev_metrics['loss']:.6f} "
                f"dev_acc={dev_metrics['token_accuracy']:.4f}",
                flush=True,
            )

            improved = dev_metrics["loss"] < best_dev_loss
            if improved:
                best_dev_loss = dev_metrics["loss"]
            model_state = model_state_without_wavlm(model)
            common = {
                "model": model_state,
                "epoch": epoch,
                "global_step": global_step,
                "best_dev_loss": best_dev_loss,
                "args": vars(args),
            }
            if improved:
                atomic_save(common, args.output_dir / "best.pt")
                print(f"saved={args.output_dir / 'best.pt'}")
            latest = dict(common)
            latest.update({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            })
            atomic_save(latest, args.output_dir / "last.pt")
            print(f"saved={args.output_dir / 'last.pt'}")
            status_state.update({
                "global_step": global_step,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "dev_loss": f"{dev_metrics['loss']:.6f}",
                "token_accuracy": f"{dev_metrics['token_accuracy']:.6f}",
                "checkpoint": str(args.output_dir / "last.pt"),
                "status": "RUNNING" if epoch < args.epochs else "FINISHED",
            })
            write_status(args.status_file, args, status_state)
    except Exception:
        status_state["status"] = "FAILED"
        write_status(args.status_file, args, status_state)
        raise
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
