#!/usr/bin/env python3
"""Decode fixed-length Qwen-TSE TSE S3 tokens with UD or training-free CSG."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from se_align.data.tse_dataset import TSECollator, TSEManifestDataset  # noqa: E402
from se_align.train.fsq_neighbors import hamming_digits  # noqa: E402
from se_align.tse import build_tse_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--wavlm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("ud", "csg"), default="ud")
    parser.add_argument("--csg-lambda", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference-token-dir", type=Path)
    return parser.parse_args()


def token_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode("utf-8")).hexdigest()[:12]
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in trial_id)
    return f"{safe[:100]}-{digest}.npy"


def load_generator(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = build_tse_model(
        qwen_path=str(args.qwen),
        wavlm_path=str(args.wavlm),
        device=str(device),
        dtype=dtype,
        freeze_wavlm=True,
        freeze_qwen=True,
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("wavlm.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch missing={missing} "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    model.llm.gradient_checkpointing_disable()
    model.llm.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, checkpoint


def main() -> int:
    args = parse_args()
    if args.mode == "ud" and args.csg_lambda != 0:
        raise ValueError("UD requires --csg-lambda 0")
    if args.mode == "csg" and args.csg_lambda < 0:
        raise ValueError("CSG lambda must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_dir = args.output_dir / "tokens"
    token_dir.mkdir(exist_ok=True)
    dataset = TSEManifestDataset(args.manifest, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TSECollator(),
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    model, checkpoint = load_generator(args)
    device = torch.device(args.device)
    rows = []
    failures = []
    started = time.monotonic()

    processed = 0
    for raw_batch in loader:
        trial_ids = raw_batch.pop("trial_ids")
        try:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw_batch.items()
            }
            batch_started = time.monotonic()
            generated_rows = model.generate_batch(
                **batch,
                csg_lambda=args.csg_lambda,
            )
            batch_seconds = time.monotonic() - batch_started
            for offset, (trial_id, generated_device) in enumerate(
                zip(trial_ids, generated_rows)
            ):
                generated = generated_device.cpu()
                destination = token_dir / token_name(trial_id)
                evidence_length = int(batch["evidence_lengths"][offset])
                np.save(destination, generated.numpy().astype(np.int64, copy=False))
                evidence = batch[
                    "evidence_tokens"
                ][offset, :evidence_length].cpu()
                distances = hamming_digits(generated, evidence)
                exact_reference = None
                if args.reference_token_dir is not None:
                    reference_path = args.reference_token_dir / destination.name
                    if not reference_path.is_file():
                        raise FileNotFoundError(
                            f"missing invariant reference {reference_path}"
                        )
                    reference = np.load(reference_path, allow_pickle=False).reshape(-1)
                    exact_reference = bool(np.array_equal(reference, generated.numpy()))
                    if not exact_reference:
                        raise RuntimeError("lambda=0 failed exact UD token equality")
                row_index = processed + offset
                row = {
                    "trial_id": trial_id,
                    "mode": args.mode,
                    "csg_lambda": args.csg_lambda,
                    "token_path": str(destination.resolve()),
                    "evidence_token_path": dataset.rows[row_index]["evidence_token_path"],
                    "expected_tokens": evidence_length,
                    "generated_tokens": int(generated.numel()),
                    "token_id_min": int(generated.min()),
                    "token_id_max": int(generated.max()),
                    "token_flip_rate_vs_evidence": float(
                        (generated != evidence).float().mean()
                    ),
                    "mean_fsq_hamming_vs_evidence": float(distances.float().mean()),
                    "exact_reference_match": exact_reference,
                    "decode_seconds": batch_seconds / len(trial_ids),
                }
                rows.append(row)
            processed += len(trial_ids)
            if processed == len(trial_ids) or processed % 20 < len(trial_ids):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"decoded={processed}/{len(dataset)} "
                    f"rate={processed / elapsed:.3f}_trials_s",
                    flush=True,
                )
        except Exception as error:  # noqa: BLE001
            for trial_id in trial_ids:
                failure = {
                    "trial_id": trial_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                print(f"FAILED {json.dumps(failure)}", flush=True)
            processed += len(trial_ids)

    records_path = args.output_dir / "per_trial_tokens.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    failures_path = args.output_dir / "failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_step": int(checkpoint["global_step"]),
        "mode": args.mode,
        "csg_lambda": args.csg_lambda,
        "expected_trials": len(dataset),
        "decoded_trials": len(rows),
        "failed_trials": len(failures),
        "missing_trials": len(dataset) - len(rows) - len(failures),
        "all_lengths_exact": all(
            row["generated_tokens"] == row["expected_tokens"] for row in rows
        ),
        "all_ids_valid": all(
            0 <= row["token_id_min"] <= row["token_id_max"] < 6561 for row in rows
        ),
        "reference_exact": (
            bool(rows)
            and not failures
            and all(row["exact_reference_match"] is True for row in rows)
            if args.reference_token_dir is not None else None
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "token_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures and len(rows) == len(dataset) else 2


if __name__ == "__main__":
    raise SystemExit(main())
