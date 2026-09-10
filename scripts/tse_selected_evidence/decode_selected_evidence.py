#!/usr/bin/env python3
"""Leak-free Q-Full UD/CSG decoding from frozen selected evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from se_align.data.store import read_manifest  # noqa: E402
from se_align.train.fsq_neighbors import hamming_digits  # noqa: E402
from se_align.tse import build_tse_model  # noqa: E402
from se_align.utils.audio import load_wav  # noqa: E402
from scripts.resource_guard import ResourceGuardStop, check, check_or_raise  # noqa: E402


FORBIDDEN_SUBSTRINGS = (
    "target", "interferer", "transcript", "sisdr", "si_sdr", "qc", "reference", "label"
)
EXPECTED_CHECKPOINT_SHA256 = "34cf1f1cfb73c5c7d4d0fb51a380e90acf160c7a5a13cd5c50518d9e81a27eb7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", choices=("pool_full", "pool_b", "pool_d"), required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "experiments/qfull_sme_rawqwen_seed0_20260810_0858/checkpoints/best.pt",
    )
    parser.add_argument("--qwen", type=Path, default=ROOT / "pretrained/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--wavlm", type=Path, default=ROOT / "pretrained/wavlm-base-plus")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("ud", "csg"), required=True)
    parser.add_argument("--csg-lambda", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--reference-token-dir", type=Path)
    parser.add_argument("--expected", type=int, default=6000)
    parser.add_argument("--status-every", type=int, default=50)
    parser.add_argument("--resource-seconds", type=float, default=20.0)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--guard-log", type=Path,
        default=ROOT / "analysis/selected_evidence/resource_guard.jsonl",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in trial_id)
    return f"{safe[:100]}-{digest}.npy"


def load_tokens(path: str) -> torch.Tensor:
    values = np.load(path, allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
    if values.size == 0 or values.min() < 0 or values.max() >= 6561:
        raise ValueError(f"invalid raw S3 evidence tokens: {path}")
    return torch.from_numpy(values.copy())


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_rss_bytes() -> int:
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def valid_saved_record(
    row: dict[str, Any], expected_trial_id: str, token_dir: Path,
    reference_token_dir: Path | None,
) -> bool:
    try:
        if row.get("trial_id") != expected_trial_id:
            return False
        path = Path(row["token_path"])
        if path.parent.resolve() != token_dir.resolve() or path.name != token_name(expected_trial_id):
            return False
        values = np.load(path, allow_pickle=False).reshape(-1)
        if (
            values.size != int(row["expected_tokens"])
            or values.size != int(row["generated_tokens"])
            or values.size == 0
            or not np.issubdtype(values.dtype, np.integer)
            or int(values.min()) < 0
            or int(values.max()) >= 6561
        ):
            return False
        if reference_token_dir is not None:
            reference = np.load(reference_token_dir / path.name, allow_pickle=False).reshape(-1)
            if not np.array_equal(reference, values):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


class SelectedEvidenceDataset(Dataset):
    def __init__(self, manifest: Path, pool: str, split: str) -> None:
        self.manifest = manifest
        self.rows = read_manifest(manifest)
        self.pool = pool
        self.evidence_field = f"{pool}_evidence_token_path"
        required = {
            "trial_id", "split", "mixture_wav", "enrollment_wav",
            "speaker_embedding_path", self.evidence_field,
        }
        if not self.rows:
            raise ValueError("empty selected-evidence inference manifest")
        for index, row in enumerate(self.rows):
            missing = sorted(field for field in required if not row.get(field))
            leaked = sorted(
                key for key in row
                if any(value in key.lower() for value in FORBIDDEN_SUBSTRINGS)
            )
            if missing or leaked or row.get("split") != split:
                raise ValueError(
                    f"fail-closed inference manifest row={index} missing={missing} leaked={leaked} split={row.get('split')}"
                )
        ids = [row["trial_id"] for row in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate inference trial IDs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        mixture, sample_rate = load_wav(row["mixture_wav"], target_sr=16000)
        if sample_rate != 16000:
            raise ValueError(f"failed to obtain 16 kHz mixture: {row['trial_id']}")
        mixture = mixture.squeeze(0).float()
        output_length = math.ceil(mixture.numel() / 640)
        evidence = load_tokens(row[self.evidence_field])
        if evidence.numel() != output_length:
            raise ValueError(
                f"evidence length {evidence.numel()} != mixture-derived {output_length}: {row['trial_id']}"
            )
        speaker = np.load(row["speaker_embedding_path"], allow_pickle=False)
        speaker = torch.from_numpy(speaker.astype(np.float32, copy=False).reshape(-1).copy())
        if speaker.numel() != 192 or not torch.isfinite(speaker).all():
            raise ValueError(f"invalid deployment speaker embedding: {row['trial_id']}")
        return {
            "trial_id": row["trial_id"],
            "mixture": mixture,
            "evidence": evidence,
            "speaker": speaker,
            "output_length": output_length,
            "evidence_token_path": row[self.evidence_field],
        }


class SelectedEvidenceCollator:
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        mixtures = [row["mixture"] for row in rows]
        evidence = [row["evidence"] for row in rows]
        mixture_lengths = torch.tensor([value.numel() for value in mixtures], dtype=torch.long)
        mixture_values = pad_sequence(mixtures, batch_first=True)
        return {
            "trial_ids": [row["trial_id"] for row in rows],
            "evidence_token_paths": [row["evidence_token_path"] for row in rows],
            "mixture_values": mixture_values,
            "mixture_attention_mask": (
                torch.arange(mixture_values.shape[1])[None, :] < mixture_lengths[:, None]
            ).long(),
            "speaker_embeddings": torch.stack([row["speaker"] for row in rows]),
            "evidence_tokens": pad_sequence(evidence, batch_first=True, padding_value=-1),
            "evidence_lengths": torch.tensor([value.numel() for value in evidence], dtype=torch.long),
            "output_lengths": torch.tensor([row["output_length"] for row in rows], dtype=torch.long),
        }


def build_model(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = build_tse_model(
        qwen_path=str(args.qwen), wavlm_path=str(args.wavlm), device=str(device),
        dtype=dtype, freeze_wavlm=True, freeze_qwen=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("wavlm.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch missing={missing} unexpected={list(incompatible.unexpected_keys)}"
        )
    model.llm.gradient_checkpointing_disable()
    model.llm.config.use_cache = True
    model.requires_grad_(False)
    model.eval()
    return model, checkpoint


def main() -> int:
    args = parse_args()
    if args.mode == "ud" and args.csg_lambda != 0.0:
        raise ValueError("UD requires csg_lambda=0")
    if args.mode == "csg" and args.csg_lambda < 0.0:
        raise ValueError("CSG lambda must be nonnegative")
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen Q-Full checkpoint SHA-256 mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = SelectedEvidenceDataset(args.manifest, args.pool, args.split)
    if len(dataset) != args.expected:
        raise ValueError(f"expected {args.expected} trials, found {len(dataset)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_dir = args.output_dir / "tokens"
    token_dir.mkdir(exist_ok=True)
    progress_path = args.output_dir / "decode_progress.jsonl"
    records_path = args.output_dir / "per_trial_tokens.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    guard_log = args.guard_log
    guard_log.parent.mkdir(parents=True, exist_ok=True)
    phase = f"selected_{args.pool}_qfull_{args.mode}"

    prior_rows: list[dict[str, Any]] = []
    for source in (progress_path, records_path):
        if source.is_file():
            with source.open(encoding="utf-8") as handle:
                prior_rows.extend(json.loads(line) for line in handle if line.strip())
    prior_by_id: dict[str, dict[str, Any]] = {}
    dataset_ids = {row["trial_id"] for row in dataset.rows}
    for row in prior_rows:
        trial_id = row.get("trial_id")
        if trial_id in dataset_ids and valid_saved_record(
            row, trial_id, token_dir, args.reference_token_dir
        ):
            prior_by_id[trial_id] = row
    pending_indices = [
        index for index, row in enumerate(dataset.rows)
        if row["trial_id"] not in prior_by_id
    ]

    preflight = check_or_raise(
        phase=phase, disk_path=ROOT, log_path=guard_log,
        starting_new_stage=bool(pending_indices),
    )
    loader_options: dict[str, Any] = {
        "dataset": Subset(dataset, pending_indices),
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": SelectedEvidenceCollator(),
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": False,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 1
    loader = DataLoader(**loader_options)
    device = torch.device(args.device)
    model = None
    checkpoint_metadata: dict[str, int]
    if pending_indices:
        model, checkpoint = build_model(args)
        checkpoint_metadata = {
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
        }
        del checkpoint
        gc.collect()
        loaded = check(
            phase=f"{phase}_model_loaded", disk_path=ROOT,
            log_path=guard_log, starting_new_stage=False,
        )
        if loaded["evaluation"]["decision"] == "GRACEFUL_STOP":
            raise ResourceGuardStop("; ".join(loaded["evaluation"]["stop_reasons"]))
    else:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        checkpoint_metadata = {
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
        }
        del checkpoint
        gc.collect()

    records_by_id = dict(prior_by_id)
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    processed = 0
    last_guard = time.monotonic()
    guard_stop: str | None = None
    try:
        for raw_batch in loader:
            trial_ids = raw_batch.pop("trial_ids")
            evidence_paths = raw_batch.pop("evidence_token_paths")
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw_batch.items()
            }
            try:
                batch_started = time.monotonic()
                if model is None:
                    raise RuntimeError("model is unavailable for pending decode")
                generated_rows = model.generate_batch(
                    **batch, csg_lambda=args.csg_lambda
                )
                batch_seconds = time.monotonic() - batch_started
                batch_records = []
                for offset, (trial_id, generated) in enumerate(
                    zip(trial_ids, generated_rows)
                ):
                    generated = generated.cpu()
                    length = int(batch["output_lengths"][offset])
                    evidence = batch["evidence_tokens"][offset, :length].cpu()
                    if generated.numel() != length:
                        raise RuntimeError("generated output length mismatch")
                    destination = token_dir / token_name(trial_id)
                    array = generated.numpy().astype(np.int32, copy=False)
                    atomic_npy(destination, array)
                    reference_exact = None
                    if args.reference_token_dir is not None:
                        reference_path = args.reference_token_dir / destination.name
                        if not reference_path.is_file():
                            raise FileNotFoundError(reference_path)
                        reference = np.load(
                            reference_path, allow_pickle=False
                        ).reshape(-1)
                        reference_exact = bool(np.array_equal(reference, array))
                        if not reference_exact:
                            raise RuntimeError(
                                "lambda=0 is not exactly identical to UD"
                            )
                    distances = hamming_digits(generated, evidence)
                    record = {
                        "trial_id": trial_id,
                        "split": args.split,
                        "candidate_pool": args.pool,
                        "mode": args.mode,
                        "csg_lambda": args.csg_lambda,
                        "token_path": str(destination.resolve()),
                        "evidence_token_path": evidence_paths[offset],
                        "expected_tokens": length,
                        "generated_tokens": int(generated.numel()),
                        "token_id_min": int(generated.min()),
                        "token_id_max": int(generated.max()),
                        "token_flip_rate_vs_evidence": float(
                            (generated != evidence).float().mean()
                        ),
                        "mean_fsq_hamming_vs_evidence": float(
                            distances.float().mean()
                        ),
                        "exact_reference_match": reference_exact,
                        "decode_seconds": batch_seconds / len(trial_ids),
                        "output_length_source": (
                            "ceil(valid mixture samples / 640)"
                        ),
                        "clean_reference_used": False,
                        "target_length_used": False,
                        "test_used": args.split == "test",
                        "process_rss_bytes": process_rss_bytes(),
                        "cuda_allocated_bytes": (
                            int(torch.cuda.memory_allocated(device))
                            if device.type == "cuda" else 0
                        ),
                        "cuda_reserved_bytes": (
                            int(torch.cuda.memory_reserved(device))
                            if device.type == "cuda" else 0
                        ),
                    }
                    if not valid_saved_record(
                        record, trial_id, token_dir, args.reference_token_dir
                    ):
                        raise RuntimeError("post-write token validation failed")
                    batch_records.append(record)
                for record in batch_records:
                    append_jsonl(progress_path, record)
                    records_by_id[record["trial_id"]] = record
            except Exception as error:  # noqa: BLE001
                for trial_id in trial_ids:
                    failures.append({
                        "trial_id": trial_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
            processed += len(trial_ids)
            del batch
            if processed % args.status_every < len(trial_ids):
                gc.collect()
            now = time.monotonic()
            if now - last_guard >= args.resource_seconds:
                resource = check(
                    phase=phase, disk_path=ROOT, log_path=guard_log,
                    starting_new_stage=False,
                )
                last_guard = now
                if resource["evaluation"]["decision"] == "GRACEFUL_STOP":
                    guard_stop = "; ".join(
                        resource["evaluation"]["stop_reasons"]
                    )
                    raise ResourceGuardStop(guard_stop)
            done = len(records_by_id)
            if (
                processed == len(trial_ids)
                or done % args.status_every < len(trial_ids)
                or processed == len(pending_indices)
            ):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"decoded={done}/{len(dataset)} new_rate="
                    f"{processed / elapsed:.3f}/s failures={len(failures)}",
                    flush=True,
                )
    except ResourceGuardStop as error:
        guard_stop = str(error)
        print(f"RESOURCE_GUARD_STOP: {guard_stop}", flush=True)
    finally:
        before_unload = check(
            phase=f"{phase}_before_unload", disk_path=ROOT,
            log_path=guard_log, starting_new_stage=False,
        )
        if model is not None:
            del model
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        after_unload = check(
            phase=f"{phase}_after_unload", disk_path=ROOT,
            log_path=guard_log, starting_new_stage=False,
        )
        (args.output_dir / "unload_snapshot.json").write_text(
            json.dumps({"before": before_unload, "after": after_unload}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    ordered_records = [
        records_by_id[row["trial_id"]]
        for row in dataset.rows if row["trial_id"] in records_by_id
    ]
    atomic_jsonl(records_path, ordered_records)
    atomic_jsonl(failures_path, failures)
    summary = {
        "status": (
            "COMPLETE" if len(ordered_records) == len(dataset) and not failures
            else "RESOURCE_GUARD_STOP" if guard_stop else "PARTIAL"
        ),
        "manifest": str(args.manifest),
        "manifest_fields": sorted(dataset.rows[0]),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_step": checkpoint_metadata["global_step"],
        "candidate_pool": args.pool,
        "mode": args.mode,
        "csg_lambda": args.csg_lambda,
        "expected_trials": len(dataset),
        "decoded_trials": len(ordered_records),
        "failed_trials": len(failures),
        "missing_trials": len(dataset) - len(ordered_records),
        "duplicate_trials": len(ordered_records) - len({row["trial_id"] for row in ordered_records}),
        "resumed_valid_trials": len(prior_by_id),
        "new_trials": len(ordered_records) - len(prior_by_id),
        "all_lengths_exact": all(row["generated_tokens"] == row["expected_tokens"] for row in ordered_records),
        "all_ids_valid": all(0 <= row["token_id_min"] <= row["token_id_max"] < 6561 for row in ordered_records),
        "reference_exact": (
            bool(ordered_records) and not failures and all(row["exact_reference_match"] is True for row in ordered_records)
            if args.reference_token_dir is not None else None
        ),
        "clean_reference_used": False,
        "target_length_used": False,
        "test_used": args.split == "test",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "resource_guard_stop": guard_stop,
        "preflight": preflight["evaluation"]["decision"],
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "token_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] == "COMPLETE":
        return 0
    return 3 if guard_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
