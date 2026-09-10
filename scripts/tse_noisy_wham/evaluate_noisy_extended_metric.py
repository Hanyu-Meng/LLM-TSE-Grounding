#!/usr/bin/env python3
"""Resume-safe evaluation of one additional noisy TSE metric for one system."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.resource_guard import ResourceGuardStop, check, check_or_raise  # noqa: E402
from se_align.eval.metrics import (  # noqa: E402
    _PhonemeRecognizer,
    _SpeechBERTScore,
    _UTMOS,
    phoneme_levenshtein_sim,
    stoi_score,
)
from se_align.utils.audio import load_wav, resample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument(
        "--metric", choices=("duration_estoi", "utmos", "speechbertscore", "lps"),
        required=True,
    )
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--status-every", type=int, default=100)
    parser.add_argument("--resource-seconds", type=float, default=20.0)
    parser.add_argument(
        "--speechbert-model", type=Path,
        default=ROOT / "pretrained/wavlm-base-plus",
    )
    parser.add_argument("--speechbert-layer", type=int, default=8)
    parser.add_argument(
        "--phoneme-model", default="facebook/wav2vec2-lv-60-espeak-cv-ft"
    )
    parser.add_argument(
        "--guard-log", type=Path,
        default=ROOT / "analysis/noisy_wham/resource_guard.jsonl",
    )
    parser.add_argument(
        "--shared-cache-root", type=Path,
        help=(
            "Optional split-local cache for exact scalar reuse and target "
            "SpeechBERT/phoneme features across systems"
        ),
    )
    parser.add_argument(
        "--verify-cache-hits", type=int, default=2,
        help="Recompute this many reference-cache hits and require exact equality",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def file_identity(path_value: str) -> dict[str, int]:
    """Identity shared by hard links, invalidated by atomic replacement/write."""
    stat = Path(path_value).stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_configuration(args: argparse.Namespace) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "noisy_extended_exact_cache_v1",
        "metric": args.metric,
        "device": args.device,
    }
    if args.metric == "speechbertscore":
        value.update({
            "model": str(args.speechbert_model.resolve()),
            "layer": int(args.speechbert_layer),
        })
    elif args.metric == "lps":
        value["model"] = str(args.phoneme_model)
    return value


def stable_key(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def scalar_cache_key(args: argparse.Namespace, row: dict[str, Any]) -> str:
    value: dict[str, Any] = {
        "configuration": cache_configuration(args),
        "output": file_identity(row["output_wav"]),
    }
    if args.metric != "utmos":
        value["target"] = file_identity(row["target_wav"])
    return stable_key(value)


def reference_cache_key(
    args: argparse.Namespace, path_value: str, samples: int,
) -> str:
    return stable_key({
        "configuration": cache_configuration(args),
        "reference": file_identity(path_value),
        "samples": int(samples),
    })


def valid_values(values: Any, metric: str) -> bool:
    if not isinstance(values, dict):
        return False
    for key in metric_keys(metric):
        value = values.get(key)
        if isinstance(value, bool):
            continue
        if value is None or not math.isfinite(float(value)):
            return False
    return True


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value.detach().cpu(), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_tensor(path: Path, device: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=device)
    if (
        not isinstance(value, torch.Tensor) or value.ndim != 2
        or value.shape[0] <= 0 or value.shape[1] <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"invalid cached reference tensor: {path}")
    return value


def cached_speechbert_reference(
    args: argparse.Namespace,
    model: Any,
    waveform: np.ndarray,
    path_value: str,
    stats: dict[str, int],
) -> torch.Tensor:
    if args.shared_cache_root is None:
        return model._feats(waveform)
    key = reference_cache_key(args, path_value, len(waveform))
    path = args.shared_cache_root / "speechbert_reference" / f"{key}.pt"
    if path.is_file():
        cached = load_tensor(path, model.device)
        stats["reference_hits"] += 1
        if stats["reference_exact_checks"] < args.verify_cache_hits:
            fresh = model._feats(waveform)
            if fresh.shape != cached.shape or not torch.equal(fresh, cached):
                raise ValueError("cached SpeechBERT reference feature mismatch")
            stats["reference_exact_checks"] += 1
        return cached
    value = model._feats(waveform)
    atomic_tensor(path, value)
    stats["reference_misses"] += 1
    return value


def cached_reference_phonemes(
    args: argparse.Namespace,
    model: Any,
    waveform: np.ndarray,
    path_value: str,
    stats: dict[str, int],
) -> list[str]:
    if args.shared_cache_root is None:
        return model.phonemes(waveform)
    key = reference_cache_key(args, path_value, len(waveform))
    path = args.shared_cache_root / "phoneme_reference" / f"{key}.json"
    if path.is_file():
        value = json.loads(path.read_text())
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"invalid cached reference phonemes: {path}")
        stats["reference_hits"] += 1
        if stats["reference_exact_checks"] < args.verify_cache_hits:
            fresh = model.phonemes(waveform)
            if fresh != value:
                raise ValueError("cached reference phoneme sequence mismatch")
            stats["reference_exact_checks"] += 1
        return value
    value = model.phonemes(waveform)
    atomic_text(path, json.dumps(value) + "\n")
    stats["reference_misses"] += 1
    return value


def audio16(path: str) -> np.ndarray:
    waveform, rate = load_wav(path, mono=True)
    return resample(waveform, rate, 16000).reshape(-1).numpy().astype(np.float32)


def metric_keys(metric: str) -> tuple[str, ...]:
    return {
        "duration_estoi": (
            "estoi", "reference_duration_seconds", "output_duration_seconds",
            "common_duration_seconds", "output_to_reference_length_ratio",
            "duration_mismatch_gt_5pct",
        ),
        "utmos": ("utmos",),
        "speechbertscore": ("speechbertscore",),
        "lps": ("lps",),
    }[metric]


def valid(row: dict[str, Any], trial_id: str, metric: str) -> bool:
    if row.get("trial_id") != trial_id or row.get("metric") != metric:
        return False
    for key in metric_keys(metric):
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if value is None or not math.isfinite(float(value)):
            return False
    return True


def build_metric(args: argparse.Namespace) -> Any:
    if args.metric == "utmos":
        return _UTMOS(device=args.device)
    if args.metric == "speechbertscore":
        return _SpeechBERTScore(
            str(args.speechbert_model), layer=args.speechbert_layer, device=args.device
        )
    if args.metric == "lps":
        return _PhonemeRecognizer(args.phoneme_model, device=args.device)
    return None


def score(
    args: argparse.Namespace,
    model: Any,
    row: dict[str, Any],
    cache_stats: dict[str, int],
) -> dict[str, Any]:
    output = audio16(row["output_wav"])
    if args.metric == "utmos":
        values = {"utmos": float(model(output))}
    else:
        reference = audio16(row["target_wav"])
        common = min(len(reference), len(output))
        if common <= 0:
            raise ValueError("empty common waveform")
        r = reference[:common]
        o = output[:common]
        if args.metric == "duration_estoi":
            ratio = len(output) / len(reference)
            values = {
                "estoi": float(stoi_score(r, o, 16000, extended=True)),
                "reference_duration_seconds": len(reference) / 16000.0,
                "output_duration_seconds": len(output) / 16000.0,
                "common_duration_seconds": common / 16000.0,
                "output_to_reference_length_ratio": ratio,
                "duration_mismatch_gt_5pct": abs(ratio - 1.0) > 0.05,
            }
        elif args.metric == "speechbertscore":
            reference_features = cached_speechbert_reference(
                args, model, r, row["target_wav"], cache_stats
            )
            output_features = model._feats(o)
            similarity = output_features @ reference_features.T
            precision = similarity.max(dim=1).values.mean()
            recall = similarity.max(dim=0).values.mean()
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            values = {"speechbertscore": float(f1)}
        else:
            values = {"lps": float(phoneme_levenshtein_sim(
                cached_reference_phonemes(
                    args, model, r, row["target_wav"], cache_stats
                ),
                model.phonemes(o),
            ))}
    if not all(
        isinstance(value, bool) or math.isfinite(float(value))
        for value in values.values()
    ):
        raise ValueError("non-finite metric output")
    return values


def main() -> int:
    args = parse_args()
    source = read_jsonl(args.source)
    ids = [row["trial_id"] for row in source]
    id_set = set(ids)
    if len(source) != args.expected or len(set(ids)) != args.expected:
        raise ValueError("extended metric source coverage mismatch")
    if any(row.get("split") != args.split or row.get("decode_status") != "ok" for row in source):
        raise ValueError("extended metric source split/status mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.jsonl"
    records_path = args.output_dir / "per_trial.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    prior: dict[str, dict[str, Any]] = {}
    for path in (progress_path, records_path):
        if path.is_file():
            for row in read_jsonl(path):
                if row.get("trial_id") in id_set and valid(row, row["trial_id"], args.metric):
                    prior[row["trial_id"]] = row
    pending = [row for row in source if row["trial_id"] not in prior]
    cache_stats = {
        "scalar_hits": 0,
        "scalar_added": 0,
        "reference_hits": 0,
        "reference_misses": 0,
        "reference_exact_checks": 0,
    }
    scalar_cache_path = None
    scalar_cache_rows: dict[str, dict[str, Any]] = {}
    if args.shared_cache_root is not None:
        args.shared_cache_root.mkdir(parents=True, exist_ok=True)
        scalar_cache_path = args.shared_cache_root / "scalar_metrics.jsonl"
        if scalar_cache_path.is_file():
            for cached in read_jsonl(scalar_cache_path):
                key = cached.get("cache_key")
                metric = cached.get("metric")
                if (
                    key is not None
                    and metric in {"duration_estoi", "utmos", "speechbertscore", "lps"}
                    and valid_values(cached.get("values"), metric)
                ):
                    scalar_cache_rows[str(key)] = cached
        still_pending = []
        for row in pending:
            key = scalar_cache_key(args, row)
            cached = scalar_cache_rows.get(key)
            if cached is None or cached.get("metric") != args.metric:
                still_pending.append(row)
                continue
            record = {
                "trial_id": row["trial_id"],
                "split": args.split,
                "system": args.system_name,
                "metric": args.metric,
                **cached["values"],
                "shared_scalar_cache_hit": True,
                "test_used": args.split == "test",
            }
            if not valid(record, row["trial_id"], args.metric):
                raise ValueError("invalid exact scalar cache record")
            append_jsonl(progress_path, record)
            prior[row["trial_id"]] = record
            cache_stats["scalar_hits"] += 1
        pending = still_pending
    phase = f"noisy_{args.metric}_{args.system_name}"
    check_or_raise(
        phase=phase, disk_path=ROOT, log_path=args.guard_log,
        starting_new_stage=bool(pending),
    )
    model = None
    asset_error = None
    if pending:
        try:
            model = build_metric(args)
        except Exception as error:  # noqa: BLE001
            asset_error = f"{type(error).__name__}: {error}"
    if asset_error:
        summary = {
            "status": "ASSET_UNAVAILABLE", "system": args.system_name,
            "metric": args.metric, "expected": args.expected,
            "completed": len(prior), "coverage": len(prior) / args.expected,
            "asset_error": asset_error, "test_used": args.split == "test",
        }
        atomic_text(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 4

    records = dict(prior)
    failures = []
    started = time.monotonic()
    last_guard = started
    guard_stop = None
    try:
        for index, row in enumerate(pending, 1):
            trial_id = row["trial_id"]
            try:
                values = score(args, model, row, cache_stats)
                record = {
                    "trial_id": trial_id,
                    "split": args.split,
                    "system": args.system_name,
                    "metric": args.metric,
                    **values,
                    "test_used": args.split == "test",
                }
                if not valid(record, trial_id, args.metric):
                    raise ValueError("post-score validation failed")
                append_jsonl(progress_path, record)
                records[trial_id] = record
                if scalar_cache_path is not None:
                    key = scalar_cache_key(args, row)
                    scalar_cache_rows[key] = {
                        "cache_key": key,
                        "metric": args.metric,
                        "configuration": cache_configuration(args),
                        "target_wav": (
                            str(Path(row["target_wav"]).resolve())
                            if args.metric != "utmos" else None
                        ),
                        "output_wav": str(Path(row["output_wav"]).resolve()),
                        "values": values,
                    }
                    cache_stats["scalar_added"] += 1
            except Exception as error:  # noqa: BLE001
                failures.append({
                    "trial_id": trial_id, "error_type": type(error).__name__,
                    "error": str(error),
                })
            now = time.monotonic()
            if now - last_guard >= args.resource_seconds:
                resource = check(
                    phase=phase, disk_path=ROOT, log_path=args.guard_log,
                    starting_new_stage=False,
                )
                last_guard = now
                if resource["evaluation"]["decision"] == "GRACEFUL_STOP":
                    guard_stop = "; ".join(resource["evaluation"]["stop_reasons"])
                    raise ResourceGuardStop(guard_stop)
            if index == 1 or len(records) % args.status_every == 0 or index == len(pending):
                print(
                    f"{args.metric}={len(records)}/{len(source)} "
                    f"failures={len(failures)} rate={index / max(now-started, 1e-6):.3f}/s",
                    flush=True,
                )
    except ResourceGuardStop as error:
        guard_stop = str(error)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        check(
            phase=f"{phase}_after_unload", disk_path=ROOT,
            log_path=args.guard_log, starting_new_stage=False,
        )

    if scalar_cache_path is not None:
        atomic_jsonl(
            scalar_cache_path,
            [scalar_cache_rows[key] for key in sorted(scalar_cache_rows)],
        )

    ordered = [records[trial_id] for trial_id in ids if trial_id in records]
    atomic_text(records_path, "".join(json.dumps(row) + "\n" for row in ordered))
    atomic_text(failures_path, "".join(json.dumps(row) + "\n" for row in failures))
    summary = {
        "status": (
            "COMPLETE" if len(ordered) == args.expected and not failures
            else "RESOURCE_GUARD_STOP" if guard_stop else "PARTIAL"
        ),
        "system": args.system_name,
        "metric": args.metric,
        "expected": args.expected,
        "completed": len(ordered),
        "failures": len(failures),
        "coverage": len(ordered) / args.expected,
        "mean": {
            key: (
                float(np.mean([row[key] for row in ordered])) if ordered else None
            )
            for key in metric_keys(args.metric)
        },
        "sample_alignment": (
            "mono 16 kHz; sample-zero common length; no shift/DTW/scaling"
            if args.metric != "utmos" else "reference-free full output"
        ),
        "test_used": args.split == "test",
        "shared_cache_root": (
            str(args.shared_cache_root.resolve())
            if args.shared_cache_root is not None else None
        ),
        "shared_scalar_cache_hits": cache_stats["scalar_hits"],
        "shared_scalar_cache_added": cache_stats["scalar_added"],
        "reference_cache_hits": cache_stats["reference_hits"],
        "reference_cache_misses": cache_stats["reference_misses"],
        "reference_cache_live_exact_checks": cache_stats["reference_exact_checks"],
        "resource_guard_stop": guard_stop,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_text(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "COMPLETE" else 3 if guard_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
