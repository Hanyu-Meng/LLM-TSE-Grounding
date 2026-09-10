#!/usr/bin/env python3
"""Audit materialized WeSep evidence against LibriMix clean references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit-mixtures", type=int, default=0)
    parser.add_argument("--retry-margin-db", type=float, default=5.0)
    parser.add_argument("--progress-interval", type=int, default=250)
    return parser.parse_args()


def trial_key(trial_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", trial_id).strip("_")[:96]
    digest = hashlib.sha1(trial_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def evidence_path(feature_root: str, trial_id: str) -> str:
    return str(Path(feature_root) / "evidence_wav" / f"{trial_key(trial_id)}.wav")


def load_audio(path: str) -> tuple[np.ndarray, int, int]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    channels = int(values.shape[1])
    mono = values.mean(axis=1, dtype=np.float64)
    return mono, int(sample_rate), channels


def signal_stats(values: np.ndarray) -> dict[str, float | int | bool]:
    finite = bool(np.isfinite(values).all())
    safe = values if finite else np.nan_to_num(values)
    return {
        "num_samples": int(values.size),
        "finite": finite,
        "nonzero": bool(np.any(safe != 0)),
        "rms": float(np.sqrt(np.mean(safe * safe))) if safe.size else 0.0,
        "peak": float(np.max(np.abs(safe))) if safe.size else 0.0,
    }


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    length = min(estimate.size, reference.size)
    if length == 0:
        return float("nan")
    estimate = estimate[:length] - np.mean(estimate[:length])
    reference = reference[:length] - np.mean(reference[:length])
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= eps:
        return float("nan")
    scale = float(np.dot(estimate, reference)) / (reference_energy + eps)
    projected = scale * reference
    noise = estimate - projected
    return float(
        10.0
        * np.log10(
            (float(np.dot(projected, projected)) + eps)
            / (float(np.dot(noise, noise)) + eps)
        )
    )


def audit_mixture(task: tuple[str, list[dict], str, float]) -> dict:
    mixture_path, rows, feature_root, retry_margin_db = task
    try:
        cache: dict[str, tuple[np.ndarray, int, int]] = {}

        def audio(path: str) -> tuple[np.ndarray, int, int]:
            if path not in cache:
                cache[path] = load_audio(path)
            return cache[path]

        mixture, mixture_sr, mixture_channels = audio(mixture_path)
        trials = []
        for row in rows:
            target, target_sr, target_channels = audio(row["target_wav"])
            interferer_path = row["interferer_wavs"][0]
            interferer, interferer_sr, interferer_channels = audio(interferer_path)
            path = evidence_path(feature_root, row["trial_id"])
            evidence, evidence_sr, evidence_channels = audio(path)
            evidence_stats = signal_stats(evidence)
            target_score = si_sdr(evidence, target)
            interferer_score = si_sdr(evidence, interferer)
            mixture_score = si_sdr(mixture, target)
            target_margin = target_score - interferer_score
            improvement = target_score - mixture_score
            valid_audio = (
                mixture_sr == target_sr == interferer_sr == evidence_sr == 16000
                and evidence_channels == 1
                and evidence_stats["finite"]
                and evidence_stats["nonzero"]
                and evidence_stats["num_samples"] > 0
                and all(math.isfinite(x) for x in (target_score, interferer_score, improvement))
            )
            if not valid_audio or target_margin <= 0.0:
                status = "FAIL"
            elif target_margin < retry_margin_db or improvement < 0.0:
                status = "RETRY"
            else:
                status = "PASS"
            trials.append(
                {
                    "trial_id": row["trial_id"],
                    "split": row["split"],
                    "target_speaker": row["target_speaker"],
                    "mixture_wav": mixture_path,
                    "target_wav": row["target_wav"],
                    "interferer_wav": interferer_path,
                    "enrollment_wav": row["enrollment_wav"],
                    "evidence_wav": path,
                    "sample_rates": {
                        "mixture": mixture_sr,
                        "target": target_sr,
                        "interferer": interferer_sr,
                        "evidence": evidence_sr,
                    },
                    "channels": {
                        "mixture": mixture_channels,
                        "target": target_channels,
                        "interferer": interferer_channels,
                        "evidence": evidence_channels,
                    },
                    "mixture_num_samples": int(mixture.size),
                    "evidence_num_samples": int(evidence.size),
                    "evidence_minus_mixture_samples": int(evidence.size - mixture.size),
                    "evidence_signal": evidence_stats,
                    "si_sdr_target_db": target_score,
                    "si_sdr_interferer_db": interferer_score,
                    "target_margin_db": target_margin,
                    "si_sdr_mixture_target_db": mixture_score,
                    "si_sdri_db": improvement,
                    "predicted_source": "target" if target_margin > 0.0 else "interferer",
                    "waveform_qc_status": status,
                }
            )
        pair_success = len(trials) == 2 and all(
            trial["predicted_source"] == "target" for trial in trials
        )
        return {
            "mixture_wav": mixture_path,
            "trials": trials,
            "pair_success": pair_success,
            "error": None,
        }
    except Exception as exc:
        return {
            "mixture_wav": mixture_path,
            "trials": [],
            "pair_success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def percentile(values: list[float], q: float) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, q)) if finite else None


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["mixture_wav"]].append(row)
    mixtures = sorted(grouped.items())
    if args.limit_mixtures:
        mixtures = mixtures[: args.limit_mixtures]
    tasks = [
        (mixture, pair, str(args.feature_root), args.retry_margin_db)
        for mixture, pair in mixtures
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    trials_path = args.output_dir / "trial_metrics.jsonl"
    pairs_path = args.output_dir / "pair_metrics.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    counts: Counter[str] = Counter()
    margins: list[float] = []
    improvements: list[float] = []
    target_scores: list[float] = []
    length_differences: Counter[int] = Counter()
    started = time.monotonic()

    with (
        trials_path.open("w", buffering=1) as trial_file,
        pairs_path.open("w", buffering=1) as pair_file,
        failures_path.open("w", buffering=1) as failure_file,
        ProcessPoolExecutor(max_workers=args.workers) as executor,
    ):
        results = executor.map(audit_mixture, tasks, chunksize=8)
        for index, result in enumerate(results, 1):
            pair_record = {
                "mixture_wav": result["mixture_wav"],
                "trial_ids": [trial["trial_id"] for trial in result["trials"]],
                "pair_success": result["pair_success"],
                "error": result["error"],
            }
            pair_file.write(json.dumps(pair_record) + "\n")
            counts["mixtures"] += 1
            counts["pair_success"] += int(result["pair_success"])
            if result["error"]:
                counts["errors"] += 1
                failure_file.write(json.dumps(pair_record) + "\n")
            for trial in result["trials"]:
                trial_file.write(json.dumps(trial) + "\n")
                status = trial["waveform_qc_status"]
                counts["trials"] += 1
                counts[status] += 1
                margins.append(trial["target_margin_db"])
                improvements.append(trial["si_sdri_db"])
                target_scores.append(trial["si_sdr_target_db"])
                length_differences[trial["evidence_minus_mixture_samples"]] += 1
                if status != "PASS":
                    failure_file.write(json.dumps(trial) + "\n")
            if index == 1 or index % args.progress_interval == 0 or index == len(tasks):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = index / elapsed
                eta = (len(tasks) - index) / max(rate, 1e-6)
                print(
                    f"[{index}/{len(tasks)}] mixtures rate={rate:.2f}/s "
                    f"eta={eta / 60:.1f}m PASS={counts['PASS']} "
                    f"RETRY={counts['RETRY']} FAIL={counts['FAIL']}",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    summary = {
        "manifest": str(args.manifest.resolve()),
        "feature_root": str(args.feature_root.resolve()),
        "workers": args.workers,
        "retry_margin_db": args.retry_margin_db,
        "elapsed_seconds": elapsed,
        "counts": dict(counts),
        "pair_failure": counts["mixtures"] - counts["pair_success"],
        "target_margin_db": {
            "min": min(margins, default=None),
            "p01": percentile(margins, 1),
            "p05": percentile(margins, 5),
            "median": percentile(margins, 50),
            "mean": float(np.mean(margins)) if margins else None,
        },
        "si_sdr_target_db": {
            "min": min(target_scores, default=None),
            "p05": percentile(target_scores, 5),
            "median": percentile(target_scores, 50),
            "mean": float(np.mean(target_scores)) if target_scores else None,
        },
        "si_sdri_db": {
            "min": min(improvements, default=None),
            "p05": percentile(improvements, 5),
            "median": percentile(improvements, 50),
            "mean": float(np.mean(improvements)) if improvements else None,
        },
        "evidence_minus_mixture_samples": {
            str(key): value for key, value in sorted(length_differences.items())
        },
        "status_definition": {
            "FAIL": "invalid audio or SI-SDR target margin <= 0 dB",
            "RETRY": (
                f"0 < target margin < {args.retry_margin_db:g} dB or SI-SDRi < 0 dB"
            ),
            "PASS": (
                f"target margin >= {args.retry_margin_db:g} dB and SI-SDRi >= 0 dB"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
