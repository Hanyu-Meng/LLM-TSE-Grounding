#!/usr/bin/env python3
"""Compute clean-reference SI-SDR diagnostics for frozen DEV candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "analysis/candidate_gate/candidate_input_manifest.jsonl",
    )
    parser.add_argument("--candidate-paths", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--expected-trials", type=int, default=5991)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_mono16(path: str) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if int(sample_rate) != 16000:
        raise ValueError(f"expected 16 kHz audio: {path}")
    return values.mean(axis=1)


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= EPS:
        return float("nan")
    scale = float(np.dot(estimate, reference)) / (reference_energy + EPS)
    projected = scale * reference
    noise = estimate - projected
    return float(
        10.0
        * np.log10(
            (float(np.dot(projected, projected)) + EPS)
            / (float(np.dot(noise, noise)) + EPS)
        )
    )


def score_trial(task: tuple[dict[str, Any], dict[str, str], dict[str, str], str]) -> dict[str, Any]:
    manifest, candidate_paths, checkpoints, split = task
    if manifest["split"] != split:
        raise ValueError(f"wrong split input rejected: {manifest['trial_id']}")
    target = load_mono16(manifest["target_wav"])
    interferer = load_mono16(manifest["interferer_wav"])
    candidates: dict[str, dict[str, Any]] = {}
    for name, path in candidate_paths.items():
        output = load_mono16(path)
        common = min(output.size, target.size, interferer.size)
        if common <= 0:
            raise ValueError(f"empty waveform: {manifest['trial_id']} {name}")
        target_score = si_sdr(output[:common], target[:common])
        interferer_score = si_sdr(output[:common], interferer[:common])
        if not math.isfinite(target_score) or not math.isfinite(interferer_score):
            raise ValueError(f"non-finite SI-SDR: {manifest['trial_id']} {name}")
        candidates[name] = {
            "output_wav": path,
            "checkpoint": checkpoints[name],
            "common_num_samples": int(common),
            "sisdr_target_db": target_score,
            "sisdr_interferer_db": interferer_score,
            "sisdr_margin_db": target_score - interferer_score,
            "target_correct": target_score - interferer_score > 0.0,
            "strong_target_correct": target_score - interferer_score >= 5.0 and target_score > 0.0,
        }
    return {
        "trial_id": manifest["trial_id"],
        "split": split,
        "cohort": manifest["cohort"],
        "target_speaker": manifest["target_speaker"],
        "interferer_speaker": manifest["interferer_speaker"],
        "target_wav": manifest["target_wav"],
        "interferer_wav": manifest["interferer_wav"],
        "target_enrollment_embedding_path": manifest["target_enrollment_embedding_path"],
        "interferer_enrollment_embedding_path": manifest["interferer_enrollment_embedding_path"],
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    manifest_by_id = {row["trial_id"]: row for row in manifest_rows}
    if len(manifest_rows) != args.expected_trials or len(manifest_by_id) != args.expected_trials:
        raise ValueError(
            f"candidate manifest must contain {args.expected_trials:,} unique frozen DEV trials"
        )

    paths_by_id: dict[str, dict[str, str]] = {trial_id: {} for trial_id in manifest_by_id}
    checkpoints_by_id: dict[str, dict[str, str]] = {trial_id: {} for trial_id in manifest_by_id}
    for source in args.candidate_paths:
        rows = read_jsonl(source)
        if set(row["trial_id"] for row in rows) != set(manifest_by_id):
            raise ValueError(f"candidate path coverage mismatch: {source}")
        for row in rows:
            trial_id = row["trial_id"]
            if row.get("split") != args.split:
                raise ValueError(f"candidate path split mismatch: {source}")
            for name, path in row["candidate_paths"].items():
                if name in paths_by_id[trial_id]:
                    raise ValueError(f"duplicate candidate {name} for {trial_id}")
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
                paths_by_id[trial_id][name] = path
                checkpoints_by_id[trial_id][name] = row["checkpoint"]
    candidate_sets = {tuple(sorted(value)) for value in paths_by_id.values()}
    if len(candidate_sets) != 1 or not next(iter(candidate_sets)):
        raise ValueError(f"inconsistent candidate sets: {candidate_sets}")

    tasks = [
        (manifest_by_id[trial_id], paths_by_id[trial_id], checkpoints_by_id[trial_id], args.split)
        for trial_id in sorted(manifest_by_id)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    started = time.monotonic()
    with temporary.open("w", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, record in enumerate(executor.map(score_trial, tasks, chunksize=8), 1):
                handle.write(json.dumps(record) + "\n")
                if index == 1 or index % args.progress_interval == 0 or index == len(tasks):
                    print(
                        f"acoustic={index}/{len(tasks)} rate={index / max(time.monotonic() - started, 1e-6):.2f}/s",
                        flush=True,
                    )
    temporary.replace(args.output)
    summary = {
        "status": "COMPLETE",
        "split": args.split,
        "trials": len(tasks),
        "candidate_names": list(next(iter(candidate_sets))),
        "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started,
        "test_used": args.split == "test",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
