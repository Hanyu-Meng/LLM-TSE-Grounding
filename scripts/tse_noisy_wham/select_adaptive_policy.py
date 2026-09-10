#!/usr/bin/env python3
"""Fit and freeze the DEV-only adaptive CSG policy without TEST access."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
SNRS = (-5, 0, 5, 10, 15)
BANDS = (-2.5, 2.5, 7.5, 12.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--evaluation", type=Path, required=True)
    calibrate.add_argument("--residual-sidecar", type=Path, required=True)
    calibrate.add_argument("--grid", action="append", required=True,
                           help="LAMBDA=per_trial_metrics.jsonl")
    calibrate.add_argument("--output-policy", type=Path, required=True)
    calibrate.add_argument("--output-sidecar", type=Path, required=True)
    temporal = sub.add_parser("select-temporal")
    temporal.add_argument("--w0", type=Path, required=True)
    temporal.add_argument("--w1", type=Path, required=True)
    temporal.add_argument("--evaluation", type=Path, required=True)
    temporal.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select-policy")
    select.add_argument("--source-metrics", type=Path, required=True)
    select.add_argument("--tse-metrics", type=Path, required=True)
    select.add_argument("--source-sidecar", type=Path, required=True)
    select.add_argument("--tse-sidecar", type=Path, required=True)
    select.add_argument("--temporal-selection", type=Path, required=True)
    select.add_argument("--evaluation", type=Path, required=True)
    select.add_argument("--output-selection", type=Path, required=True)
    select.add_argument("--output-sidecar", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {row["trial_id"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"duplicate trial IDs: {path}")
    return result


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2) + "\n")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty policy evaluation rows")
    wers = np.asarray([float(row["target_WER"]) for row in rows])
    switch = np.asarray([bool(row["content_switch"]) for row in rows])
    high = wers > 0.5
    margin = np.asarray([float(row["speaker_margin"]) for row in rows])
    result = {
        "trials": len(rows),
        "raw_wer": float(np.mean(wers)),
        "content_switch_rate": float(np.mean(switch)),
        "high_error_rate": float(np.mean(high)),
        "speaker_margin": float(np.mean(margin)),
    }
    result["reliability_objective"] = (
        result["raw_wer"]
        + 0.5 * result["content_switch_rate"]
        + 0.25 * result["high_error_rate"]
    )
    return result


def controlled_ids(evaluation: dict[str, dict[str, Any]]) -> list[str]:
    result = [
        trial_id for trial_id, row in evaluation.items()
        if row.get("benchmark") == "controlled"
    ]
    if len(result) != 2400:
        raise ValueError(f"expected 2400 controlled DEV trials, got {len(result)}")
    return result


def calibrate(args: argparse.Namespace) -> int:
    evaluation = keyed(args.evaluation)
    controlled = controlled_ids(evaluation)
    residual = keyed(args.residual_sidecar)
    if set(residual) != set(evaluation):
        raise ValueError("residual sidecar/full DEV coverage mismatch")
    sources: dict[float, dict[str, dict[str, Any]]] = {}
    for spec in args.grid:
        text_lambda, text_path = spec.split("=", 1)
        value = float(text_lambda)
        if value not in LAMBDAS or value in sources:
            raise ValueError(f"invalid/repeated grid lambda: {value}")
        rows = keyed(Path(text_path))
        if set(rows) != set(controlled):
            raise ValueError(f"grid coverage mismatch: {text_path}")
        sources[value] = rows
    if set(sources) != set(LAMBDAS):
        raise ValueError(f"lambda grid incomplete: {sorted(sources)}")
    by_snr: dict[int, dict[float, dict[str, float]]] = {}
    for snr in SNRS:
        ids = [trial_id for trial_id in controlled
               if int(evaluation[trial_id]["snr_db"]) == snr]
        if len(ids) != 480:
            raise ValueError(f"SNR={snr} expected 480, got {len(ids)}")
        by_snr[snr] = {
            value: metrics([sources[value][trial_id] for trial_id in ids])
            for value in LAMBDAS
        }
    candidates = []
    for sequence in itertools.product(LAMBDAS, repeat=len(SNRS)):
        if any(sequence[index] < sequence[index + 1]
               for index in range(len(sequence) - 1)):
            continue
        values = [by_snr[snr][value] for snr, value in zip(SNRS, sequence)]
        candidates.append((
            float(np.mean([row["reliability_objective"] for row in values])),
            float(np.mean([row["raw_wer"] for row in values])),
            float(np.mean([row["content_switch_rate"] for row in values])),
            float(np.mean([row["high_error_rate"] for row in values])),
            tuple(sequence),
        ))
    best = min(candidates)
    sequence = best[-1]
    mapping = {str(snr): value for snr, value in zip(SNRS, sequence)}
    rows = []
    for trial_id in evaluation:
        difficulty = float(residual[trial_id]["difficulty_value"])
        band = int(np.searchsorted(np.asarray(BANDS), difficulty, side="right"))
        value = float(sequence[band])
        rows.append({
            "trial_id": trial_id,
            "policy": "TSE-DEV calibrated difficulty-conditioned CSG",
            "difficulty_value": difficulty,
            "selected_lambda": value,
            "calibrated_snr_band_db": SNRS[band],
            "clean_reference_used": False,
            "test_used": False,
        })
    policy = {
        "status": "FROZEN_ON_DEV",
        "candidate_lambdas": list(LAMBDAS),
        "snr_levels_db": list(SNRS),
        "band_boundaries_db": list(BANDS),
        "selected_lambda_by_snr_band": mapping,
        "objective": "mean raw WER + 0.5*content-switch rate + 0.25*high-error rate",
        "monotonic_constraint": "lambda non-increasing as SNR increases",
        "selection_key": {
            "objective": best[0], "raw_wer": best[1],
            "content_switch_rate": best[2], "high_error_rate": best[3],
            "lambda_sequence": list(sequence),
        },
        "grid_metrics_by_snr": {
            str(snr): {str(value): row for value, row in by_snr[snr].items()}
            for snr in SNRS
        },
        "test_used": False,
    }
    atomic_json(args.output_policy, policy)
    atomic_jsonl(args.output_sidecar, rows)
    print(json.dumps(policy, indent=2))
    return 0


def select_temporal(args: argparse.Namespace) -> int:
    evaluation = keyed(args.evaluation)
    ids = controlled_ids(evaluation)
    w0 = keyed(args.w0)
    w1 = keyed(args.w1)
    if set(w0) != set(ids) or set(w1) != set(ids):
        raise ValueError("w0/w1 controlled coverage mismatch")
    m0 = metrics([w0[trial_id] for trial_id in ids])
    m1 = metrics([w1[trial_id] for trial_id in ids])
    choose_w1 = (
        m1["reliability_objective"] <= m0["reliability_objective"] - 0.0025
        and m1["raw_wer"] <= m0["raw_wer"]
        and m1["speaker_margin"] >= m0["speaker_margin"] - 0.005
    )
    result = {
        "status": "FROZEN_ON_DEV",
        "comparison": "source thresholds on controlled DEV",
        "w0": m0,
        "w1": m1,
        "selected_temporal_tolerance": 1 if choose_w1 else 0,
        "selection_rule": (
            "choose w1 only if objective improves >=0.0025, raw WER is not "
            "worse, and speaker margin drop <=0.005"
        ),
        "test_used": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def select_policy(args: argparse.Namespace) -> int:
    evaluation = keyed(args.evaluation)
    ids = controlled_ids(evaluation)
    source_metrics = keyed(args.source_metrics)
    tse_metrics = keyed(args.tse_metrics)
    if set(source_metrics) != set(ids) or set(tse_metrics) != set(ids):
        raise ValueError("source/TSE policy controlled coverage mismatch")
    source = metrics([source_metrics[trial_id] for trial_id in ids])
    tse = metrics([tse_metrics[trial_id] for trial_id in ids])
    source_preferred = (
        source["reliability_objective"] <= tse["reliability_objective"] + 0.005
        and source["raw_wer"] <= tse["raw_wer"] + 0.005
        and source["content_switch_rate"] <= tse["content_switch_rate"] + 0.005
        and source["high_error_rate"] <= tse["high_error_rate"] + 0.005
    )
    temporal = json.loads(args.temporal_selection.read_text())
    chosen_path = args.source_sidecar if source_preferred else args.tse_sidecar
    chosen = read_jsonl(chosen_path)
    if len(chosen) != 8400 or len({row["trial_id"] for row in chosen}) != 8400:
        raise ValueError("selected full DEV sidecar coverage mismatch")
    name = (
        "Source-threshold difficulty-conditioned CSG"
        if source_preferred else "TSE-DEV calibrated difficulty-conditioned CSG"
    )
    output_rows = [{
        **row,
        "policy": name,
        "selected_temporal_tolerance": temporal["selected_temporal_tolerance"],
        "policy_selection_frozen": True,
    } for row in chosen]
    result = {
        "status": "FROZEN_ON_DEV",
        "source_policy": source,
        "tse_dev_calibrated_policy": tse,
        "selected_policy": name,
        "selected_sidecar_source": str(chosen_path),
        "selected_temporal_tolerance": temporal["selected_temporal_tolerance"],
        "source_preference_rule": (
            "source objective within 0.005 and raw WER/content-switch/high-error "
            "each no more than 0.005 worse"
        ),
        "test_used": False,
    }
    atomic_json(args.output_selection, result)
    atomic_jsonl(args.output_sidecar, output_rows)
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "select-temporal":
        return select_temporal(args)
    return select_policy(args)


if __name__ == "__main__":
    raise SystemExit(main())
