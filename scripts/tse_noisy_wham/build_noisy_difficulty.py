#!/usr/bin/env python3
"""Compute deployable residual difficulty and freeze adaptive CSG sidecars."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
LAMBDAS = (2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0)
THRESHOLDS = (-3.013, -1.640, -0.267, 1.179, 2.697, 4.216, 7.515, 13.158)
EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    unit = sub.add_parser("unit")
    unit.add_argument("--evaluation", type=Path, required=True)
    unit.add_argument("--output", type=Path, required=True)
    residual = sub.add_parser("residual")
    residual.add_argument("--inference", type=Path, required=True)
    residual.add_argument("--output", type=Path, required=True)
    residual.add_argument("--status-every", type=int, default=100)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--residual", type=Path, required=True)
    calibrate.add_argument("--evaluation", type=Path, required=True)
    calibrate.add_argument("--calibration-output", type=Path, required=True)
    calibrate.add_argument("--residual-sidecar", type=Path, required=True)
    calibrate.add_argument("--known-sidecar", type=Path, required=True)
    calibrate.add_argument("--controlled-inference", type=Path, required=True)
    calibrate.add_argument("--inference", type=Path, required=True)
    apply_frozen = sub.add_parser("apply-frozen")
    apply_frozen.add_argument("--residual", type=Path, required=True)
    apply_frozen.add_argument("--calibration", type=Path, required=True)
    apply_frozen.add_argument("--inference", type=Path, required=True)
    apply_frozen.add_argument("--output", type=Path, required=True)
    apply_frozen.add_argument("--adaptive-choice", type=Path, required=True)
    apply_frozen.add_argument("--tse-policy", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2) + "\n")


def select_lambda(value: float) -> float:
    index = int(np.searchsorted(np.asarray(THRESHOLDS), value, side="right"))
    return float(LAMBDAS[index])


def unit(evaluation_path: Path, output: Path) -> int:
    rows = read_jsonl(evaluation_path)
    sidecar = []
    for row in rows:
        value = float(row["snr_db"]) if row["benchmark"] == "controlled" else 0.0
        sidecar.append({
            "trial_id": row["trial_id"],
            "policy": "unit_known_snr_for_controlled_fixed_zero_for_natural",
            "difficulty_value": value,
            "selected_lambda": select_lambda(value),
            "test_used": False,
        })
    atomic_jsonl(output, sidecar)
    return 0


def load_mono(path: str) -> np.ndarray:
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    if int(rate) != 16000:
        raise ValueError(f"expected 16 kHz: {path}")
    waveform = values.mean(axis=1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid waveform: {path}")
    return waveform


def residual(inference_path: Path, output: Path, status_every: int) -> int:
    rows = read_jsonl(inference_path)
    records = []
    for index, row in enumerate(rows, 1):
        mixture = load_mono(row["mixture_wav"])
        evidence = load_mono(row["cdcs5_evidence_waveform"])
        if len(mixture) != len(evidence):
            raise ValueError(f"residual length mismatch: {row['trial_id']}")
        dot = float(np.dot(mixture, evidence))
        energy = float(np.dot(evidence, evidence))
        alpha = max(0.0, dot / (energy + EPS))
        scaled = alpha * evidence
        remainder = mixture - scaled
        numerator = float(np.dot(scaled, scaled))
        denominator = float(np.dot(remainder, remainder))
        xi = float(10.0 * np.log10((numerator + EPS) / (denominator + EPS)))
        if not all(math.isfinite(value) for value in (alpha, xi)):
            raise ValueError(f"non-finite residual proxy: {row['trial_id']}")
        records.append({
            "trial_id": row["trial_id"],
            "alpha": alpha,
            "xi_raw_db": xi,
            "evidence_energy": energy,
            "scaled_evidence_energy": numerator,
            "residual_energy": denominator,
            "clean_reference_used": False,
            "shift_search_used": False,
            "test_used": row.get("split") == "test",
        })
        if index == 1 or index % status_every == 0 or index == len(rows):
            print(f"residual={index}/{len(rows)}", flush=True)
    atomic_jsonl(output, records)
    return 0


def calibrate(args: argparse.Namespace) -> int:
    residual_rows = read_jsonl(args.residual)
    residual_by_id = {row["trial_id"]: row for row in residual_rows}
    evaluation = read_jsonl(args.evaluation)
    evaluation_by_id = {row["trial_id"]: row for row in evaluation}
    if set(residual_by_id) != set(evaluation_by_id):
        raise ValueError("residual/evaluation coverage mismatch")
    controlled = [row for row in evaluation if row["benchmark"] == "controlled"]
    x = np.asarray([residual_by_id[row["trial_id"]]["xi_raw_db"] for row in controlled])
    y = np.asarray([float(row["snr_db"]) for row in controlled])
    slope, intercept, low_slope, high_slope = stats.theilslopes(y, x, method="separate")
    predicted = slope * x + intercept
    errors = predicted - y
    tir = []
    genders = []
    for row in controlled:
        target = load_mono(row["target_wav"])
        interferer = load_mono(row["interferer_wav"])
        length = min(len(target), len(interferer))
        tir.append(float(10.0 * np.log10(
            (float(np.dot(target[:length], target[:length])) + EPS)
            / (float(np.dot(interferer[:length], interferer[:length])) + EPS)
        )))
        genders.append(str(row["gender_cohort"]))
    tir_array = np.asarray(tir)
    gender_array = np.asarray(genders)
    tir_edges = np.quantile(tir_array, (0.25, 0.50, 0.75))
    tir_bin = np.searchsorted(tir_edges, tir_array, side="right")
    spearman = float(stats.spearmanr(x, y).statistic)
    pearson = float(stats.pearsonr(x, y).statistic)
    mae = float(np.mean(np.abs(errors)))
    valid = spearman >= 0.50 and mae <= 4.0
    calibration = {
        "status": "COMPLETE",
        "method": "scipy.stats.theilslopes method=separate",
        "n": len(controlled),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci95": [float(low_slope), float(high_slope)],
        "mae_db": mae,
        "rmse_db": float(np.sqrt(np.mean(errors ** 2))),
        "median_absolute_error_db": float(np.median(np.abs(errors))),
        "spearman": spearman,
        "pearson": pearson,
        "bias_by_snr_db": {
            str(snr): float(np.mean(errors[y == snr])) for snr in (-5, 0, 5, 10, 15)
        },
        "tir_evaluation_only": {
            "definition": "10log10(target waveform energy / interferer waveform energy)",
            "mean_db": float(np.mean(tir_array)),
            "quartile_edges_db": [float(value) for value in tir_edges],
            "spearman_xi_vs_tir": float(stats.spearmanr(x, tir_array).statistic),
            "spearman_error_vs_tir": float(stats.spearmanr(errors, tir_array).statistic),
            "by_quartile": {
                str(index): {
                    "n": int(np.sum(tir_bin == index)),
                    "bias_db": float(np.mean(errors[tir_bin == index])),
                    "mae_db": float(np.mean(np.abs(errors[tir_bin == index]))),
                }
                for index in range(4)
            },
        },
        "speaker_gender_relation_evaluation_only": {
            gender: {
                "n": int(np.sum(gender_array == gender)),
                "bias_db": float(np.mean(errors[gender_array == gender])),
                "mae_db": float(np.mean(np.abs(errors[gender_array == gender]))),
                "spearman": float(stats.spearmanr(
                    x[gender_array == gender], y[gender_array == gender]
                ).statistic),
            }
            for gender in sorted(set(genders))
        },
        "validity_gate": {"spearman_min": 0.50, "mae_max_db": 4.0},
        "residual_snr_valid": valid,
        "policy_name": "Residual-SNR CSG" if valid else "Residual-Ratio CSG",
        "test_used": False,
    }
    atomic_json(args.calibration_output, calibration)

    inference_rows = read_jsonl(args.inference)
    residual_sidecar = []
    for row in inference_rows:
        raw = float(residual_by_id[row["trial_id"]]["xi_raw_db"])
        estimated = float(slope * raw + intercept)
        residual_sidecar.append({
            "trial_id": row["trial_id"],
            "policy": calibration["policy_name"],
            "xi_raw_db": raw,
            "difficulty_value": estimated,
            "calibrated_residual_score_db": estimated,
            "estimated_snr_db": estimated if valid else None,
            "selected_lambda": select_lambda(estimated),
            "test_used": False,
        })
    atomic_jsonl(args.residual_sidecar, residual_sidecar)

    controlled_ids = {row["trial_id"] for row in controlled}
    controlled_inference = [row for row in inference_rows if row["trial_id"] in controlled_ids]
    atomic_jsonl(args.controlled_inference, controlled_inference)
    known = [{
        "trial_id": row["trial_id"],
        "policy": "Known-SNR CSG diagnostic oracle",
        "difficulty_value": float(evaluation_by_id[row["trial_id"]]["snr_db"]),
        "selected_lambda": select_lambda(float(evaluation_by_id[row["trial_id"]]["snr_db"])),
        "test_used": False,
    } for row in controlled_inference]
    atomic_jsonl(args.known_sidecar, known)
    print(json.dumps(calibration, indent=2))
    return 0


def apply_frozen(args: argparse.Namespace) -> int:
    calibration = json.loads(args.calibration.read_text())
    if (
        calibration.get("status") != "COMPLETE"
        or calibration.get("test_used")
        or not math.isfinite(float(calibration["slope"]))
        or not math.isfinite(float(calibration["intercept"]))
    ):
        raise ValueError("invalid frozen DEV residual calibration")
    residual_rows = read_jsonl(args.residual)
    residual_by_id = {row["trial_id"]: row for row in residual_rows}
    inference = read_jsonl(args.inference)
    ids = [row["trial_id"] for row in inference]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(residual_by_id)
        or any(row.get("split") not in {"dev", "test"} for row in inference)
    ):
        raise ValueError("frozen-calibration inference/residual coverage mismatch")
    slope = float(calibration["slope"])
    intercept = float(calibration["intercept"])
    policy_name = str(calibration["policy_name"])
    choice = json.loads(args.adaptive_choice.read_text())
    tse_policy = json.loads(args.tse_policy.read_text())
    selected_policy = str(choice["selected_policy"])
    temporal_tolerance = int(choice["selected_temporal_tolerance"])
    if selected_policy == "TSE-DEV calibrated difficulty-conditioned CSG":
        sequence = tuple(
            float(tse_policy["selected_lambda_by_snr_band"][str(snr)])
            for snr in (-5, 0, 5, 10, 15)
        )
        boundaries = np.asarray(tse_policy["band_boundaries_db"], dtype=float)
    elif selected_policy == "Source-threshold difficulty-conditioned CSG":
        sequence = None
        boundaries = None
    else:
        raise ValueError(f"unknown frozen adaptive policy: {selected_policy}")
    sidecar = []
    for row in inference:
        raw = float(residual_by_id[row["trial_id"]]["xi_raw_db"])
        estimated = slope * raw + intercept
        if not math.isfinite(estimated):
            raise ValueError(f"non-finite frozen difficulty: {row['trial_id']}")
        selected_lambda = (
            select_lambda(estimated)
            if sequence is None else float(sequence[int(np.searchsorted(
                boundaries, estimated, side="right"
            ))])
        )
        sidecar.append({
            "trial_id": row["trial_id"],
            "policy": selected_policy,
            "xi_raw_db": raw,
            "difficulty_value": estimated,
            "calibrated_residual_score_db": estimated,
            "estimated_snr_db": estimated if calibration["residual_snr_valid"] else None,
            "selected_lambda": selected_lambda,
            "selected_temporal_tolerance": temporal_tolerance,
            "calibration_source": str(args.calibration),
            "test_used": row.get("split") == "test",
        })
    atomic_jsonl(args.output, sidecar)
    print(json.dumps({
        "status": "COMPLETE",
        "rows": len(sidecar),
        "policy_name": selected_policy,
        "selected_temporal_tolerance": temporal_tolerance,
        "residual_snr_valid": calibration["residual_snr_valid"],
        "calibration_refit": False,
        "test_used": any(row["test_used"] for row in sidecar),
    }, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "unit":
        return unit(args.evaluation, args.output)
    if args.command == "residual":
        return residual(args.inference, args.output, args.status_every)
    if args.command == "apply-frozen":
        return apply_frozen(args)
    return calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
