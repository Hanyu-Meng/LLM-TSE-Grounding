#!/usr/bin/env python3
"""Evaluation-only CDCS-2/CDCS-5 noisy candidate analysis and SNR figure."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
CDCS2_CANDIDATES = ("full", "tfmap_context_full")
CDCS5_CANDIDATES = ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=ROOT / "analysis/noisy_wham/dev/full/candidates_evaluation.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis/noisy_wham/dev/full/candidate_analysis",
    )
    parser.add_argument(
        "--figure-prefix", type=Path,
        default=ROOT / "figures/noisy_candidate_recovery_by_snr",
    )
    parser.add_argument("--expected", type=int, default=8400)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2) + "\n")


def mean_bool(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return mean_bool([bool(row[key]) for row in rows])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    swaps = [row for row in rows if row["noisy_primary_swap"]]
    controls = [row for row in rows if row["noisy_primary_correct_control"]]
    result: dict[str, Any] = {
        "trials": len(rows),
        "primary_noisy_swap_count": len(swaps),
        "primary_noisy_swap_rate": len(swaps) / len(rows) if rows else None,
        "primary_noisy_correct_control_count": len(controls),
        "primary_noisy_correct_control_rate": len(controls) / len(rows) if rows else None,
        "selection_counts_cdcs2": dict(Counter(row["cdcs2_direct"] for row in rows)),
        "selection_counts_cdcs5": dict(Counter(row["cdcs5_direct"] for row in rows)),
    }
    for name in ORDER:
        result[f"{name}_target_correct_rate"] = rate(rows, f"{name}_target_correct")
        result[f"{name}_recovery_rate_on_primary_swaps"] = rate(
            swaps, f"{name}_target_correct"
        )
    for pool in ("cdcs2", "cdcs5"):
        result[f"{pool}_selected_target_correct_rate"] = rate(
            rows, f"{pool}_selected_target_correct"
        )
        result[f"{pool}_oracle_availability_rate"] = rate(
            rows, f"{pool}_oracle_available"
        )
        result[f"{pool}_selected_recovery_rate_on_primary_swaps"] = rate(
            swaps, f"{pool}_selected_target_correct"
        )
        result[f"{pool}_oracle_recovery_rate_on_primary_swaps"] = rate(
            swaps, f"{pool}_oracle_available"
        )
        result[f"{pool}_control_regression_rate"] = mean_bool([
            not row[f"{pool}_selected_target_correct"] for row in controls
        ])
        selected = result[f"{pool}_selected_recovery_rate_on_primary_swaps"]
        oracle = result[f"{pool}_oracle_recovery_rate_on_primary_swaps"]
        result[f"{pool}_selector_gap_on_primary_swaps"] = (
            oracle - selected if oracle is not None and selected is not None else None
        )
    result["cdcs5_tfmap_recovery_share"] = (
        float(np.mean([
            row["cdcs5_direct"] == "tfmap_context_full"
            for row in swaps if row["cdcs5_direct_target_correct"]
        ]))
        if any(row["cdcs5_direct_target_correct"] for row in swaps) else None
    )
    result["cdcs5_segment_recovery_share"] = (
        float(np.mean([
            row["cdcs5_direct"] in {"first", "middle", "final"}
            for row in swaps if row["cdcs5_direct_target_correct"]
        ]))
        if any(row["cdcs5_direct_target_correct"] for row in swaps) else None
    )
    # Mutually interpretable recovery sources on the primary-swap cohort.
    # "Overlap" deliberately means cross-family overlap (TF-map and at least
    # one segmented view), rather than overlap with the primary full view.
    tfmap_only = []
    segment_only = []
    overlap = []
    for row in swaps:
        tfmap = bool(row["tfmap_context_full_target_correct"])
        segment = any(bool(row[f"{name}_target_correct"])
                      for name in ("first", "middle", "final"))
        full = bool(row["full_target_correct"])
        tfmap_only.append(tfmap and not full and not segment)
        segment_only.append(segment and not full and not tfmap)
        overlap.append(tfmap and segment)
    result["cdcs5_tfmap_only_recovery_rate_on_primary_swaps"] = mean_bool(tfmap_only)
    result["cdcs5_segment_only_recovery_rate_on_primary_swaps"] = mean_bool(segment_only)
    result["cdcs5_tfmap_segment_overlap_rate_on_primary_swaps"] = mean_bool(overlap)
    result["cdcs5_direct_joint_recovery_rate_on_primary_swaps"] = result[
        "cdcs5_direct_recovery_rate_on_primary_swaps"
    ]
    return result


def complementarity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for a, b in itertools.combinations(ORDER, 2):
        aa = np.asarray([row[f"{a}_target_correct"] for row in rows], dtype=bool)
        bb = np.asarray([row[f"{b}_target_correct"] for row in rows], dtype=bool)
        union = aa | bb
        intersection = aa & bb
        results.append({
            "candidate_a": a,
            "candidate_b": b,
            "trials": len(rows),
            "both_correct": float(np.mean(intersection)) if len(rows) else None,
            "a_only": float(np.mean(aa & ~bb)) if len(rows) else None,
            "b_only": float(np.mean(bb & ~aa)) if len(rows) else None,
            "either_correct": float(np.mean(union)) if len(rows) else None,
            "neither_correct": float(np.mean(~union)) if len(rows) else None,
            "union_gain_over_best": (
                float(np.mean(union)) - max(float(np.mean(aa)), float(np.mean(bb)))
                if len(rows) else None
            ),
            "correct_set_jaccard": (
                float(intersection.sum() / union.sum()) if union.sum() else None
            ),
        })
    return results


def plot_by_snr(rows: list[dict[str, Any]], prefix: Path) -> None:
    controlled = [row for row in rows if row["benchmark"] == "controlled"]
    snrs = (-5, 0, 5, 10, 15)
    series = (
        ("Primary", lambda row: row["full_target_correct"]),
        ("First", lambda row: row["first_target_correct"]),
        ("Middle", lambda row: row["middle_target_correct"]),
        ("Final", lambda row: row["final_target_correct"]),
        ("TF-map", lambda row: row["tfmap_context_full_target_correct"]),
        ("CDCS-2", lambda row: row["cdcs2_direct_target_correct"]),
        ("CDCS-5 cosine", lambda row: row["cdcs5_direct_target_correct"]),
        ("CDCS-5 oracle", lambda row: row["cdcs5_oracle_available"]),
    )
    colors = (
        "#64748B", "#7C3AED", "#A855F7", "#D946EF",
        "#E76F51", "#0EA5E9", "#0F766E", "#F59E0B",
    )
    markers = ("o", "^", "v", "s", "D", "P", "o", "*")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for (label, fn), color, marker in zip(series, colors, markers):
        values = []
        for snr in snrs:
            group = [row for row in controlled if int(row["snr_db"]) == snr]
            values.append(100.0 * float(np.mean([fn(row) for row in group])))
        ax.plot(snrs, values, label=label, color=color, marker=marker,
                linewidth=2.0, markersize=6)
    ax.set_xlabel("Speech-mixture-to-noise SNR (dB)")
    ax.set_ylabel("Target-correct candidate rate (%)")
    ax.set_xticks(snrs)
    ax.set_ylim(0, 102)
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    ax.legend(ncol=2, frameon=False, loc="lower right")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source = read_jsonl(args.evaluation)
    ids = [row["trial_id"] for row in source]
    if len(source) != args.expected or len(set(ids)) != args.expected:
        raise ValueError("candidate evaluation coverage mismatch")
    if any(row.get("split") != args.split for row in source):
        raise ValueError("candidate evaluation split mismatch")
    trials = []
    for row in source:
        candidates = row["candidates"]
        if tuple(candidates) != ORDER:
            raise ValueError(f"candidate order mismatch: {row['trial_id']}")
        primary = candidates["full"]
        cdcs2_direct = row["selected"]["cdcs2"]
        cdcs5_direct = row["selected"]["cdcs5"]
        oracle_b = max(
            CDCS2_CANDIDATES,
            key=lambda name: (float(candidates[name]["sisdr_margin_db"]), -ORDER.index(name)),
        )
        oracle_d = max(
            CDCS5_CANDIDATES,
            key=lambda name: (float(candidates[name]["sisdr_margin_db"]), -ORDER.index(name)),
        )
        trial = {
            "trial_id": row["trial_id"],
            "base_trial_id": row["base_trial_id"],
            "split": row["split"],
            "benchmark": row["benchmark"],
            "cohort": row["cohort"],
            "gender_cohort": row["gender_cohort"],
            "snr_db": row.get("snr_db"),
            "noisy_primary_swap": (
                float(primary["sisdr_margin_db"]) < -5.0
                and float(primary["sisdr_interferer_db"]) > 0.0
            ),
            "noisy_primary_correct_control": (
                float(primary["sisdr_margin_db"]) >= 5.0
                and float(primary["sisdr_target_db"]) > 0.0
            ),
            "cdcs2_direct": cdcs2_direct,
            "cdcs5_direct": cdcs5_direct,
            "cdcs2_oracle": oracle_b,
            "cdcs5_oracle": oracle_d,
            "cdcs2_direct_target_correct": bool(candidates[cdcs2_direct]["target_correct"]),
            "cdcs5_direct_target_correct": bool(candidates[cdcs5_direct]["target_correct"]),
            "cdcs2_oracle_available": any(
                candidates[name]["target_correct"] for name in CDCS2_CANDIDATES
            ),
            "cdcs5_oracle_available": any(
                candidates[name]["target_correct"] for name in CDCS5_CANDIDATES
            ),
            "cdcs2_direct_speaker_margin": float(
                candidates[cdcs2_direct]["speaker_embedding_margin"]
            ),
            "cdcs5_direct_speaker_margin": float(
                candidates[cdcs5_direct]["speaker_embedding_margin"]
            ),
            "test_used": args.split == "test",
        }
        for name in ORDER:
            trial[f"{name}_target_correct"] = bool(candidates[name]["target_correct"])
            trial[f"{name}_strong_target_correct"] = bool(
                candidates[name]["strong_target_correct"]
            )
            trial[f"{name}_sisdr_margin_db"] = float(candidates[name]["sisdr_margin_db"])
            trial[f"{name}_enrollment_cosine"] = float(
                candidates[name]["speaker_similarity_to_enrollment"]
            )
        trials.append(trial)

    natural = [row for row in trials if row["benchmark"] == "natural"]
    controlled = [row for row in trials if row["benchmark"] == "controlled"]
    result = {
        "status": "COMPLETE",
        "split": args.split,
        "definitions": {
            "target_correct": "candidate SI-SDR target-minus-interferer margin > 0 dB",
            "noisy_primary_swap": "full margin < -5 dB and SI-SDR to interferer > 0 dB",
            "noisy_primary_correct_control": "full margin >= 5 dB and SI-SDR to target > 0 dB",
            "selection": "frozen enrollment cosine; no clean reference",
            "oracle": "evaluation-only maximum target-minus-interferer SI-SDR margin",
        },
        "all": summarize(trials),
        "natural": summarize(natural),
        "controlled": summarize(controlled),
        "controlled_by_snr": {
            str(snr): summarize([row for row in controlled if int(row["snr_db"]) == snr])
            for snr in (-5, 0, 5, 10, 15)
        },
        "natural_same_gender": summarize([
            row for row in natural if row["gender_cohort"] == "same"
        ]),
        "natural_different_gender": summarize([
            row for row in natural if row["gender_cohort"] == "different"
        ]),
        "natural_complementarity": complementarity(natural),
        "natural_primary_swap_complementarity": complementarity([
            row for row in natural if row["noisy_primary_swap"]
        ]),
        "controlled_complementarity": complementarity(controlled),
        "clean_primary_swap_reference": {
            "count": 405 if args.split == "dev" else 398,
            "trials": 6000,
            "rate": (405 if args.split == "dev" else 398) / 6000,
        },
        "test_used": args.split == "test",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_dir / "per_trial_candidate_analysis.jsonl",
        "".join(json.dumps(row) + "\n" for row in trials),
    )
    atomic_json(args.output_dir / "candidate_analysis.json", result)
    plot_by_snr(trials, args.figure_prefix)
    print(json.dumps({
        "status": "COMPLETE",
        "split": args.split,
        "trials": len(trials),
        "natural_primary_swap_rate": result["natural"]["primary_noisy_swap_rate"],
        "cdcs5_recovery": result["natural"]["cdcs5_direct_recovery_rate_on_primary_swaps"],
        "cdcs5_oracle_recovery": result["natural"]["cdcs5_oracle_recovery_rate_on_primary_swaps"],
        "test_used": args.split == "test",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
