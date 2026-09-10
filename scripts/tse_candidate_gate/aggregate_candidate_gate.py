#!/usr/bin/env python3
"""Join candidate metrics and emit the frozen alternative-candidate gate artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MULTIVIEW = ["full", "first", "middle", "final"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic-metrics", type=Path, required=True)
    parser.add_argument("--speaker-metrics", type=Path, required=True)
    parser.add_argument("--asr-metrics", type=Path, required=True)
    parser.add_argument(
        "--preregistered-gate",
        type=Path,
        default=PROJECT_ROOT / "analysis/candidate_gate/preregistered_gate.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "analysis/candidate_gate"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def rate(values: list[Any]) -> float | None:
    return float(np.mean([bool(value) for value in values])) if values else None


def wilson(successes: int, total: int) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return {"low": center - half, "high": center + half}


def candidate_order(names: list[str]) -> list[str]:
    return [name for name in MULTIVIEW if name in names] + sorted(
        name for name in names if name not in MULTIVIEW
    )


def aggregate_candidate(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    metrics = [row["candidates"][candidate] for row in rows]
    correct_count = sum(bool(value["target_correct"]) for value in metrics)
    return {
        "candidate": candidate,
        "count": len(metrics),
        "target_correct_count": correct_count,
        "target_correct_rate": correct_count / len(metrics) if metrics else None,
        "target_correct_rate_ci95_wilson": wilson(correct_count, len(metrics)),
        "strong_target_correct_count": sum(bool(value["strong_target_correct"]) for value in metrics),
        "strong_target_correct_rate": rate([value["strong_target_correct"] for value in metrics]),
        "sisdr_target_db_mean": finite_mean([value["sisdr_target_db"] for value in metrics]),
        "sisdr_interferer_db_mean": finite_mean([value["sisdr_interferer_db"] for value in metrics]),
        "sisdr_margin_db_mean": finite_mean([value["sisdr_margin_db"] for value in metrics]),
        "speaker_similarity_to_enrollment_mean": finite_mean(
            [value["speaker_similarity_to_enrollment"] for value in metrics]
        ),
        "speaker_embedding_margin_mean": finite_mean(
            [value["speaker_embedding_margin"] for value in metrics]
        ),
        "target_WER_mean": finite_mean([value["target_WER"] for value in metrics]),
        "target_WER_median": (
            float(np.median([value["target_WER"] for value in metrics if value["target_WER"] is not None]))
            if any(value["target_WER"] is not None for value in metrics)
            else None
        ),
        "content_switch_count": sum(bool(value["content_switch"]) for value in metrics),
        "content_switch_rate": rate([value["content_switch"] for value in metrics]),
    }


def chosen_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    choices = [row["selection"][key] for row in rows]
    values = [row["candidates"][choice] for row, choice in zip(rows, choices)]
    correct_count = sum(bool(value["target_correct"]) for value in values)
    return {
        "selection": key,
        "count": len(rows),
        "selected_candidate_counts": {
            candidate: choices.count(candidate) for candidate in sorted(set(choices))
        },
        "target_correct_count": correct_count,
        "target_correct_rate": correct_count / len(rows) if rows else None,
        "target_correct_rate_ci95_wilson": wilson(correct_count, len(rows)),
        "wrong_speaker_count": len(rows) - correct_count,
        "wrong_speaker_rate": 1.0 - correct_count / len(rows) if rows else None,
        "sisdr_target_db_mean": finite_mean([value["sisdr_target_db"] for value in values]),
        "sisdr_interferer_db_mean": finite_mean([value["sisdr_interferer_db"] for value in values]),
        "sisdr_margin_db_mean": finite_mean([value["sisdr_margin_db"] for value in values]),
        "speaker_similarity_to_enrollment_mean": finite_mean(
            [value["speaker_similarity_to_enrollment"] for value in values]
        ),
        "speaker_embedding_margin_mean": finite_mean(
            [value["speaker_embedding_margin"] for value in values]
        ),
        "target_WER_mean": finite_mean([value["target_WER"] for value in values]),
        "content_switch_count": sum(bool(value["content_switch"]) for value in values),
        "content_switch_rate": rate([value["content_switch"] for value in values]),
    }


def pool_summary(rows: list[dict[str, Any]], pool_name: str, pool: list[str]) -> dict[str, Any]:
    correct_any = [any(row["candidates"][name]["target_correct"] for name in pool) for row in rows]
    count = sum(correct_any)
    return {
        "pool": pool_name,
        "candidate_names": pool,
        "count": len(rows),
        "at_least_one_target_correct_count": count,
        "oracle_candidate_recovery_rate": count / len(rows) if rows else None,
        "oracle_candidate_recovery_rate_ci95_wilson": wilson(count, len(rows)),
        "oracle_final_wrong_speaker_count": len(rows) - count,
        "oracle_candidate_wrong_speaker_rate": 1.0 - count / len(rows) if rows else None,
        "oracle_selection": chosen_metrics(rows, f"oracle::{pool_name}"),
        "cosine_selection": chosen_metrics(rows, f"cosine::{pool_name}"),
    }


def complementarity(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    output = []
    for a, b in itertools.combinations(names, 2):
        av = [bool(row["candidates"][a]["target_correct"]) for row in rows]
        bv = [bool(row["candidates"][b]["target_correct"]) for row in rows]
        both = sum(x and y for x, y in zip(av, bv))
        a_only = sum(x and not y for x, y in zip(av, bv))
        b_only = sum(not x and y for x, y in zip(av, bv))
        either = both + a_only + b_only
        output.append(
            {
                "candidate_a": a,
                "candidate_b": b,
                "count": len(rows),
                "a_correct_count": sum(av),
                "b_correct_count": sum(bv),
                "both_correct_count": both,
                "a_only_correct_count": a_only,
                "b_only_correct_count": b_only,
                "either_correct_count": either,
                "neither_correct_count": len(rows) - either,
                "union_recovery_rate": either / len(rows) if rows else None,
                "incremental_union_over_best_individual_rate": (
                    (either - max(sum(av), sum(bv))) / len(rows) if rows else None
                ),
                "correct_set_jaccard": both / either if either else None,
            }
        )
    return output


def flatten_summary_row(
    cohort: str, kind: str, name: str, summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "row_kind": kind,
        "candidate_or_selection": name,
        "count": summary.get("count"),
        "target_correct_count": summary.get("target_correct_count"),
        "target_correct_rate": summary.get("target_correct_rate"),
        "wrong_speaker_count": summary.get("wrong_speaker_count"),
        "wrong_speaker_rate": summary.get("wrong_speaker_rate"),
        "strong_target_correct_count": summary.get("strong_target_correct_count"),
        "strong_target_correct_rate": summary.get("strong_target_correct_rate"),
        "sisdr_target_db_mean": summary.get("sisdr_target_db_mean"),
        "sisdr_interferer_db_mean": summary.get("sisdr_interferer_db_mean"),
        "sisdr_margin_db_mean": summary.get("sisdr_margin_db_mean"),
        "speaker_similarity_to_enrollment_mean": summary.get(
            "speaker_similarity_to_enrollment_mean"
        ),
        "speaker_embedding_margin_mean": summary.get("speaker_embedding_margin_mean"),
        "target_WER_mean": summary.get("target_WER_mean"),
        "target_WER_median": summary.get("target_WER_median"),
        "content_switch_count": summary.get("content_switch_count"),
        "content_switch_rate": summary.get("content_switch_rate"),
        "selected_candidate_counts": json.dumps(
            summary.get("selected_candidate_counts"), sort_keys=True
        )
        if summary.get("selected_candidate_counts") is not None
        else None,
    }


def main() -> int:
    args = parse_args()
    preregistered = json.loads(args.preregistered_gate.read_text(encoding="utf-8"))
    acoustic_rows = read_jsonl(args.acoustic_metrics)
    if len(acoustic_rows) != 5991 or len({row["trial_id"] for row in acoustic_rows}) != 5991:
        raise ValueError("acoustic metrics must contain 5,991 unique DEV trials")
    if any(row.get("split") != "dev" for row in acoustic_rows):
        raise ValueError("non-DEV acoustic row rejected")
    speaker_rows = read_jsonl(args.speaker_metrics)
    asr_rows = read_jsonl(args.asr_metrics)
    expected_candidates = set(acoustic_rows[0]["candidates"])
    expected_keys = {
        (row["trial_id"], candidate)
        for row in acoustic_rows
        for candidate in row["candidates"]
    }
    speaker_by_key = {(row["trial_id"], row["candidate"]): row for row in speaker_rows}
    asr_by_key = {(row["trial_id"], row["candidate"]): row for row in asr_rows}
    if set(speaker_by_key) != expected_keys or set(asr_by_key) != expected_keys:
        raise ValueError("speaker/ASR metric coverage does not exactly match acoustic metrics")
    if any(set(row["candidates"]) != expected_candidates for row in acoustic_rows):
        raise ValueError("inconsistent candidate sets")
    names = candidate_order(list(expected_candidates))
    if not set(MULTIVIEW).issubset(names):
        raise ValueError(f"missing required multiview candidates: {set(MULTIVIEW) - set(names)}")

    joined: list[dict[str, Any]] = []
    pools = {"multiview": MULTIVIEW, "expanded": names}
    for acoustic in acoustic_rows:
        candidates: dict[str, dict[str, Any]] = {}
        for candidate in names:
            key = (acoustic["trial_id"], candidate)
            metrics = dict(acoustic["candidates"][candidate])
            speaker = speaker_by_key[key]
            asr = asr_by_key[key]
            if speaker["output_wav"] != metrics["output_wav"] or asr["output_wav"] != metrics["output_wav"]:
                raise ValueError(f"output path mismatch: {key}")
            metrics.update(
                {
                    "candidate_embedding_path": speaker["candidate_embedding_path"],
                    "speaker_similarity_to_enrollment": speaker[
                        "speaker_similarity_to_enrollment"
                    ],
                    "speaker_similarity_to_interferer_enrollment": speaker[
                        "speaker_similarity_to_interferer_enrollment"
                    ],
                    "speaker_embedding_margin": speaker["speaker_embedding_margin"],
                    "target_WER": asr["target_WER"],
                    "interferer_WER": asr["interferer_WER"],
                    "content_switch": asr["content_switch"],
                    "target_text": asr["target_text"],
                    "interferer_text": asr["interferer_text"],
                    "output_text": asr["output_text"],
                }
            )
            candidates[candidate] = metrics
        selection: dict[str, str] = {}
        for pool_name, pool in pools.items():
            selection[f"oracle::{pool_name}"] = max(
                pool, key=lambda name: (candidates[name]["sisdr_margin_db"], -names.index(name))
            )
            selection[f"cosine::{pool_name}"] = max(
                pool,
                key=lambda name: (
                    candidates[name]["speaker_similarity_to_enrollment"],
                    -names.index(name),
                ),
            )
        joined.append(
            {
                key: acoustic[key]
                for key in (
                    "trial_id",
                    "split",
                    "cohort",
                    "target_speaker",
                    "interferer_speaker",
                    "target_wav",
                    "interferer_wav",
                    "target_enrollment_embedding_path",
                    "interferer_enrollment_embedding_path",
                )
            }
            | {"candidates": candidates, "selection": selection}
        )

    swaps = [row for row in joined if row["cohort"] == "natural_primary_swap"]
    controls = [row for row in joined if row["cohort"] == "primary_correct_control"]
    if len(swaps) != 405 or len(controls) != 5586:
        raise ValueError(f"frozen cohort mismatch: swaps={len(swaps)} controls={len(controls)}")
    if any(row["candidates"]["full"]["target_correct"] for row in swaps):
        raise ValueError("frozen high-confidence swap subset no longer all primary-wrong")
    if any(not row["candidates"]["full"]["target_correct"] for row in controls):
        raise ValueError("frozen primary-correct control subset no longer all primary-correct")

    individual = {
        cohort_name: {
            candidate: aggregate_candidate(rows, candidate) for candidate in names
        }
        for cohort_name, rows in (
            ("natural_primary_swap", swaps),
            ("primary_correct_control", controls),
        )
    }
    pool_results = {
        cohort_name: {
            pool_name: pool_summary(rows, pool_name, pool)
            for pool_name, pool in pools.items()
        }
        for cohort_name, rows in (
            ("natural_primary_swap", swaps),
            ("primary_correct_control", controls),
        )
    }
    all_multiview_alternatives_correct = rate(
        [
            all(row["candidates"][name]["target_correct"] for name in ("first", "middle", "final"))
            for row in controls
        ]
    )
    multiview_instability = rate(
        [
            len({row["candidates"][name]["target_correct"] for name in MULTIVIEW}) > 1
            for row in controls
        ]
    )
    expanded_instability = rate(
        [
            len({row["candidates"][name]["target_correct"] for name in names}) > 1
            for row in controls
        ]
    )
    multiview_correct_ids = {
        row["trial_id"]
        for row in swaps
        if any(row["candidates"][name]["target_correct"] for name in MULTIVIEW)
    }
    independent_names = [name for name in names if name not in MULTIVIEW]
    independent_correct_ids = {
        row["trial_id"]
        for row in swaps
        if any(row["candidates"][name]["target_correct"] for name in independent_names)
    }

    independent_present = names != MULTIVIEW
    decision_pool = "expanded" if independent_present else "multiview"
    swap_pool = pool_results["natural_primary_swap"][decision_pool]
    control_pool = pool_results["primary_correct_control"][decision_pool]
    thresholds = preregistered["go_thresholds"]
    checks = {
        "oracle_candidate_recovery": {
            "value": swap_pool["oracle_candidate_recovery_rate"],
            "operator": ">=",
            "threshold": thresholds["oracle_candidate_recovery_rate_min"],
            "pass": swap_pool["oracle_candidate_recovery_rate"]
            >= thresholds["oracle_candidate_recovery_rate_min"],
        },
        "cosine_selected_swap_recovery": {
            "value": swap_pool["cosine_selection"]["target_correct_rate"],
            "operator": ">=",
            "threshold": thresholds["cosine_selected_swap_recovery_rate_min"],
            "pass": swap_pool["cosine_selection"]["target_correct_rate"]
            >= thresholds["cosine_selected_swap_recovery_rate_min"],
        },
        "cosine_selected_control_regression": {
            "value": control_pool["cosine_selection"]["wrong_speaker_rate"],
            "operator": "<=",
            "threshold": thresholds["cosine_selected_correct_case_regression_rate_max"],
            "pass": control_pool["cosine_selection"]["wrong_speaker_rate"]
            <= thresholds["cosine_selected_correct_case_regression_rate_max"],
        },
        "all_multiview_alternatives_correct_retention": {
            "value": all_multiview_alternatives_correct,
            "operator": ">=",
            "threshold": thresholds["all_alternative_views_correct_retention_rate_min"],
            "pass": all_multiview_alternatives_correct
            >= thresholds["all_alternative_views_correct_retention_rate_min"],
        },
    }
    decision = "GO" if all(check["pass"] for check in checks.values()) else "NO-GO"

    results = {
        "status": "COMPLETE",
        "scope": {
            "split": "dev",
            "natural_primary_swap_count": len(swaps),
            "primary_correct_control_count": len(controls),
            "excluded_ambiguous_primary_wrong_count": 9,
            "test_used": False,
            "training_used": False,
            "selector_trained": False,
        },
        "candidate_names": names,
        "candidate_pools": pools,
        "individual_candidate_results": individual,
        "natural_primary_swap": pool_results["natural_primary_swap"],
        "primary_correct_control": pool_results["primary_correct_control"],
        "correct_case_stability": {
            "alternative_candidate_target_correct_rates": {
                name: individual["primary_correct_control"][name]["target_correct_rate"]
                for name in names
                if name != "full"
            },
            "all_first_middle_final_correct_retention_rate": all_multiview_alternatives_correct,
            "multiview_candidate_instability_rate": multiview_instability,
            "expanded_candidate_instability_rate": expanded_instability,
            "oracle_multiview_regression_rate": pool_results["primary_correct_control"][
                "multiview"
            ]["oracle_selection"]["wrong_speaker_rate"],
            "cosine_multiview_regression_rate": pool_results["primary_correct_control"][
                "multiview"
            ]["cosine_selection"]["wrong_speaker_rate"],
            "oracle_expanded_regression_rate": pool_results["primary_correct_control"][
                "expanded"
            ]["oracle_selection"]["wrong_speaker_rate"],
            "cosine_expanded_regression_rate": pool_results["primary_correct_control"][
                "expanded"
            ]["cosine_selection"]["wrong_speaker_rate"],
        },
        "pairwise_complementarity": {
            "natural_primary_swap": complementarity(swaps, names),
            "primary_correct_control": complementarity(controls, names),
            "multiview_pool_vs_independent_pool_on_natural_primary_swap": {
                "multiview_correct_count": len(multiview_correct_ids),
                "independent_correct_count": len(independent_correct_ids),
                "both_correct_count": len(multiview_correct_ids & independent_correct_ids),
                "multiview_only_correct_count": len(
                    multiview_correct_ids - independent_correct_ids
                ),
                "independent_only_correct_count": len(
                    independent_correct_ids - multiview_correct_ids
                ),
                "either_correct_count": len(multiview_correct_ids | independent_correct_ids),
                "neither_correct_count": len(swaps)
                - len(multiview_correct_ids | independent_correct_ids),
            },
        },
        "gate_evaluation": {
            "preregistered_gate_path": str(args.preregistered_gate.resolve()),
            "decision_pool": decision_pool,
            "independent_frozen_extractor_present": independent_present,
            "checks": checks,
            "decision": decision,
            "failed_checks": [name for name, check in checks.items() if not check["pass"]],
        },
        "metric_definitions": preregistered["metric_definitions"],
        "data_quality": {
            "candidate_metric_key_count": len(expected_keys),
            "speaker_metric_key_count": len(speaker_by_key),
            "asr_metric_key_count": len(asr_by_key),
            "exact_metric_coverage": True,
            "full_primary_matches_frozen_cohort_labels": True,
            "selection_uses_clean_references": {
                "oracle": True,
                "cosine": False,
            },
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_trial_path = args.output_dir / "per_trial_candidates.jsonl"
    per_trial_path.write_text(
        "".join(json.dumps(row) + "\n" for row in joined), encoding="utf-8"
    )
    results_path = args.output_dir / "oracle_candidate_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    csv_rows = []
    for cohort_name, rows in (
        ("natural_primary_swap", swaps),
        ("primary_correct_control", controls),
    ):
        for candidate in names:
            csv_rows.append(
                flatten_summary_row(
                    cohort_name,
                    "individual_candidate",
                    candidate,
                    individual[cohort_name][candidate],
                )
            )
        for pool_name in pools:
            for selection_name in ("oracle_selection", "cosine_selection"):
                summary = pool_results[cohort_name][pool_name][selection_name]
                csv_rows.append(
                    flatten_summary_row(
                        cohort_name,
                        f"{pool_name}_{selection_name}",
                        summary["selection"],
                        summary,
                    )
                )
    csv_path = args.output_dir / "candidate_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "candidate_names": names,
                "decision_pool": decision_pool,
                "gate_decision": decision,
                "failed_checks": results["gate_evaluation"]["failed_checks"],
                "outputs": [str(per_trial_path), str(csv_path), str(results_path)],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
