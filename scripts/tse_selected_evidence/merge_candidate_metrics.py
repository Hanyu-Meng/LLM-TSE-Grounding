#!/usr/bin/env python3
"""Complete the 6,000-trial candidate metric audit and pool quality ablation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/selected_evidence"
GATE = ROOT / "analysis/candidate_gate"
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
POOLS = {
    "Pool A": ("full",),
    "Pool B": ("full", "tfmap_context_full"),
    "Pool C": ("full", "first", "middle", "final"),
    "Pool D": ORDER,
}
POOL_KEYS = {"Pool A": "pool_a", "Pool B": "pool_b", "Pool C": "pool_c", "Pool D": "pool_d"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def finite_mean(values: list[Any]) -> float | None:
    result = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(result)) if result else None


def merge_ambiguous() -> list[dict[str, Any]]:
    acoustic = read_jsonl(ANALYSIS / "ambiguous_acoustic_metrics.jsonl")
    speaker = {
        (row["trial_id"], row["candidate"]): row
        for row in read_jsonl(ANALYSIS / "ambiguous_speaker_embeddings/candidate_speaker_metrics.jsonl")
    }
    asr = {
        (row["trial_id"], row["candidate"]): row
        for row in read_jsonl(ANALYSIS / "ambiguous_asr_metrics.jsonl")
    }
    if len(acoustic) != 9 or len(speaker) != 45 or len(asr) != 45:
        raise ValueError("incomplete ambiguous candidate metrics")
    output = []
    for row in acoustic:
        candidates = {}
        for name in ORDER:
            key = (row["trial_id"], name)
            base = dict(row["candidates"][name])
            spk = speaker[key]
            text = asr[key]
            base.update({
                "candidate_embedding_path": spk["candidate_embedding_path"],
                "speaker_similarity_to_enrollment": spk["speaker_similarity_to_enrollment"],
                "speaker_similarity_to_interferer_enrollment": spk[
                    "speaker_similarity_to_interferer_enrollment"
                ],
                "speaker_embedding_margin": spk["speaker_embedding_margin"],
                "target_WER": text["target_WER"],
                "interferer_WER": text["interferer_WER"],
                "content_switch": text["content_switch"],
                "target_text": text["target_text"],
                "interferer_text": text["interferer_text"],
                "output_text": text["output_text"],
            })
            candidates[name] = base
        output.append({key: row[key] for key in (
            "trial_id", "split", "cohort", "target_speaker", "interferer_speaker",
            "target_wav", "interferer_wav", "target_enrollment_embedding_path",
            "interferer_enrollment_embedding_path",
        )} | {"candidates": candidates})
    return output


def main() -> int:
    selection_rows = read_jsonl(ANALYSIS / "full_dev_candidates.jsonl")
    selection = {row["trial_id"]: row for row in selection_rows}
    old = read_jsonl(GATE / "per_trial_candidates.jsonl")
    ambiguous = merge_ambiguous()
    by_id = {row["trial_id"]: row for row in old + ambiguous}
    if len(old) != 5991 or len(ambiguous) != 9 or len(by_id) != 6000 or set(by_id) != set(selection):
        raise ValueError("candidate metric completion did not yield exactly 6,000 aligned trials")
    full = []
    for row in selection_rows:
        metric = by_id[row["trial_id"]]
        if tuple(metric["candidates"]) != ORDER:
            raise ValueError(f"candidate order mismatch: {row['trial_id']}")
        for name in ORDER:
            observed = float(metric["candidates"][name]["speaker_similarity_to_enrollment"])
            chosen_score = float(row["candidates"][name]["enrollment_cosine"])
            # The two frozen CUDA ECAPA passes can differ at the 1e-4 level;
            # fail if this exceeds 5e-4. Candidate names are checked below by
            # the already-frozen selection artifact, so no post-reference
            # reselection is performed.
            if abs(observed - chosen_score) > 5e-4:
                raise ValueError(f"selection cosine mismatch: {row['trial_id']} {name}")
        full.append(metric | {"selected": row["selected"]})
    write_jsonl(ANALYSIS / "full_dev_candidate_metrics.jsonl", full)

    csv_rows = []
    cohort_filters = {
        "full_dev": lambda _row: True,
        "natural_primary_swap": lambda row: row["cohort"] == "natural_primary_swap",
        "primary_correct_control": lambda row: row["cohort"] == "primary_correct_control",
        "ambiguous_primary_wrong": lambda row: row["cohort"] == "ambiguous_primary_wrong",
    }
    for pool_name, names in POOLS.items():
        key = POOL_KEYS[pool_name]
        for cohort_name, predicate in cohort_filters.items():
            rows = [row for row in full if predicate(row)]
            choices = [row["selected"][key] for row in rows]
            values = [row["candidates"][name] for row, name in zip(rows, choices)]
            correct = [bool(value["target_correct"]) for value in values]
            csv_rows.append({
                "candidate_pool": pool_name,
                "pool_candidates": "+".join(names),
                "cohort": cohort_name,
                "trials": len(rows),
                "selected_candidate_counts": json.dumps(dict(Counter(choices)), sort_keys=True),
                "target_correct_count": sum(correct),
                "target_correct_rate": float(np.mean(correct)),
                "wrong_speaker_rate": float(1.0 - np.mean(correct)),
                "target_WER": finite_mean([value["target_WER"] for value in values]),
                "content_switch_rate": float(np.mean([value["content_switch"] for value in values])),
                "speaker_embedding_margin": finite_mean(
                    [value["speaker_embedding_margin"] for value in values]
                ),
                "sisdr_target_db": finite_mean([value["sisdr_target_db"] for value in values]),
                "sisdr_margin_db": finite_mean([value["sisdr_margin_db"] for value in values]),
                "primary_correct_regression_rate": (
                    float(1.0 - np.mean(correct))
                    if cohort_name == "primary_correct_control" else None
                ),
            })
    output = ROOT / "results/selected_evidence/candidate_pool_quality.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "status": "COMPLETE",
        "trials": len(full),
        "cohorts": dict(Counter(row["cohort"] for row in full)),
        "candidate_metric_rows": sum(len(row["candidates"]) for row in full),
        "candidate_order": ORDER,
        "selection_was_frozen_before_clean_reference_join": True,
        "test_used": False,
        "pool_quality_csv": str(output),
    }
    (ANALYSIS / "full_dev_candidate_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
