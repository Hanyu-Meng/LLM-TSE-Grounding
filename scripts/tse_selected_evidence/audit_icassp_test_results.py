#!/usr/bin/env python3
"""Independent integrity and headline-metric audit for frozen TEST outputs.

This intentionally does not import the report builder or its DEV helper module.
It checks the persisted trial-level artifacts and writes a compact QA record.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/selected_evidence_test"
RESULTS = ROOT / "results/selected_evidence_test"
EXPECTED = 6000
SYSTEMS = OrderedDict([
    ("Primary WeSep", "primary_wesep"),
    ("Pool B Selected Candidate", "pool_b_selected"),
    ("Pool D Selected Candidate", "pool_d_selected"),
    ("Pool D Selected S3 Recon", "pool_d_s3_recon"),
    ("Original Q-Full UD", "original_qfull_ud"),
    ("Original Q-Full + CSG", "original_qfull_csg"),
    ("Pool B Selected -> Q-Full UD", "pool_b_qfull_ud"),
    ("Pool B Selected -> Q-Full + CSG", "pool_b_qfull_csg"),
    ("Pool D Selected -> Q-Full UD", "pool_d_qfull_ud"),
    ("Pool D Selected -> Q-Full + CSG", "pool_d_qfull_csg"),
])
REQUIRED_FINITE = (
    "target_WER", "speaker_margin", "dnsmos_p808", "sim_target",
    "sim_interferer", "si_sdr_db", "si_sdri_db", "stoi",
)
EXPECTED_COHORTS = Counter({
    "natural_primary_swap": 398,
    "primary_correct_control": 5588,
    "ambiguous_primary_wrong": 14,
})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260819)
    estimates: list[np.ndarray] = []
    remaining = 20_000
    while remaining:
        batch = min(200, remaining)
        indexes = rng.integers(0, differences.size, size=(batch, differences.size))
        estimates.append(differences[indexes].mean(axis=1))
        remaining -= batch
    values = np.concatenate(estimates)
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def main() -> int:
    selections = read_jsonl(ANALYSIS / "full_test_candidates.jsonl")
    candidates = read_jsonl(ANALYSIS / "full_test_candidate_metrics.jsonl")
    ids = [row["trial_id"] for row in selections]
    selection_by_id = {row["trial_id"]: row for row in selections}
    candidate_by_id = {row["trial_id"]: row for row in candidates}
    canonical_ids = set(ids)
    cohort_by_id = {row["trial_id"]: row["cohort"] for row in selections}

    checks: dict[str, bool] = {
        "candidate_selection_6000_unique": len(ids) == EXPECTED and len(canonical_ids) == EXPECTED,
        "candidate_metric_6000_unique": len(candidates) == EXPECTED and len(candidate_by_id) == EXPECTED,
        "candidate_metric_exact_id_match": set(candidate_by_id) == canonical_ids,
        "cohort_counts_exact": Counter(cohort_by_id.values()) == EXPECTED_COHORTS,
    }
    per_system: dict[str, Any] = {}
    rows_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for name, directory in SYSTEMS.items():
        path = RESULTS / "systems" / directory / "per_trial_metrics.jsonl"
        rows = read_jsonl(path)
        row_ids = [row.get("trial_id") for row in rows]
        by_id = {row["trial_id"]: row for row in rows}
        rows_by_name[name] = by_id
        missing_required = 0
        nonfinite_required = 0
        invalid_boolean = 0
        provenance_errors = 0
        missing_audio = 0
        target_identity_mismatches = 0
        for row in rows:
            if any(row.get(key) is None for key in REQUIRED_FINITE):
                missing_required += 1
            for key in REQUIRED_FINITE:
                value = row.get(key)
                if value is not None and not math.isfinite(float(value)):
                    nonfinite_required += 1
            if not isinstance(row.get("content_switch"), bool) or not isinstance(
                row.get("acoustic_speaker_switch"), bool
            ):
                invalid_boolean += 1
            if row.get("split") != "test" or row.get("test_used") is not True:
                provenance_errors += 1
            if not Path(str(row.get("output_wav", ""))).is_file():
                missing_audio += 1
            canonical = rows_by_name["Primary WeSep"].get(row["trial_id"])
            if canonical is not None and any(
                row.get(key) != canonical.get(key)
                for key in ("target_spk", "interferer_spk", "target_text", "interferer_text")
            ):
                target_identity_mismatches += 1
        status = {
            "rows": len(rows),
            "unique_trial_ids": len(set(row_ids)),
            "duplicate_rows": len(rows) - len(set(row_ids)),
            "missing_trial_ids": len(canonical_ids - set(row_ids)),
            "extra_trial_ids": len(set(row_ids) - canonical_ids),
            "decode_failures": sum(row.get("decode_status") != "ok" for row in rows),
            "missing_required_metrics": missing_required,
            "nonfinite_required_metrics": nonfinite_required,
            "invalid_switch_booleans": invalid_boolean,
            "provenance_errors": provenance_errors,
            "missing_output_audio": missing_audio,
            "target_identity_mismatches": target_identity_mismatches,
        }
        status["pass"] = all(value == 0 for key, value in status.items() if key not in {
            "rows", "unique_trial_ids", "pass"
        }) and status["rows"] == EXPECTED and status["unique_trial_ids"] == EXPECTED
        per_system[name] = status

    checks["all_ten_systems_pass_integrity"] = len(per_system) == 10 and all(
        value["pass"] for value in per_system.values()
    )
    checks["all_systems_share_exact_trial_ids"] = all(
        set(rows) == canonical_ids for rows in rows_by_name.values()
    )

    table_rows: dict[tuple[str, str], dict[str, str]] = {}
    with (RESULTS / "MAIN_ICASSP_TEST_TABLE.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table_rows[(row["cohort"], row["system"])] = row
    summary_mismatches: list[dict[str, Any]] = []
    cohort_values = OrderedDict([
        ("full_test", None),
        ("natural_primary_swap", "natural_primary_swap"),
        ("primary_correct_control", "primary_correct_control"),
        ("ambiguous_primary_wrong", "ambiguous_primary_wrong"),
    ])
    metric_mapping = {
        "target_wer": "target_WER",
        "content_switch_rate": "content_switch",
        "acoustic_switch_rate": "acoustic_speaker_switch",
        "speaker_margin": "speaker_margin",
        "dnsmos_p808": "dnsmos_p808",
    }
    for system_name, by_id in rows_by_name.items():
        for cohort_name, cohort_value in cohort_values.items():
            selected_ids = [
                trial_id for trial_id in ids
                if cohort_value is None or cohort_by_id[trial_id] == cohort_value
            ]
            reported = table_rows[(cohort_name, system_name)]
            for report_key, trial_key in metric_mapping.items():
                computed = float(np.mean([float(by_id[trial_id][trial_key]) for trial_id in selected_ids]))
                expected = float(reported[report_key])
                if not math.isclose(computed, expected, rel_tol=0.0, abs_tol=1e-12):
                    summary_mismatches.append({
                        "cohort": cohort_name, "system": system_name,
                        "metric": report_key, "computed": computed, "reported": expected,
                    })
    checks["all_200_core_summary_cells_recompute"] = not summary_mismatches

    selector_names = {
        "Primary WeSep": "pool_full",
        "Pool B Selected Candidate": "pool_b",
        "Pool D Selected Candidate": "pool_d",
    }
    candidate_rates: dict[str, dict[str, float]] = {}
    selector_mismatches = 0
    for system_name, selector_key in selector_names.items():
        candidate_rates[system_name] = {}
        for cohort_name, cohort_value in cohort_values.items():
            selected_ids = [
                trial_id for trial_id in ids
                if cohort_value is None or cohort_by_id[trial_id] == cohort_value
            ]
            correct: list[bool] = []
            for trial_id in selected_ids:
                candidate_name = selection_by_id[trial_id]["selected"][selector_key]
                if rows_by_name[system_name][trial_id].get("selected_candidate") != candidate_name:
                    selector_mismatches += 1
                correct.append(bool(candidate_by_id[trial_id]["candidates"][candidate_name]["target_correct"]))
            candidate_rates[system_name][cohort_name] = float(np.mean(correct))
    checks["persisted_candidate_choices_match_frozen_selector"] = selector_mismatches == 0

    primary = rows_by_name["Primary WeSep"]
    pool_d = rows_by_name["Pool D Selected Candidate"]
    primary_wer = float(np.mean([float(primary[trial_id]["target_WER"]) for trial_id in ids]))
    pool_d_wer = float(np.mean([float(pool_d[trial_id]["target_WER"]) for trial_id in ids]))
    differences = np.asarray([
        float(pool_d[trial_id]["target_WER"]) - float(primary[trial_id]["target_WER"])
        for trial_id in ids
    ])
    ci_low, ci_high = bootstrap_ci(differences)
    confirmation = json.loads((RESULTS / "confirmatory_replication.json").read_text(encoding="utf-8"))
    swap_rate = candidate_rates["Pool D Selected Candidate"]["natural_primary_swap"]
    control_regression = 1.0 - candidate_rates["Pool D Selected Candidate"]["primary_correct_control"]
    checks.update({
        "primary_wer_matches_confirmation": math.isclose(
            primary_wer, confirmation["primary_vs_pool_d_target_wer"]["base_mean"], abs_tol=1e-12
        ),
        "pool_d_wer_matches_confirmation": math.isclose(
            pool_d_wer, confirmation["primary_vs_pool_d_target_wer"]["new_mean"], abs_tol=1e-12
        ),
        "swap_rate_matches_confirmation": math.isclose(
            swap_rate, confirmation["pool_d_swap_target_correct_rate"], abs_tol=1e-12
        ),
        "control_regression_matches_confirmation": math.isclose(
            control_regression, confirmation["pool_d_control_wrong_speaker_regression_rate"], abs_tol=1e-12
        ),
        "independent_bootstrap_ci_excludes_zero": ci_high < 0.0,
    })

    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "frozen natural TEST; 10 systems x 6,000 trials",
        "checks": checks,
        "cohort_counts": dict(Counter(cohort_by_id.values())),
        "per_system": per_system,
        "summary_cell_mismatches": summary_mismatches,
        "selector_mismatches": selector_mismatches,
        "candidate_target_correct_rates": candidate_rates,
        "independent_headline_recomputation": {
            "primary_target_wer": primary_wer,
            "pool_d_target_wer": pool_d_wer,
            "absolute_difference_new_minus_base": pool_d_wer - primary_wer,
            "independent_bootstrap_resamples": 20_000,
            "independent_bootstrap_seed": 20260819,
            "independent_ci95_low": ci_low,
            "independent_ci95_high": ci_high,
            "pool_d_swap_target_correct_rate": swap_rate,
            "pool_d_control_wrong_speaker_regression_rate": control_regression,
        },
    }
    atomic_json(RESULTS / "data_quality_audit.json", output)
    print(json.dumps(output, indent=2))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
