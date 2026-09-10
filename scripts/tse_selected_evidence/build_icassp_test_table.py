#!/usr/bin/env python3
"""Validate the one-shot frozen TEST replication and build its report."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tse_selected_evidence import build_icassp_table as core


ROOT = PROJECT_ROOT
ANALYSIS = ROOT / "analysis/selected_evidence_test"
RESULTS = ROOT / "results/selected_evidence_test"
SYSTEM_ROOT = RESULTS / "systems"
EXPECTED = 6000
COHORTS = OrderedDict([
    ("full_test", None),
    ("natural_primary_swap", "natural_primary_swap"),
    ("primary_correct_control", "primary_correct_control"),
    ("ambiguous_primary_wrong", "ambiguous_primary_wrong"),
])
SYSTEMS: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("Primary WeSep", {
        "path": SYSTEM_ROOT / "primary_wesep/per_trial_metrics.jsonl",
        "pool": "full", "kind": "deterministic",
    }),
    ("CDCS-2 direct", {
        "path": SYSTEM_ROOT / "cdcs2_direct/per_trial_metrics.jsonl",
        "pool": "full+tfmap_context_full", "kind": "deterministic",
    }),
    ("CDCS-5 direct", {
        "path": SYSTEM_ROOT / "cdcs5_direct/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "deterministic",
    }),
    ("CDCS-5 S3 reconstruction", {
        "path": SYSTEM_ROOT / "cdcs5_s3_recon/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "codec_control",
    }),
    ("Qwen-TSE UD (Primary evidence)", {
        "path": SYSTEM_ROOT / "qwen_tse_ud_primary/per_trial_metrics.jsonl",
        "pool": "full", "kind": "generative",
    }),
    ("Qwen-TSE fixed CSG (Primary evidence)", {
        "path": SYSTEM_ROOT / "qwen_tse_fixed_csg_primary/per_trial_metrics.jsonl",
        "pool": "full", "kind": "generative",
    }),
    ("Qwen-TSE UD (CDCS-2 evidence)", {
        "path": SYSTEM_ROOT / "qwen_tse_ud_cdcs2/per_trial_metrics.jsonl",
        "pool": "full+tfmap_context_full", "kind": "generative",
    }),
    ("Qwen-TSE fixed CSG (CDCS-2 evidence)", {
        "path": SYSTEM_ROOT / "qwen_tse_fixed_csg_cdcs2/per_trial_metrics.jsonl",
        "pool": "full+tfmap_context_full", "kind": "generative",
    }),
    ("Qwen-TSE UD (CDCS-5 evidence)", {
        "path": SYSTEM_ROOT / "qwen_tse_ud_cdcs5/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "generative",
    }),
    ("Qwen-TSE fixed CSG (CDCS-5 evidence)", {
        "path": SYSTEM_ROOT / "qwen_tse_fixed_csg_cdcs5/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "generative",
    }),
])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def validate_system(
    name: str, path: Path, ids: list[str], cohorts: dict[str, str],
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    row_ids = [row.get("trial_id") for row in rows]
    failed = sum(row.get("decode_status") != "ok" for row in rows)
    if not (
        len(rows) == EXPECTED and len(set(row_ids)) == EXPECTED
        and set(row_ids) == set(ids) and failed == 0
    ):
        raise ValueError(
            f"{name} incomplete rows={len(rows)} unique={len(set(row_ids))} failed={failed}"
        )
    required = (
        "target_WER", "content_switch", "acoustic_speaker_switch",
        "speaker_margin", "dnsmos_p808",
    )
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("split") != "test" or row.get("test_used") is not True:
            raise ValueError(f"{name} split/provenance mismatch: {row.get('trial_id')}")
        if any(row.get(key) is None for key in required):
            raise ValueError(f"{name} missing metric: {row.get('trial_id')}")
        copied = dict(row)
        copied["cohort"] = cohorts[row["trial_id"]]
        output_text = str(copied.get("output_text") or "").strip()
        target_text = str(copied.get("target_text") or "").strip()
        threshold = max(3, math.ceil(0.25 * len(target_text.split())))
        copied["unrelated_short_output"] = bool(
            copied.get("unrelated_short_output", len(output_text.split()) < threshold)
        )
        copied["empty_output"] = not output_text
        output[row["trial_id"]] = copied
    return output


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def percent(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{100 * value:.{digits}f}%"


def number(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> int:
    protocol = json.loads((ANALYSIS / "preregistered_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_BEFORE_SELECTED_EVIDENCE_TEST_INFERENCE_OR_RESULT_INSPECTION":
        raise ValueError("TEST protocol is not the immutable pre-result registration")
    selections = read_jsonl(ANALYSIS / "full_test_candidates.jsonl")
    candidate_rows = read_jsonl(ANALYSIS / "full_test_candidate_metrics.jsonl")
    ids = [row["trial_id"] for row in selections]
    if len(ids) != EXPECTED or len(set(ids)) != EXPECTED:
        raise ValueError("candidate selection is not exactly 6,000 unique TEST trials")
    cohorts = {row["trial_id"]: row["cohort"] for row in selections}
    counts = Counter(cohorts.values())
    expected_counts = Counter({
        "natural_primary_swap": 398,
        "primary_correct_control": 5588,
        "ambiguous_primary_wrong": 14,
    })
    if counts != expected_counts:
        raise ValueError(f"TEST cohort counts changed: {counts}")
    selection_by_id = {row["trial_id"]: row for row in selections}
    candidate_by_id = {row["trial_id"]: row for row in candidate_rows}
    if len(candidate_by_id) != EXPECTED or set(candidate_by_id) != set(ids):
        raise ValueError("candidate diagnostic coverage mismatch")

    systems = {
        name: validate_system(name, metadata["path"], ids, cohorts)
        for name, metadata in SYSTEMS.items()
    }
    core.enrich_deterministic(systems, selection_by_id, candidate_by_id)
    primary = systems["Primary WeSep"]
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    table_rows: list[dict[str, Any]] = []
    # Keep the frozen DEV decision implementation's internal key name while
    # exposing full_test in every persisted artifact.
    frozen_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for name, metadata in SYSTEMS.items():
        summaries[name] = {}
        frozen_summaries[name] = {}
        for cohort_name, cohort_value in COHORTS.items():
            rows = [
                systems[name][trial_id] for trial_id in ids
                if cohort_value is None or cohorts[trial_id] == cohort_value
            ]
            summary = core.summarize(rows, primary)
            summaries[name][cohort_name] = summary
            frozen_summaries[name]["full_dev" if cohort_name == "full_test" else cohort_name] = summary
            table_rows.append({
                "cohort": cohort_name, "system": name,
                "candidate_pool": metadata["pool"], "system_kind": metadata["kind"],
                "expected": len(rows), "unique": len(rows), "missing": 0,
                "duplicate": 0, "failed": 0, **summary,
            })

    # Reuse exactly the frozen comparison code and frozen decision rules.
    paired = core.paired_comparisons(systems, ids)
    for row in paired:
        if row["cohort"] == "full_dev":
            row["cohort"] = "full_test"
    frozen_paired = [dict(row, cohort="full_dev" if row["cohort"] == "full_test" else row["cohort"]) for row in paired]
    qwen, csg = core.decide_values(frozen_summaries, frozen_paired)
    recommended = core.choose_system(frozen_summaries, qwen, csg)

    cdcs5 = summaries["CDCS-5 direct"]
    cdcs2 = summaries["CDCS-2 direct"]
    primary_full = summaries["Primary WeSep"]["full_test"]
    d_full = cdcs5["full_test"]
    swap = cdcs5["natural_primary_swap"]
    control = cdcs5["primary_correct_control"]
    rng = np.random.default_rng(1986 + 991)
    replication_wer = core.continuous_test(
        np.asarray([primary[trial_id]["target_WER"] for trial_id in ids], dtype=float),
        np.asarray([systems["CDCS-5 direct"][trial_id]["target_WER"] for trial_id in ids], dtype=float),
        rng,
    )
    criteria = {
        "cdcs5_wer_lower_ci_excludes_zero": (
            replication_wer["absolute_difference_new_minus_base"] < 0
            and replication_wer["ci95_high"] < 0
        ),
        "cdcs5_content_switch_no_higher": d_full["content_switch_rate"] <= primary_full["content_switch_rate"],
        "cdcs5_acoustic_switch_no_higher": d_full["acoustic_switch_rate"] <= primary_full["acoustic_switch_rate"],
        "cdcs5_swap_target_correct_at_least_75pct": swap["candidate_target_correct_rate"] >= 0.75,
        "cdcs5_control_wrong_speaker_regression_at_most_0p5pct": (
            1.0 - control["candidate_target_correct_rate"] <= 0.005
        ),
        "all_ten_systems_exact_6000_zero_failure": True,
    }
    replication = "PASS" if all(criteria.values()) else "FAIL"

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "MAIN_ICASSP_TEST_TABLE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    paired_fields = list(dict.fromkeys(key for row in paired for key in row))
    with (RESULTS / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired)
    atomic_json(RESULTS / "selected_evidence_test_summary.json", summaries)
    atomic_json(RESULTS / "paired_comparisons.json", paired)
    atomic_json(RESULTS / "confirmatory_replication.json", {
        "status": replication, "criteria": criteria,
        "primary_vs_cdcs5_target_wer": replication_wer,
        "cdcs5_swap_target_correct_rate": swap["candidate_target_correct_rate"],
        "cdcs5_control_wrong_speaker_regression_rate": 1.0 - control["candidate_target_correct_rate"],
        "qwen_additional_value": qwen, "csg_additional_value": csg,
        "frozen_recommended_system_on_test_metrics": recommended,
        "test_used": True, "post_test_tuning_authorized": False,
    })

    full_rows: list[list[str]] = []
    swap_rows: list[list[str]] = []
    control_rows: list[list[str]] = []
    for name, metadata in SYSTEMS.items():
        full = summaries[name]["full_test"]
        sw = summaries[name]["natural_primary_swap"]
        ctl = summaries[name]["primary_correct_control"]
        full_rows.append([
            name, metadata["pool"], percent(full["target_wer"]),
            percent(full["content_switch_rate"]), percent(full["acoustic_switch_rate"]),
            number(full["speaker_margin"]), number(full["dnsmos_p808"]),
            percent(full["short_output_rate"]),
        ])
        swap_rows.append([
            name, percent(sw["target_wer"]), percent(sw["content_switch_rate"]),
            percent(sw["acoustic_switch_rate"]), number(sw["speaker_margin"]),
            percent(sw["speaker_recovery_rate"]), percent(sw["content_recovery_rate"]),
            percent(sw["joint_recovery_rate"]),
        ])
        control_rows.append([
            name, percent(ctl["target_wer"]), percent(ctl["content_switch_rate"]),
            percent(ctl["acoustic_switch_rate"]), number(ctl["speaker_margin"]),
            percent(ctl["content_regression_rate"]), percent(ctl["speaker_regression_rate"]),
            percent(ctl["joint_regression_rate"]),
        ])
    criterion_rows = [[key, "PASS" if value else "FAIL"] for key, value in criteria.items()]
    comparison_rows = []
    for label in core.COMPARISONS:
        for metric in ("target_wer", "speaker_margin", "dnsmos_p808"):
            row = next(value for value in paired if value["comparison"] == label and value["cohort"] == "full_test" and value["metric"] == metric)
            comparison_rows.append([
                label, metric, number(row["base_mean"], 4), number(row["new_mean"], 4),
                number(row["absolute_difference_new_minus_base"], 4),
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]", f"{row['p_value']:.4g}",
            ])

    report = "\n".join([
        "# ICASSP 2027 Selected-Evidence Frozen TEST Report",
        "",
        f"**CONFIRMATORY_REPLICATION = {replication}**",
        "",
        "This is the single pre-registered frozen TEST evaluation. DEV remains the model-selection split; no TEST-dependent tuning or second TEST run is authorized.",
        "",
        f"- `QWEN_ADDITIONAL_VALUE_ON_FROZEN_TEST_RULE = {qwen}`",
        f"- `CSG_ADDITIONAL_VALUE_ON_FROZEN_TEST_RULE = {csg}`",
        f"- `FROZEN_RULE_RECOMMENDED_SYSTEM_ON_TEST_METRICS = {recommended}`",
        "- `TEST_USED = YES`",
        "",
        "## Confirmatory criteria",
        "",
        markdown_table(["Pre-registered criterion", "Result"], criterion_rows),
        "",
        f"Primary-to-CDCS-5 WER difference is {replication_wer['absolute_difference_new_minus_base']:.4f} with paired 95% CI [{replication_wer['ci95_low']:.4f}, {replication_wer['ci95_high']:.4f}]. CDCS-5 recovers {percent(swap['candidate_target_correct_rate'])} of 398 frozen primary swaps; its candidate wrong-speaker regression on 5,588 controls is {percent(1.0 - control['candidate_target_correct_rate'], 4)}.",
        "",
        "## Full natural TEST (6,000 trials)",
        "",
        markdown_table(
            ["System", "Candidate pool", "Target WER ↓", "Content switch ↓", "Acoustic switch ↓", "Speaker margin ↑", "DNSMOS ↑", "Short output ↓"],
            full_rows,
        ),
        "",
        "## Frozen primary-swap cohort (398 trials)",
        "",
        markdown_table(
            ["System", "Target WER ↓", "Content switch ↓", "Acoustic switch ↓", "Speaker margin ↑", "Speaker recovery ↑", "Content recovery ↑", "Joint recovery ↑"],
            swap_rows,
        ),
        "",
        "## Frozen primary-correct controls (5,588 trials)",
        "",
        markdown_table(
            ["System", "Target WER ↓", "Content switch ↓", "Acoustic switch ↓", "Speaker margin ↑", "Content regression ↓", "Speaker regression ↓", "Joint regression ↓"],
            control_rows,
        ),
        "",
        "## Frozen four paired comparisons",
        "",
        "Differences are new minus base. The same 10,000-resample paired bootstrap/sign-flip definitions used on DEV are retained; all binary endpoints are available in `results/selected_evidence_test/paired_comparisons.csv`.",
        "",
        markdown_table(["Comparison", "Metric", "Base", "New", "Δ", "95% CI", "p"], comparison_rows),
        "",
        "## Candidate availability and safety",
        "",
        f"CDCS-2 target-correct candidate availability on primary swaps is {percent(cdcs2['natural_primary_swap']['candidate_target_correct_rate'])}; CDCS-5 is {percent(swap['candidate_target_correct_rate'])}. Selection uses only the complete target enrollment and frozen candidate audio. Clean targets, interferers, transcripts, SI-SDR, WER, and cohort labels are evaluation-only and never enter candidate construction or selection.",
        "",
        "## Interpretation",
        "",
        f"The frozen DEV mainline is **{'confirmed on TEST' if replication == 'PASS' else 'not fully confirmed on TEST'}** under the pre-registered conjunction. This result must be reported as-is. It does not reopen candidate design, selector training, Qwen-TSE checkpoint selection, or CSG tuning.",
        "",
        "## Reproducibility",
        "",
        "- Protocol: `analysis/selected_evidence_test/preregistered_protocol.json`",
        "- Per-system outputs: `results/selected_evidence_test/systems/`",
        "- Main table: `results/selected_evidence_test/MAIN_ICASSP_TEST_TABLE.csv`",
        "- Paired statistics: `results/selected_evidence_test/paired_comparisons.csv`",
        "- Frozen checkpoint: epoch 4 / step 13,900 content stored as `checkpoints/best.pt` (SHA-256 locked by the decoder).",
        "",
        "`POST_TEST_TUNING = PROHIBITED`",
    ]) + "\n"
    atomic_text(ROOT / "docs/ICASSP2027_SELECTED_EVIDENCE_TEST_REPORT.md", report)
    terminal = {
        "FULL_TEST_COMPLETE": "YES", "CONFIRMATORY_REPLICATION": replication,
        "CDCS5_SWAP_TARGET_CORRECT": swap["candidate_target_correct_rate"],
        "CDCS5_CONTROL_WRONG_SPEAKER_REGRESSION": 1.0 - control["candidate_target_correct_rate"],
        "PRIMARY_TARGET_WER": primary_full["target_wer"],
        "CDCS5_DIRECT_TARGET_WER": d_full["target_wer"],
        "QWEN_ADDITIONAL_VALUE": qwen, "CSG_ADDITIONAL_VALUE": csg,
        "RECOMMENDED_SYSTEM_BY_FROZEN_RULE": recommended,
        "TEST_USED": "YES", "POST_TEST_TUNING": "NO",
        "REPORT": "docs/ICASSP2027_SELECTED_EVIDENCE_TEST_REPORT.md",
    }
    atomic_json(RESULTS / "terminal_summary.json", terminal)
    print(json.dumps(terminal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
