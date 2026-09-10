#!/usr/bin/env python3
"""Merge noisy TSE metrics and compile split/cohort paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = {
    "D0": ("primary_wesep", "Primary WeSep"),
    "D3": ("pool_d_selected", "Pool D direct"),
    "G2": ("pool_d_qfull_ud", "Pool D Q-Full UD"),
    "G3": ("pool_d_fixed_csg", "Pool D fixed CSG"),
    "G5": ("pool_d_adaptive_csg", "Pool D adaptive CSG"),
}
FULL_CODES = ("D0", "D3", "G2", "G3", "G5", "G6")
EXTENDED_METRICS = {
    "duration_estoi": ("estoi", "reference_duration_seconds", "output_duration_seconds",
                       "common_duration_seconds", "output_to_reference_length_ratio",
                       "duration_mismatch_gt_5pct"),
    "utmos": ("utmos",),
    "speechbertscore": ("speechbertscore",),
    "lps": ("lps",),
}
MEAN_KEYS = (
    "target_WER_raw", "target_WER_capped", "interferer_WER", "interferer_leakage",
    "sim_target", "sim_interferer", "speaker_margin", "si_sdr_db", "si_sdri_db",
    "stoi", "estoi", "pesq_wb", "dnsmos_p808", "dnsmos_sig", "dnsmos_bak",
    "dnsmos_ovl", "utmos", "speechbertscore", "lps",
    "output_to_reference_length_ratio", "selected_lambda", "modified_token_rate",
    "gnr_edit_rate", "gnr_mean_hamming_distance",
)
BINARY_KEYS = (
    "content_switch", "target_content_preference", "valid_target_content",
    "empty_output", "short_output", "high_error", "acoustic_speaker_switch",
    "duration_mismatch_gt_5pct", "speaker_binding_correct",
    "joint_speaker_content_recovery",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--expected", type=int, default=8400)
    parser.add_argument("--allow-missing-extended", action="store_true")
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


def load_keyed(path: Path, expected: int | None = None) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {row["trial_id"]: row for row in rows}
    if len(rows) != len(result) or (expected is not None and len(rows) != expected):
        raise ValueError(f"coverage/duplicate failure: {path}")
    return result


def finite(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def summarize(rows: list[dict[str, Any]], tail_ids: dict[str, set[str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"trials": len(rows)}
    for key in MEAN_KEYS:
        values = [value for row in rows if (value := finite(row, key)) is not None]
        summary[key] = float(np.mean(values)) if values else None
        summary[f"{key}_coverage"] = len(values) / len(rows) if rows else None
    for key in BINARY_KEYS:
        values = [bool(row[key]) for row in rows if row.get(key) is not None]
        summary[f"{key}_rate"] = float(np.mean(values)) if values else None
        summary[f"{key}_coverage"] = len(values) / len(rows) if rows else None
    wers = [float(row["target_WER_raw"]) for row in rows]
    for percentile in (50, 90, 95, 99):
        summary[f"target_WER_raw_p{percentile}"] = (
            float(np.percentile(wers, percentile)) if wers else None
        )
    for name, ids in tail_ids.items():
        values = [float(row["target_WER_raw"]) for row in rows if row["trial_id"] in ids]
        summary[f"frozen_{name}_mean_raw_WER"] = (
            float(np.mean(values)) if values else None
        )
        summary[f"frozen_{name}_coverage"] = len(values) / len(ids)
    return summary


def system_candidate_correct(code: str, candidate: dict[str, Any], row: dict[str, Any]) -> bool:
    if code == "D0":
        return bool(candidate["full_target_correct"])
    if code == "D1":
        return bool(candidate["tfmap_context_full_target_correct"])
    if code == "D2":
        return bool(candidate["pool_b_selected_target_correct"])
    if code == "D3":
        return bool(candidate["pool_d_selected_target_correct"])
    return float(row["speaker_margin"]) > 0.0


def main() -> int:
    args = parse_args()
    split = args.split
    result_root = ROOT / f"results/noisy_wham/{split}/systems"
    analysis_root = ROOT / f"analysis/noisy_wham/{split}/full"
    evaluation = load_keyed(
        ROOT / f"manifests/noisy_wham/{split}_full_evaluation.jsonl", args.expected
    )
    candidate = load_keyed(
        analysis_root / "candidate_analysis/per_trial_candidate_analysis.jsonl",
        args.expected,
    )
    tail = json.loads((
        ROOT / f"analysis/noisy_wham/{split}/noisy_qfull_tail_bank.json"
    ).read_text())
    tail_ids = {
        name: set(value["trial_ids"])
        for name, value in tail["sets"].items()
    }
    gnr_selection = json.loads((
        ROOT / "analysis/noisy_wham/dev/full/gnr_selection.json"
    ).read_text())
    gnr_slug = (
        gnr_selection["selected_dev_slug"]
        if split == "dev" else "pool_d_adaptive_csg_gnr"
    )
    systems = SYSTEMS | {"G6": (gnr_slug, "Pool D adaptive CSG + GNR-LLM")}
    codes = FULL_CODES
    compiled: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    output_dir = ROOT / f"results/noisy_wham/{split}/compiled"
    for code in codes:
        slug, name = systems[code]
        expected = args.expected
        source = load_keyed(result_root / slug / "per_trial_metrics.jsonl", expected)
        extended: dict[str, dict[str, dict[str, Any]]] = {}
        for metric in EXTENDED_METRICS:
            path = analysis_root / f"extended/{metric}/{slug}/per_trial.jsonl"
            if path.is_file():
                extended[metric] = load_keyed(path, expected)
            elif not args.allow_missing_extended:
                raise FileNotFoundError(path)
        token_path = result_root / slug / "per_trial_tokens.jsonl"
        tokens = load_keyed(token_path, expected) if token_path.is_file() else {}
        adaptive_anchor = {}
        if code == "G6":
            adaptive_anchor = load_keyed(
                result_root / "pool_d_adaptive_csg/per_trial_tokens.jsonl", args.expected
            )
        rows = []
        for trial_id, base in source.items():
            metadata = evaluation[trial_id]
            cand = candidate[trial_id]
            raw_wer = float(base["target_WER"])
            interferer_wer = (
                float(base["interferer_WER"]) if base.get("interferer_WER") is not None else None
            )
            output_text = str(base.get("output_text") or "")
            row = dict(base)
            row.update({
                "system_code": code,
                "system_slug": slug,
                "system_name": name,
                "benchmark": metadata["benchmark"],
                "snr_db": metadata.get("snr_db"),
                "gender_cohort": metadata["gender_cohort"],
                "clean_primary_cohort": metadata["cohort"],
                "base_trial_id": metadata["base_trial_id"],
                "noisy_primary_swap": cand["noisy_primary_swap"],
                "noisy_primary_correct_control": cand["noisy_primary_correct_control"],
                "target_WER_raw": raw_wer,
                "target_WER_capped": min(raw_wer, 1.0),
                "empty_output": not bool(output_text),
                "short_output": bool(base["unrelated_short_output"]),
                "high_error": raw_wer > 0.5,
                "target_content_preference": (
                    interferer_wer is not None and raw_wer <= interferer_wer
                ),
                "valid_target_content": (
                    not bool(base["content_switch"])
                    and min(raw_wer, 1.0) < 0.5
                    and bool(output_text)
                    and not bool(base["unrelated_short_output"])
                ),
                "selected_lambda": None,
                "modified_token_rate": None,
                "gnr_edit_rate": None,
                "gnr_mean_hamming_distance": None,
            })
            for metric, keyed in extended.items():
                ext = keyed[trial_id]
                for key in EXTENDED_METRICS[metric]:
                    row[key] = ext[key]
            if trial_id in tokens:
                token = tokens[trial_id]
                row["modified_token_rate"] = token.get("token_flip_rate_vs_evidence")
                row["selected_lambda"] = token.get(
                    "selected_lambda", 1.0 if code in {"G1", "G3"} else 0.0 if code in {"G0", "G2"} else None
                )
                row["gnr_edit_rate"] = token.get("gnr_edit_rate")
                row["gnr_mean_hamming_distance"] = token.get("mean_hamming_edit_distance")
            if code == "G6":
                row["selected_lambda"] = adaptive_anchor[trial_id].get("selected_lambda")
            speaker_ok = system_candidate_correct(code, cand, row)
            row["speaker_binding_correct"] = speaker_ok
            row["joint_speaker_content_recovery"] = (
                speaker_ok and row["valid_target_content"]
            )
            rows.append(row)
        compiled[code] = rows
        natural = [row for row in rows if row["benchmark"] == "natural"]
        controlled = [row for row in rows if row["benchmark"] == "controlled"]
        summaries[code] = {
            "system_slug": slug,
            "system_name": name,
            "all": summarize(rows, tail_ids),
            "natural": summarize(natural, tail_ids),
            "controlled": summarize(controlled, tail_ids),
            "clean_defined_primary_swaps": summarize([
                row for row in natural if row["clean_primary_cohort"] == "clean_defined_primary_swap"
            ], tail_ids),
            "noisy_primary_swaps": summarize([
                row for row in natural if row["noisy_primary_swap"]
            ], tail_ids),
            "noisy_primary_correct_controls": summarize([
                row for row in natural if row["noisy_primary_correct_control"]
            ], tail_ids),
            "same_gender_natural": summarize([
                row for row in natural if row["gender_cohort"] == "same"
            ], tail_ids),
            "different_gender_natural": summarize([
                row for row in natural if row["gender_cohort"] == "different"
            ], tail_ids),
            "controlled_by_snr": {
                str(snr): summarize([
                    row for row in controlled if int(row["snr_db"]) == snr
                ], tail_ids)
                for snr in (-5, 0, 5, 10, 15)
            },
            "test_used": split == "test",
        }
        atomic_text(
            output_dir / f"{slug}.jsonl",
            "".join(json.dumps(row) + "\n" for row in rows),
        )
        atomic_json(output_dir / f"{slug}.summary.json", summaries[code])

    atomic_json(output_dir / "all_system_summaries.json", {
        "status": "COMPLETE", "split": split, "systems": summaries,
        "test_used": split == "test",
    })
    table_codes = FULL_CODES
    columns = (
        "split", "system_code", "system", "target_WER_capped", "target_WER_raw",
        "content_switch_rate", "acoustic_speaker_switch_rate", "speaker_margin",
        "lps", "speechbertscore", "dnsmos_p808", "dnsmos_sig", "dnsmos_bak",
        "dnsmos_ovl", "utmos", "target_WER_raw_p50", "target_WER_raw_p90",
        "target_WER_raw_p95", "target_WER_raw_p99", "high_error_rate",
        "empty_output_rate", "short_output_rate", "target_content_preference_rate",
        "sim_target", "sim_interferer", "joint_speaker_content_recovery_rate",
    )
    table_rows = []
    for code in table_codes:
        summary = summaries[code]["natural"]
        table_rows.append({
            "split": split.upper(), "system_code": code,
            "system": summaries[code]["system_name"],
            **{key: summary.get(key) for key in columns[3:]},
        })
    table_path = output_dir / "main_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = table_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table_rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(table_path)

    controlled_codes = ("D3", "G2", "G3", "G5", "G6")
    controlled_rows = []
    for snr in (-5, 0, 5, 10, 15):
        for code in controlled_codes:
            summary = summaries[code]["controlled_by_snr"][str(snr)]
            controlled_rows.append({
                "split": split.upper(), "snr_db": snr, "system_code": code,
                "system": summaries[code]["system_name"],
                "target_WER_capped": summary["target_WER_capped"],
                "target_WER_raw": summary["target_WER_raw"],
                "content_switch_rate": summary["content_switch_rate"],
                "acoustic_speaker_switch_rate": summary["acoustic_speaker_switch_rate"],
                "speaker_margin": summary["speaker_margin"],
                "dnsmos_p808": summary["dnsmos_p808"],
                "dnsmos_sig": summary["dnsmos_sig"],
                "dnsmos_bak": summary["dnsmos_bak"],
                "dnsmos_ovl": summary["dnsmos_ovl"],
                "utmos": summary["utmos"],
                "target_WER_raw_p95": summary["target_WER_raw_p95"],
                "high_error_rate": summary["high_error_rate"],
                "mean_selected_lambda": summary["selected_lambda"],
                "modified_token_rate": summary["modified_token_rate"],
                "gnr_edit_rate": summary["gnr_edit_rate"],
            })
    controlled_path = output_dir / "controlled_snr_table.csv"
    temporary = controlled_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(controlled_rows[0]))
        writer.writeheader(); writer.writerows(controlled_rows)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(controlled_path)
    print(json.dumps({
        "status": "COMPLETE", "split": split, "systems": len(summaries),
        "rows": {code: len(rows) for code, rows in compiled.items()},
        "test_used": split == "test",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
