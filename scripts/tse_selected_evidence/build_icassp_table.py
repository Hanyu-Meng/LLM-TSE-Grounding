#!/usr/bin/env python3
"""Validate selected-evidence results and build the frozen ICASSP tables."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/selected_evidence"
RESULTS = ROOT / "results/selected_evidence"
SYSTEM_ROOT = RESULTS / "systems"
EXPECTED = 6000
SEED = 1986
BOOTSTRAPS = 10000

SYSTEMS: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("Primary WeSep", {
        "path": ROOT / "dev_outputs/WeSep/per_trial_metrics.jsonl",
        "pool": "full", "kind": "deterministic",
    }),
    ("Pool B Selected Candidate", {
        "path": SYSTEM_ROOT / "pool_b_selected/per_trial_metrics.jsonl",
        "pool": "full+tfmap_context_full", "kind": "deterministic",
    }),
    ("Pool D Selected Candidate", {
        "path": SYSTEM_ROOT / "pool_d_selected/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "deterministic",
    }),
    ("Pool D Selected S3 Recon", {
        "path": SYSTEM_ROOT / "pool_d_s3_recon/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "codec_control",
    }),
    ("Original Q-Full UD", {
        "path": ROOT / "grounding/csg_lambda_0/per_trial_metrics.jsonl",
        "pool": "full", "kind": "generative",
    }),
    ("Original Q-Full + CSG", {
        "path": ROOT / "grounding/csg_lambda_1/per_trial_metrics.jsonl",
        "pool": "full", "kind": "generative",
    }),
    ("Pool B Selected -> Q-Full UD", {
        "path": SYSTEM_ROOT / "pool_b_qfull_ud/per_trial_metrics.jsonl",
        "pool": "full+tfmap_context_full", "kind": "generative",
    }),
    ("Pool B Selected -> Q-Full + CSG", {
        "path": SYSTEM_ROOT / "pool_b_qfull_csg/per_trial_metrics.jsonl",
        "pool": "full+tfmap_context_full", "kind": "generative",
    }),
    ("Pool D Selected -> Q-Full UD", {
        "path": SYSTEM_ROOT / "pool_d_qfull_ud/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "generative",
    }),
    ("Pool D Selected -> Q-Full + CSG", {
        "path": SYSTEM_ROOT / "pool_d_qfull_csg/per_trial_metrics.jsonl",
        "pool": "all five", "kind": "generative",
    }),
])

COHORTS = OrderedDict([
    ("full_dev", None),
    ("natural_primary_swap", "natural_primary_swap"),
    ("primary_correct_control", "primary_correct_control"),
    ("ambiguous_primary_wrong", "ambiguous_primary_wrong"),
])

COMPARISONS = OrderedDict([
    ("A", (
        "Pool D Selected Candidate", "Pool D Selected -> Q-Full UD",
        "Does Qwen generation add value beyond selected deterministic evidence?",
    )),
    ("B", (
        "Pool D Selected -> Q-Full UD", "Pool D Selected -> Q-Full + CSG",
        "Does CSG add value on selected evidence?",
    )),
    ("C", (
        "Original Q-Full + CSG", "Pool D Selected -> Q-Full + CSG",
        "Does candidate correction improve the final generative system?",
    )),
    ("D", (
        "Pool B Selected -> Q-Full + CSG", "Pool D Selected -> Q-Full + CSG",
        "Are the three segmented enrollment views worth their cost?",
    )),
])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2) + "\n")


def load_candidates() -> tuple[
    list[str], dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    selections = read_jsonl(ANALYSIS / "full_dev_candidates.jsonl")
    candidate_metrics = read_jsonl(ANALYSIS / "full_dev_candidate_metrics.jsonl")
    ids = [row["trial_id"] for row in selections]
    if len(ids) != EXPECTED or len(set(ids)) != EXPECTED:
        raise ValueError("frozen candidate manifest is not exactly 6,000 unique trials")
    cohort = {row["trial_id"]: row["cohort"] for row in selections}
    counts = {name: list(cohort.values()).count(name) for name in set(cohort.values())}
    if counts != {
        "natural_primary_swap": 405,
        "primary_correct_control": 5586,
        "ambiguous_primary_wrong": 9,
    }:
        raise ValueError(f"frozen cohort counts changed: {counts}")
    selection_by_id = {row["trial_id"]: row for row in selections}
    metric_by_id = {row["trial_id"]: row for row in candidate_metrics}
    if len(metric_by_id) != EXPECTED or set(metric_by_id) != set(ids):
        raise ValueError("candidate metric join coverage mismatch")
    return ids, cohort, selection_by_id, metric_by_id


def validate_system(
    name: str, path: Path, ids: list[str], cohort: dict[str, str],
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    row_ids = [row.get("trial_id") for row in rows]
    unique = len(set(row_ids))
    missing = len(set(ids) - set(row_ids))
    duplicate = len(row_ids) - unique
    failed = sum(row.get("decode_status") != "ok" for row in rows)
    if not (
        len(rows) == EXPECTED and unique == EXPECTED and missing == 0
        and duplicate == 0 and failed == 0 and set(row_ids) == set(ids)
    ):
        raise ValueError(
            f"{name} incomplete expected={EXPECTED} rows={len(rows)} unique={unique} "
            f"missing={missing} duplicate={duplicate} failed={failed}"
        )
    result: dict[str, dict[str, Any]] = {}
    required = (
        "target_WER", "content_switch", "acoustic_speaker_switch",
        "speaker_margin", "dnsmos_p808",
    )
    for row in rows:
        trial_id = row["trial_id"]
        if any(row.get(key) is None for key in required):
            raise ValueError(f"{name} missing required metrics: {trial_id}")
        copied = dict(row)
        copied["cohort"] = cohort[trial_id]
        output_text = str(copied.get("output_text") or "").strip()
        target_text = str(copied.get("target_text") or "").strip()
        threshold = max(3, math.ceil(0.25 * len(target_text.split())))
        copied["unrelated_short_output"] = bool(
            copied.get("unrelated_short_output", len(output_text.split()) < threshold)
        )
        copied["empty_output"] = not output_text
        result[trial_id] = copied
    return result


def enrich_deterministic(
    systems: dict[str, dict[str, dict[str, Any]]],
    selections: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> None:
    mapping = {
        "Primary WeSep": lambda trial_id: "full",
        "Pool B Selected Candidate": lambda trial_id: selections[trial_id]["selected"]["pool_b"],
        "Pool D Selected Candidate": lambda trial_id: selections[trial_id]["selected"]["pool_d"],
    }
    for system_name, chooser in mapping.items():
        for trial_id, row in systems[system_name].items():
            candidate_name = chooser(trial_id)
            source = candidates[trial_id]["candidates"][candidate_name]
            row.update({
                "selected_candidate": candidate_name,
                "candidate_target_correct": bool(source["target_correct"]),
                "candidate_sisdr_target_db": float(source["sisdr_target_db"]),
                "candidate_sisdr_interferer_db": float(source["sisdr_interferer_db"]),
                "candidate_sisdr_margin_db": float(source["sisdr_margin_db"]),
            })


def subset_rows(
    values: dict[str, dict[str, Any]], ids: list[str], cohort: str | None,
) -> list[dict[str, Any]]:
    return [
        values[trial_id] for trial_id in ids
        if cohort is None or values[trial_id]["cohort"] == cohort
    ]


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize(
    rows: list[dict[str, Any]], primary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    wers = np.asarray([float(row["target_WER"]) for row in rows])
    content = np.asarray([bool(row["content_switch"]) for row in rows])
    acoustic = np.asarray([bool(row["acoustic_speaker_switch"]) for row in rows])
    speaker_correct = np.asarray([
        not bool(row["acoustic_speaker_switch"]) and float(row["speaker_margin"]) > 0
        for row in rows
    ])
    content_correct = np.asarray([
        not bool(row["content_switch"]) and float(row["target_WER"]) < 0.5
        and not bool(row["empty_output"])
        and not bool(row["unrelated_short_output"])
        for row in rows
    ])
    joint_recovery = speaker_correct & content_correct
    base_rows = [primary[row["trial_id"]] for row in rows]
    content_regression = np.asarray([
        not bool(base["content_switch"]) and bool(row["content_switch"])
        for row, base in zip(rows, base_rows)
    ])
    speaker_regression = np.asarray([
        not bool(base["acoustic_speaker_switch"])
        and bool(row["acoustic_speaker_switch"])
        for row, base in zip(rows, base_rows)
    ])
    output: dict[str, Any] = {
        "trials": len(rows),
        "target_wer": float(np.mean(wers)),
        "content_switch_rate": float(np.mean(content)),
        "acoustic_switch_rate": float(np.mean(acoustic)),
        "speaker_margin": mean(rows, "speaker_margin"),
        "dnsmos_p808": mean(rows, "dnsmos_p808"),
        "interferer_leakage": mean(rows, "interferer_leakage"),
        "p95_wer": float(np.percentile(wers, 95)),
        "p99_wer": float(np.percentile(wers, 99)),
        "empty_output_rate": float(np.mean([row["empty_output"] for row in rows])),
        "short_output_rate": float(np.mean([
            row["unrelated_short_output"] for row in rows
        ])),
        "decode_failure_rate": 0.0,
        "speaker_recovery_rate": float(np.mean(speaker_correct)),
        "content_recovery_rate": float(np.mean(content_correct)),
        "joint_recovery_rate": float(np.mean(joint_recovery)),
        "content_regression_rate": float(np.mean(content_regression)),
        "speaker_regression_rate": float(np.mean(speaker_regression)),
        "joint_regression_rate": float(np.mean(content_regression | speaker_regression)),
    }
    if all("candidate_target_correct" in row for row in rows):
        output.update({
            "candidate_target_correct_rate": mean(rows, "candidate_target_correct"),
            "candidate_sisdr_target_db": mean(rows, "candidate_sisdr_target_db"),
            "candidate_sisdr_interferer_db": mean(rows, "candidate_sisdr_interferer_db"),
            "candidate_sisdr_margin_db": mean(rows, "candidate_sisdr_margin_db"),
        })
    return output


def bootstrap_ci(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = differences.size
    estimates: list[np.ndarray] = []
    remaining = BOOTSTRAPS
    while remaining:
        batch = min(200, remaining)
        indexes = rng.integers(0, n, size=(batch, n))
        estimates.append(differences[indexes].mean(axis=1))
        remaining -= batch
    values = np.concatenate(estimates)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def sign_flip_pvalue(differences: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(differences)))
    if not np.any(differences):
        return 1.0
    count = 0
    remaining = BOOTSTRAPS
    while remaining:
        batch = min(200, remaining)
        signs = rng.integers(0, 2, size=(batch, differences.size), dtype=np.int8)
        signs = signs * 2 - 1
        estimates = np.abs((signs * differences).mean(axis=1))
        count += int(np.sum(estimates >= observed - 1e-15))
        remaining -= batch
    return (count + 1) / (BOOTSTRAPS + 1)


def continuous_test(
    base: np.ndarray, new: np.ndarray, rng: np.random.Generator,
) -> dict[str, Any]:
    difference = new - base
    ci = bootstrap_ci(difference, rng)
    base_mean = float(np.mean(base))
    new_mean = float(np.mean(new))
    absolute = new_mean - base_mean
    return {
        "base_mean": base_mean,
        "new_mean": new_mean,
        "absolute_difference_new_minus_base": absolute,
        "relative_difference": absolute / abs(base_mean) if base_mean else None,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "p_value": sign_flip_pvalue(difference, rng),
        "test": "paired bootstrap CI + paired sign-flip randomization",
    }


def binary_test(
    base: np.ndarray, new: np.ndarray, rng: np.random.Generator,
) -> dict[str, Any]:
    base = base.astype(bool)
    new = new.astype(bool)
    base_only = int(np.sum(base & ~new))
    new_only = int(np.sum(~base & new))
    discordant = base_only + new_only
    p_value = float(binomtest(
        min(base_only, new_only), discordant, 0.5, alternative="two-sided"
    ).pvalue) if discordant else 1.0
    differences = new.astype(float) - base.astype(float)
    ci = bootstrap_ci(differences, rng)
    base_mean = float(np.mean(base))
    new_mean = float(np.mean(new))
    absolute = new_mean - base_mean
    return {
        "base_mean": base_mean,
        "new_mean": new_mean,
        "absolute_difference_new_minus_base": absolute,
        "relative_difference": absolute / abs(base_mean) if base_mean else None,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "p_value": p_value,
        "discordant_base_only": base_only,
        "discordant_new_only": new_only,
        "test": "exact McNemar + paired bootstrap CI",
    }


def derived_binary(row: dict[str, Any], metric: str, primary: dict[str, Any]) -> bool:
    if metric == "content_switch":
        return bool(row["content_switch"])
    if metric == "acoustic_switch":
        return bool(row["acoustic_speaker_switch"])
    speaker_correct = (
        not bool(row["acoustic_speaker_switch"]) and float(row["speaker_margin"]) > 0
    )
    content_correct = (
        not bool(row["content_switch"]) and float(row["target_WER"]) < 0.5
        and bool(str(row.get("output_text") or "").strip())
        and not bool(row["unrelated_short_output"])
    )
    if metric == "speaker_recovery":
        return speaker_correct
    if metric == "content_recovery":
        return content_correct
    if metric == "joint_recovery":
        return speaker_correct and content_correct
    content_regression = (
        not bool(primary["content_switch"]) and bool(row["content_switch"])
    )
    speaker_regression = (
        not bool(primary["acoustic_speaker_switch"])
        and bool(row["acoustic_speaker_switch"])
    )
    if metric == "content_regression":
        return content_regression
    if metric == "speaker_regression":
        return speaker_regression
    if metric == "joint_regression":
        return content_regression or speaker_regression
    if metric == "empty_output":
        return bool(row["empty_output"])
    if metric == "short_output":
        return bool(row["unrelated_short_output"])
    raise KeyError(metric)


def paired_comparisons(
    systems: dict[str, dict[str, dict[str, Any]]], ids: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    primary = systems["Primary WeSep"]
    for comparison, (base_name, new_name, question) in COMPARISONS.items():
        for cohort_name, cohort_value in (
            ("full_dev", None),
            ("natural_primary_swap", "natural_primary_swap"),
            ("primary_correct_control", "primary_correct_control"),
        ):
            selected_ids = [
                trial_id for trial_id in ids
                if cohort_value is None or primary[trial_id]["cohort"] == cohort_value
            ]
            rng = np.random.default_rng(SEED + ord(comparison) + len(selected_ids))
            for metric, key in (
                ("target_wer", "target_WER"),
                ("speaker_margin", "speaker_margin"),
                ("dnsmos_p808", "dnsmos_p808"),
            ):
                base = np.asarray([systems[base_name][trial_id][key] for trial_id in selected_ids], dtype=float)
                new = np.asarray([systems[new_name][trial_id][key] for trial_id in selected_ids], dtype=float)
                result = continuous_test(base, new, rng)
                output.append({
                    "comparison": comparison, "question": question,
                    "cohort": cohort_name, "base_system": base_name,
                    "new_system": new_name, "metric": metric,
                    "n": len(selected_ids), **result,
                })
            binary_metrics = ["content_switch", "acoustic_switch", "empty_output", "short_output"]
            if cohort_value == "natural_primary_swap":
                binary_metrics += ["speaker_recovery", "content_recovery", "joint_recovery"]
            if cohort_value == "primary_correct_control":
                binary_metrics += ["content_regression", "speaker_regression", "joint_regression"]
            for metric in binary_metrics:
                base = np.asarray([
                    derived_binary(systems[base_name][trial_id], metric, primary[trial_id])
                    for trial_id in selected_ids
                ])
                new = np.asarray([
                    derived_binary(systems[new_name][trial_id], metric, primary[trial_id])
                    for trial_id in selected_ids
                ])
                result = binary_test(base, new, rng)
                output.append({
                    "comparison": comparison, "question": question,
                    "cohort": cohort_name, "base_system": base_name,
                    "new_system": new_name, "metric": metric,
                    "n": len(selected_ids), **result,
                })
    return output


def find_comparison(
    rows: list[dict[str, Any]], comparison: str, cohort: str, metric: str,
) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["comparison"] == comparison and row["cohort"] == cohort
        and row["metric"] == metric
    )


def decide_values(
    summaries: dict[str, dict[str, dict[str, Any]]],
    paired: list[dict[str, Any]],
) -> tuple[str, str]:
    qwen_wer = find_comparison(paired, "A", "full_dev", "target_wer")
    qwen_content = find_comparison(paired, "A", "full_dev", "content_switch")
    qwen_speaker = find_comparison(paired, "A", "full_dev", "acoustic_switch")
    if (
        qwen_wer["ci95_high"] < 0
        and qwen_content["absolute_difference_new_minus_base"] <= 0
        and qwen_speaker["absolute_difference_new_minus_base"] <= 0.0025
    ):
        qwen = "YES"
    elif (
        qwen_wer["absolute_difference_new_minus_base"] >= -0.005
        and (
            qwen_content["absolute_difference_new_minus_base"] > 0.01
            or qwen_speaker["absolute_difference_new_minus_base"] > 0.01
        )
    ) or qwen_wer["absolute_difference_new_minus_base"] >= 0.10:
        qwen = "NO"
    else:
        qwen = "MIXED"

    csg_wer = find_comparison(paired, "B", "full_dev", "target_wer")
    csg_content = find_comparison(paired, "B", "full_dev", "content_switch")
    csg_speaker = find_comparison(paired, "B", "full_dev", "speaker_margin")
    csg_mos = find_comparison(paired, "B", "full_dev", "dnsmos_p808")
    if (
        csg_wer["absolute_difference_new_minus_base"] <= -0.01
        and csg_wer["ci95_high"] < 0
        and csg_content["absolute_difference_new_minus_base"] <= 0.0025
        and csg_speaker["absolute_difference_new_minus_base"] >= -0.01
        and csg_mos["absolute_difference_new_minus_base"] >= -0.03
    ):
        csg = "YES"
    elif (
        csg_wer["absolute_difference_new_minus_base"] >= 0
        and (
            csg_content["absolute_difference_new_minus_base"] > 0.0025
            or csg_speaker["absolute_difference_new_minus_base"] < -0.01
            or csg_mos["absolute_difference_new_minus_base"] < -0.03
        )
    ):
        csg = "NO"
    else:
        csg = "MIXED"
    return qwen, csg


def candidate_costs() -> dict[str, Any]:
    components = {}
    for name in ("full", "segments", "tfmap"):
        path = ANALYSIS / f"cost_benchmark/{name}/inference_summary.json"
        components[name] = json.loads(path.read_text(encoding="utf-8"))
    selection_path = ANALYSIS / "cost_benchmark/selection_cost.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.is_file() else None
    )
    return {"components": components, "selection": selection}


def phase_peak_vram(system_slug: str) -> int:
    path = ANALYSIS / "resource_guard.jsonl"
    if not path.is_file():
        return 0
    peak = 0
    for row in read_jsonl(path):
        phase = str(row.get("phase") or "")
        if system_slug in phase:
            gpu = row.get("snapshot", {}).get("gpu", {})
            if gpu.get("available"):
                peak = max(peak, int(gpu.get("used_bytes", 0)))
    return peak


def qfull_cost(slug: str, pool: str, candidate_rtf: float) -> tuple[float, float]:
    token_rows = read_jsonl(SYSTEM_ROOT / f"{slug}/per_trial_tokens.jsonl")
    audio_rows = read_jsonl(SYSTEM_ROOT / f"{slug}/audio_metrics.jsonl")
    manifest = {row["trial_id"]: row for row in read_jsonl(
        ROOT / "manifests/selected_evidence_dev_evaluation.jsonl"
    )}
    audio_seconds = sum(
        float(manifest[row["trial_id"]].get("mixture_num_samples", 0)) / 16000
        if manifest[row["trial_id"]].get("mixture_num_samples") else 0
        for row in token_rows
    )
    if not audio_seconds:
        import soundfile as sf  # local import keeps table-only startup light
        audio_seconds = sum(
            sf.info(manifest[row["trial_id"]]["mixture_wav"]).frames / 16000
            for row in token_rows
        )
    qfull_seconds = sum(float(row["decode_seconds"]) for row in token_rows)
    synth_seconds = sum(float(row.get("synthesis_seconds", 0.0)) for row in audio_rows)
    peak = max(
        [int(row.get("cuda_reserved_bytes", 0)) for row in token_rows]
        + [phase_peak_vram(f"selected_audio_{slug}")]
    )
    return candidate_rtf + (qfull_seconds + synth_seconds) / audio_seconds, peak / 1024 ** 3


def fmt_percent(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{100 * value:.{digits}f}%"


def fmt_number(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_cost_table(
    summaries: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str]:
    cost = candidate_costs()
    components = cost["components"]
    selector = cost["selection"]
    generation_rtf = {
        "primary": float(components["full"]["candidate_generation_rtf"]),
        "pool_b": float(components["full"]["candidate_generation_rtf"])
        + float(components["tfmap"]["candidate_generation_rtf"]),
        "pool_d": float(components["full"]["candidate_generation_rtf"])
        + float(components["segments"]["candidate_generation_rtf"])
        + float(components["tfmap"]["candidate_generation_rtf"]),
    }
    selection_rtf = {"primary": 0.0, "pool_b": 0.0, "pool_d": 0.0}
    selection_note = "selector cost unavailable; reported RTF is a lower bound"
    if selector is not None:
        selection_rtf.update({
            "primary": float(selector["pools"]["Pool A"]["candidate_selection_rtf"]),
            "pool_b": float(selector["pools"]["Pool B"]["candidate_selection_rtf"]),
            "pool_d": float(selector["pools"]["Pool D"]["candidate_selection_rtf"]),
        })
        selection_note = "candidate generation + enrollment-cosine selection; generative rows also include Q-Full and synthesis"
    direct_rtf = {
        key: generation_rtf[key] + selection_rtf[key] for key in generation_rtf
    }
    candidate_peak = {
        "primary": components["full"]["peak_torch_cuda_reserved_bytes"] / 1024 ** 3,
        "pool_b": max(
            components["full"]["peak_torch_cuda_reserved_bytes"],
            components["tfmap"]["peak_torch_cuda_reserved_bytes"],
        ) / 1024 ** 3,
        "pool_d": max(
            components["full"]["peak_torch_cuda_reserved_bytes"],
            components["segments"]["peak_torch_cuda_reserved_bytes"],
            components["tfmap"]["peak_torch_cuda_reserved_bytes"],
        ) / 1024 ** 3,
    }
    b_rtf, b_peak = qfull_cost("pool_b_qfull_csg", "pool_b", direct_rtf["pool_b"])
    d_rtf, d_peak = qfull_cost("pool_d_qfull_csg", "pool_d", direct_rtf["pool_d"])
    definitions = [
        ("Primary only", 1, 0, 0, direct_rtf["primary"], candidate_peak["primary"], "Primary WeSep"),
        ("Pool B deterministic", 2, 0, 0, direct_rtf["pool_b"], candidate_peak["pool_b"], "Pool B Selected Candidate"),
        ("Pool D deterministic", 5, 0, 0, direct_rtf["pool_d"], candidate_peak["pool_d"], "Pool D Selected Candidate"),
        ("Pool B + Q-Full CSG", 2, 1, 1, b_rtf, max(candidate_peak["pool_b"], b_peak), "Pool B Selected -> Q-Full + CSG"),
        ("Pool D + Q-Full CSG", 5, 1, 1, d_rtf, max(candidate_peak["pool_d"], d_peak), "Pool D Selected -> Q-Full + CSG"),
    ]
    rows = []
    for label, passes, qfull, csg, rtf, peak, system in definitions:
        swap = summaries[system]["natural_primary_swap"]
        recovery = (
            swap.get("candidate_target_correct_rate")
            if SYSTEMS[system]["kind"] == "deterministic"
            else swap["joint_recovery_rate"]
        )
        rows.append({
            "system": label, "extractor_passes": passes,
            "qfull_pass": qfull, "csg": csg, "rtf": rtf,
            "peak_vram_gib": peak,
            "target_wer": summaries[system]["full_dev"]["target_wer"],
            "swap_recovery": recovery,
        })
    return rows, selection_note


def choose_system(
    summaries: dict[str, dict[str, dict[str, Any]]], qwen: str, csg: str,
) -> str:
    eligible = []
    candidates = (
        "Pool B Selected Candidate", "Pool D Selected Candidate",
        "Pool B Selected -> Q-Full UD", "Pool B Selected -> Q-Full + CSG",
        "Pool D Selected -> Q-Full UD", "Pool D Selected -> Q-Full + CSG",
    )
    for name in candidates:
        full = summaries[name]["full_dev"]
        control = summaries[name]["primary_correct_control"]
        if (
            full["acoustic_switch_rate"] <= 0.02
            and control["joint_regression_rate"] <= 0.03
            and full["decode_failure_rate"] == 0
        ):
            eligible.append(name)
    if not eligible:
        return "Pool B Selected Candidate"
    best = min(eligible, key=lambda name: (
        summaries[name]["full_dev"]["target_wer"],
        summaries[name]["full_dev"]["content_switch_rate"],
    ))
    if best.startswith("Pool D"):
        counterpart = best.replace("Pool D", "Pool B", 1)
        if counterpart in eligible:
            b = summaries[counterpart]["full_dev"]
            d = summaries[best]["full_dev"]
            if (
                b["target_wer"] <= d["target_wer"] + 0.005
                and b["content_switch_rate"] <= d["content_switch_rate"] + 0.002
            ):
                best = counterpart
    if qwen == "NO" and "Q-Full" in best:
        direct = "Pool D Selected Candidate" if best.startswith("Pool D") else "Pool B Selected Candidate"
        if direct in eligible:
            best = direct
    if csg == "NO" and "+ CSG" in best:
        ud = best.replace(" + CSG", " UD")
        if ud in eligible:
            best = ud
    return best


def build_report(
    summaries: dict[str, dict[str, dict[str, Any]]], paired: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]], cost_note: str, qwen: str, csg: str,
    recommended: str, verdict: str,
) -> str:
    full_rows = []
    swap_rows = []
    control_rows = []
    for system, meta in SYSTEMS.items():
        full = summaries[system]["full_dev"]
        swap = summaries[system]["natural_primary_swap"]
        control = summaries[system]["primary_correct_control"]
        full_rows.append([
            system, meta["pool"], fmt_percent(full["target_wer"]),
            fmt_percent(full["content_switch_rate"]),
            fmt_percent(full["acoustic_switch_rate"]),
            fmt_number(full["speaker_margin"]), fmt_number(full["dnsmos_p808"]),
            fmt_percent(full["p95_wer"]), fmt_percent(full["p99_wer"]),
        ])
        swap_rows.append([
            system, fmt_percent(swap["target_wer"]),
            fmt_percent(swap["content_switch_rate"]),
            fmt_percent(swap["acoustic_switch_rate"]),
            fmt_number(swap["speaker_margin"]),
            fmt_percent(swap["speaker_recovery_rate"]),
            fmt_percent(swap["content_recovery_rate"]),
            fmt_percent(swap["joint_recovery_rate"]),
            fmt_number(swap["dnsmos_p808"]),
        ])
        control_rows.append([
            system, fmt_percent(control["target_wer"]),
            fmt_percent(control["content_switch_rate"]),
            fmt_percent(control["acoustic_switch_rate"]),
            fmt_number(control["speaker_margin"]),
            fmt_percent(control["content_regression_rate"]),
            fmt_percent(control["speaker_regression_rate"]),
            fmt_percent(control["joint_regression_rate"]),
            fmt_number(control["dnsmos_p808"]),
        ])
    paired_rows = []
    for label in COMPARISONS:
        for metric in ("target_wer", "speaker_margin", "dnsmos_p808"):
            row = find_comparison(paired, label, "full_dev", metric)
            paired_rows.append([
                label, metric, fmt_number(row["base_mean"], 4),
                fmt_number(row["new_mean"], 4),
                fmt_number(row["absolute_difference_new_minus_base"], 4),
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]",
                f"{row['p_value']:.4g}",
            ])
    cost_md = [[
        row["system"], str(row["extractor_passes"]), str(row["qfull_pass"]),
        str(row["csg"]), f"{row['rtf']:.3f}", f"{row['peak_vram_gib']:.2f}",
        fmt_percent(row["target_wer"]), fmt_percent(row["swap_recovery"]),
    ] for row in cost_rows]
    pool_b = summaries["Pool B Selected Candidate"]["natural_primary_swap"]
    pool_d = summaries["Pool D Selected Candidate"]["natural_primary_swap"]
    primary = summaries["Primary WeSep"]["full_dev"]
    d_csg = summaries["Pool D Selected -> Q-Full + CSG"]["full_dev"]
    lines = [
        "# Selected-Evidence ICASSP 2027 Final Table Report",
        "",
        "## Technical summary",
        "",
        f"The frozen candidate selector recovers {fmt_percent(pool_b['candidate_target_correct_rate'])} "
        f"of the 405 primary swaps with Pool B and {fmt_percent(pool_d['candidate_target_correct_rate'])} "
        "with Pool D. The following full-system results determine whether Q-Full and CSG add value after that correction.",
        "",
        f"- `QWEN_ADDITIONAL_VALUE = {qwen}`",
        f"- `CSG_ADDITIONAL_VALUE = {csg}`",
        f"- `RECOMMENDED_FINAL_SYSTEM = {recommended}`",
        f"- `ICASSP_MAINLINE = {verdict}`",
        "- `TEST_USED = NO`",
        "",
        "## Table 1: Full DEV",
        "",
        "All 6,000 frozen natural DEV trials; the nine ambiguous cases are included only here.",
        "",
        markdown_table(
            ["System", "Candidate Pool", "Target WER ↓", "Content Switch ↓", "Acoustic Switch ↓", "Speaker Margin ↑", "DNSMOS ↑", "p95 WER ↓", "p99 WER ↓"],
            full_rows,
        ),
        "",
        "## Table 2: 405 primary-swap cases",
        "",
        "Recovery requires target-consistent content and positive acoustic speaker margin; empty and unrelated-short outputs never count as recovery.",
        "",
        markdown_table(
            ["System", "Target WER ↓", "Content Switch ↓", "Acoustic Switch ↓", "Speaker Margin ↑", "Speaker Recovery ↑", "Content Recovery ↑", "Joint Recovery ↑", "DNSMOS ↑"],
            swap_rows,
        ),
        "",
        "## Table 3: 5,586 primary-correct cases",
        "",
        "Regression is paired to each trial's unchanged Primary WeSep content and acoustic-speaker correctness.",
        "",
        markdown_table(
            ["System", "Target WER ↓", "Content Switch ↓", "Acoustic Switch ↓", "Speaker Margin ↑", "Content Regression ↓", "Speaker Regression ↓", "Joint Regression ↓", "DNSMOS ↑"],
            control_rows,
        ),
        "",
        "## The four paired comparisons",
        "",
        "Differences are new minus base. Continuous metrics use 10,000 paired bootstrap resamples for the 95% CI and a paired sign-flip p-value. Binary endpoints use exact McNemar with paired bootstrap CIs; their complete results are in `paired_comparisons.csv`.",
        "",
        markdown_table(
            ["Comparison", "Metric", "Base", "New", "Δ new-base", "95% CI", "p"],
            paired_rows,
        ),
        "",
        "## Table 4: Performance / cost trade-off",
        "",
        f"RTF scope: {cost_note}. Extractor passes count candidate-equivalent outputs; the three segmented views may be batched in one model call but still produce three candidates.",
        "",
        markdown_table(
            ["System", "Extractor Passes", "Q-Full Pass", "CSG", "RTF ↓", "Peak VRAM GiB ↓", "Target WER ↓", "Swap Recovery ↑"],
            cost_md,
        ),
        "",
        "## Scope and metric definitions",
        "",
        "- DEV contains 6,000 trials: 405 frozen high-confidence primary swaps, 5,586 primary-correct controls, and nine ambiguous cases.",
        "- Selection is the frozen enrollment-only ECAPA cosine argmax. Pool B is `full + tfmap_context_full`; Pool D contains all five frozen candidates.",
        "- Target WER and content switch use the same Whisper-small.en greedy ASR-consistency protocol. Acoustic switch and speaker margin use the same CosyVoice/CAMPPlus evaluation backend across all table systems.",
        "- Speaker recovery requires acoustic speaker margin above zero. Content recovery requires no content switch, target WER below 50%, and a nonempty, non-short output. Joint recovery requires both.",
        "- Deterministic SI-SDR and target-correct rates remain auxiliary candidate diagnostics; generative SI-SDR sign is not used as the primary identity decision.",
        "- Every included system passed expected 6,000, unique 6,000, missing zero, duplicate zero, and failed zero.",
        "",
        "## Direct answers for the paper",
        "",
        f"1. **Is alternative candidate selection sufficient?** Pool B and Pool D candidate recovery are {fmt_percent(pool_b['candidate_target_correct_rate'])} and {fmt_percent(pool_d['candidate_target_correct_rate'])}, with the corresponding control regression visible in Table 3. Selection is the dominant first correction step, but residual failures remain.",
        f"2. **Does Qwen improve the selected candidate?** `{qwen}` under the frozen paired decision rule.",
        f"3. **Does CSG still improve selected evidence?** `{csg}` for lambda 1 versus matching selected-evidence UD.",
        f"4. **Where does the contribution come from?** The evidence supports the combination stated by the Qwen/CSG verdicts above; candidate diversity plus speaker-consistent selection is the prerequisite contribution and any claimed refinement benefit is limited to statistically supported paired changes.",
        "",
        "## Limitations and robustness",
        "",
        "The candidate checkpoints and all decisions remain DEV-frozen; no causal claim extends beyond this experiment. DNSMOS is non-intrusive, WER is ASR-consistency rather than human-reference WER, and candidate-generation cost is measured on the locked 100-trial smoke set. The nine ambiguous cases are excluded from swap/control claims. Natural TEST was not read or run.",
        "",
        "## Recommended next step",
        "",
        f"Use **{recommended}** as the frozen DEV recommendation. Do not tune candidate definitions, selector weights, Q-Full checkpoint, CSG lambda, metrics, or failure policy from these results. Preserve natural TEST for a separately authorized final evaluation.",
        "",
        "## Further question",
        "",
        "The only decision-changing open question is whether the frozen recommendation reproduces on an untouched final evaluation set; that question is intentionally unanswered here because TEST use is prohibited.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids, cohort, selections, candidate_metrics = load_candidates()
    systems = {
        name: validate_system(name, meta["path"], ids, cohort)
        for name, meta in SYSTEMS.items()
    }
    enrich_deterministic(systems, selections, candidate_metrics)
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    primary = systems["Primary WeSep"]
    table_rows: list[dict[str, Any]] = []
    for name, meta in SYSTEMS.items():
        summaries[name] = {}
        for cohort_name, cohort_value in COHORTS.items():
            rows = subset_rows(systems[name], ids, cohort_value)
            summary = summarize(rows, primary)
            summaries[name][cohort_name] = summary
            table_rows.append({
                "cohort": cohort_name, "system": name,
                "candidate_pool": meta["pool"], "system_kind": meta["kind"],
                "expected": len(rows), "unique": len(rows), "missing": 0,
                "duplicate": 0, "failed": 0, **summary,
            })

    pool_b_swap = summaries["Pool B Selected Candidate"]["natural_primary_swap"]
    pool_d_swap = summaries["Pool D Selected Candidate"]["natural_primary_swap"]
    pool_b_control = summaries["Pool B Selected Candidate"]["primary_correct_control"]
    pool_d_control = summaries["Pool D Selected Candidate"]["primary_correct_control"]
    expected_b = 337 / 405
    expected_d = 355 / 405
    expected_control = 2 / 5586
    if not (
        math.isclose(pool_b_swap["candidate_target_correct_rate"], expected_b, abs_tol=1e-15)
        and math.isclose(pool_d_swap["candidate_target_correct_rate"], expected_d, abs_tol=1e-15)
        and math.isclose(pool_b_control["candidate_target_correct_rate"], 1 - expected_control, abs_tol=1e-15)
        and math.isclose(pool_d_control["candidate_target_correct_rate"], 1 - expected_control, abs_tol=1e-15)
    ):
        raise ValueError("STOP: frozen candidate recovery/control audit changed")

    csv_path = RESULTS / "MAIN_ICASSP_TABLE.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    paired = paired_comparisons(systems, ids)
    paired_path = RESULTS / "paired_comparisons.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        # Continuous and binary paired tests intentionally have different
        # payloads: McNemar rows add the two discordant-cell counts.  Build the
        # CSV schema from the ordered union rather than from the first
        # (continuous) row so those later fields are preserved.
        paired_fields = list(dict.fromkeys(
            key for row in paired for key in row
        ))
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired)
    atomic_json(RESULTS / "selected_evidence_summary.json", summaries)
    atomic_json(RESULTS / "paired_comparisons.json", paired)

    qwen, csg = decide_values(summaries, paired)
    cost_rows, cost_note = build_cost_table(summaries)
    with (RESULTS / "performance_cost_tradeoff.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)
    recommended = choose_system(summaries, qwen, csg)
    direct_d = summaries["Pool D Selected Candidate"]["full_dev"]
    best = summaries[recommended]["full_dev"]
    no_go = (
        best["target_wer"] - direct_d["target_wer"] >= 0.10
        or best["content_switch_rate"] - direct_d["content_switch_rate"] >= 0.05
        or best["short_output_rate"] >= 0.05
        or summaries[recommended]["primary_correct_control"]["joint_regression_rate"] >= 0.05
    )
    if no_go:
        verdict = "NOT_SUPPORTED_YET"
    elif qwen == "NO" and csg == "NO":
        verdict = "PARTIALLY_SUPPORTED"
    elif (
        (qwen == "YES" or csg == "YES")
        and
        best["target_wer"] <= direct_d["target_wer"]
        and best["content_switch_rate"] <= direct_d["content_switch_rate"]
        and best["acoustic_switch_rate"] <= 0.02
    ):
        verdict = "SUPPORTED"
    else:
        verdict = "PARTIALLY_SUPPORTED"

    report = build_report(
        summaries, paired, cost_rows, cost_note, qwen, csg, recommended, verdict
    )
    atomic_text(ROOT / "docs/SELECTED_EVIDENCE_FINAL_TABLE_REPORT.md", report)
    verdict_text = f"""# ICASSP 2027 Selected-Evidence Verdict

`{verdict}`

The frozen 6,000-trial DEV analysis gives `QWEN_ADDITIONAL_VALUE = {qwen}` and
`CSG_ADDITIONAL_VALUE = {csg}`. The recommended frozen system is
**{recommended}**. Candidate selection recovers {100 * expected_b:.2f}% of
primary swaps with Pool B and {100 * expected_d:.2f}% with Pool D while the
candidate target-correct regression rate on 5,586 controls is
{100 * expected_control:.4f}% for both pools.

This verdict is restricted to natural DEV. It does not authorize selector,
candidate-pool, checkpoint, CSG-lambda, metric, or failure-policy tuning.

`TEST_USED = NO`
"""
    atomic_text(ROOT / "docs/ICASSP2027_SELECTED_EVIDENCE_VERDICT.md", verdict_text)
    cost_text = "# Candidate Pool Cost Report\n\n" + markdown_table(
        ["System", "Extractor Passes", "Q-Full Pass", "CSG", "RTF", "Peak VRAM GiB", "Target WER", "Swap Recovery"],
        [[
            row["system"], str(row["extractor_passes"]), str(row["qfull_pass"]),
            str(row["csg"]), f"{row['rtf']:.3f}", f"{row['peak_vram_gib']:.2f}",
            fmt_percent(row["target_wer"]), fmt_percent(row["swap_recovery"]),
        ] for row in cost_rows],
    ) + f"\n\n{cost_note}. `TEST_USED = NO`.\n"
    atomic_text(ROOT / "docs/CANDIDATE_POOL_COST_REPORT.md", cost_text)
    audit = {
        "status": "PASS", "expected": EXPECTED, "unique": EXPECTED,
        "missing": 0, "duplicate": 0, "failed": 0,
        "cohorts": {"full_dev": 6000, "natural_primary_swap": 405,
                    "primary_correct_control": 5586, "ambiguous_primary_wrong": 9},
        "candidate_order": ["full", "first", "middle", "final", "tfmap_context_full"],
        "pool_b_swap_recovery": expected_b,
        "pool_d_swap_recovery": expected_d,
        "pool_b_control_regression": expected_control,
        "pool_d_control_regression": expected_control,
        "qfull_checkpoint_epoch": 4, "qfull_checkpoint_step": 13900,
        "csg_lambda": 1.0, "csg_w": 0.0, "test_used": False,
    }
    audit_text = "# Candidate Integration Audit\n\n```json\n" + json.dumps(audit, indent=2) + "\n```\n"
    atomic_text(ROOT / "docs/CANDIDATE_INTEGRATION_AUDIT.md", audit_text)
    terminal = {
        "FULL_DEV_COMPLETE": "YES",
        "POOL_B_SWAP_RECOVERY": pool_b_swap["candidate_target_correct_rate"],
        "POOL_D_SWAP_RECOVERY": pool_d_swap["candidate_target_correct_rate"],
        "SELECTED_DETERMINISTIC_TARGET_WER": direct_d["target_wer"],
        "ORIGINAL_QFULL_CSG_TARGET_WER": summaries["Original Q-Full + CSG"]["full_dev"]["target_wer"],
        "POOL_B_QFULL_UD_TARGET_WER": summaries["Pool B Selected -> Q-Full UD"]["full_dev"]["target_wer"],
        "POOL_B_QFULL_CSG_TARGET_WER": summaries["Pool B Selected -> Q-Full + CSG"]["full_dev"]["target_wer"],
        "POOL_D_QFULL_UD_TARGET_WER": summaries["Pool D Selected -> Q-Full UD"]["full_dev"]["target_wer"],
        "POOL_D_QFULL_CSG_TARGET_WER": summaries["Pool D Selected -> Q-Full + CSG"]["full_dev"]["target_wer"],
        "POOL_D_SELECTED_CONTENT_SWITCH": direct_d["content_switch_rate"],
        "POOL_D_QFULL_CSG_CONTENT_SWITCH": summaries["Pool D Selected -> Q-Full + CSG"]["full_dev"]["content_switch_rate"],
        "POOL_D_QFULL_CSG_ACOUSTIC_SWITCH": summaries["Pool D Selected -> Q-Full + CSG"]["full_dev"]["acoustic_switch_rate"],
        "POOL_D_QFULL_CSG_SPEAKER_MARGIN": summaries["Pool D Selected -> Q-Full + CSG"]["full_dev"]["speaker_margin"],
        "POOL_D_QFULL_CSG_DNSMOS": summaries["Pool D Selected -> Q-Full + CSG"]["full_dev"]["dnsmos_p808"],
        "QWEN_ADDITIONAL_VALUE": qwen,
        "CSG_ADDITIONAL_VALUE": csg,
        "RECOMMENDED_FINAL_SYSTEM": recommended,
        "ICASSP_MAINLINE": verdict,
        "TEST_USED": "NO",
        "MAIN_TABLE": "results/selected_evidence/MAIN_ICASSP_TABLE.csv",
        "REPORT": "docs/SELECTED_EVIDENCE_FINAL_TABLE_REPORT.md",
    }
    atomic_json(RESULTS / "terminal_summary.json", terminal)
    for key, value in terminal.items():
        print(f"{key}:\n{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
