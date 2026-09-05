#!/usr/bin/env python3
"""Evaluate content-preserving direct/grounded selection on cached DEV outputs.

The script never regenerates audio.  It joins cached Pool-D direct and fixed-CSG
metrics, applies inference-only gate features, and writes per-trial decisions,
aggregate summaries, per-condition results, and mixture-cluster bootstrap CIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from llm_tse_grounding.content_gate import (
    STRONG_GUARD_DEV_POLICY,
    TOKEN_ONLY_POLICY,
    ContentGatePolicy,
    choose_content_preserving_output,
)


POLICIES = {
    "token_only_gate": TOKEN_ONLY_POLICY,
    "strong_output_guard": STRONG_GUARD_DEV_POLICY,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_unique(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        trial_id = row["trial_id"]
        if trial_id in indexed:
            raise ValueError(f"duplicate trial_id in {source}: {trial_id}")
        indexed[trial_id] = row
    return indexed


def selected_enrollment_cosine(candidate: dict[str, Any]) -> float:
    selected = candidate["pool_d_selected"]
    return float(candidate[f"{selected}_enrollment_cosine"])


def summarize(rows: list[dict[str, Any]], system: str, scope: str) -> dict[str, Any]:
    chosen = [row for row in rows if row["system"] == system]
    if scope != "all":
        chosen = [row for row in chosen if row["scope"] == scope]
    if not chosen:
        raise ValueError(f"no rows for system={system}, scope={scope}")
    refs = sum(row["reference_words"] for row in chosen)
    distances = sum(row["target_word_distance"] for row in chosen)
    dnsmos = [row["dnsmos_p808"] for row in chosen if row["dnsmos_p808"] is not None]
    return {
        "system": system,
        "scope": scope,
        "trials": len(chosen),
        "grounded_trials": sum(row["chosen_arm"] == "grounded" for row in chosen),
        "grounded_share": mean(row["chosen_arm"] == "grounded" for row in chosen),
        "macro_wer": mean(row["target_wer"] for row in chosen),
        "corpus_wer": distances / refs,
        "acoustic_switch_rate": mean(row["acoustic_speaker_switch"] for row in chosen),
        "content_switch_rate": mean(row["content_switch"] for row in chosen),
        "unrelated_short_output_rate": mean(row["unrelated_short_output"] for row in chosen),
        "dnsmos_p808": mean(dnsmos),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    rows: list[dict[str, Any]], policy: str, *, draws: int, seed: int
) -> dict[str, Any]:
    by_mixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    policy_rows = [row for row in rows if row["system"] == policy]
    fixed_by_trial = {
        row["trial_id"]: row for row in rows if row["system"] == "fixed_csg"
    }
    for row in policy_rows:
        by_mixture[row["mixture_id"]].append(row)
    mixture_ids = sorted(by_mixture)
    group_stats: dict[str, tuple[float, float, int]] = {}
    for mixture_id, group in by_mixture.items():
        wer_delta = 0.0
        switch_delta = 0.0
        for row in group:
            fixed = fixed_by_trial[row["trial_id"]]
            wer_delta += row["target_wer"] - fixed["target_wer"]
            switch_delta += (
                row["acoustic_speaker_switch"] - fixed["acoustic_speaker_switch"]
            )
        group_stats[mixture_id] = (wer_delta, switch_delta, len(group))

    rng = random.Random(seed)
    wer_draws: list[float] = []
    switch_draws: list[float] = []
    for _ in range(draws):
        sample = [rng.choice(mixture_ids) for _ in mixture_ids]
        wer_sum = switch_sum = 0.0
        count = 0
        for mixture_id in sample:
            group_wer, group_switch, group_n = group_stats[mixture_id]
            wer_sum += group_wer
            switch_sum += group_switch
            count += group_n
        wer_draws.append(wer_sum / count)
        switch_draws.append(switch_sum / count)

    point_wer = mean(
        row["target_wer"] - fixed_by_trial[row["trial_id"]]["target_wer"]
        for row in policy_rows
    )
    point_switch = mean(
        row["acoustic_speaker_switch"]
        - fixed_by_trial[row["trial_id"]]["acoustic_speaker_switch"]
        for row in policy_rows
    )
    return {
        "policy": policy,
        "comparison": "minus_fixed_csg",
        "cluster": "mixture_id",
        "draws": draws,
        "seed": seed,
        "wer_delta": point_wer,
        "wer_ci95": [percentile(wer_draws, 0.025), percentile(wer_draws, 0.975)],
        "acoustic_switch_delta": point_switch,
        "acoustic_switch_ci95": [
            percentile(switch_draws, 0.025),
            percentile(switch_draws, 0.975),
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--grounded", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20270905)
    args = parser.parse_args()

    direct = index_unique(read_jsonl(args.direct), "direct")
    grounded = index_unique(read_jsonl(args.grounded), "grounded")
    tokens = index_unique(read_jsonl(args.tokens), "tokens")
    candidates = index_unique(read_jsonl(args.candidates), "candidates")
    manifest = index_unique(read_jsonl(args.manifest), "manifest")
    trial_ids = set(direct)
    for name, indexed in {
        "grounded": grounded,
        "tokens": tokens,
        "candidates": candidates,
        "manifest": manifest,
    }.items():
        if set(indexed) != trial_ids:
            missing = sorted(trial_ids - set(indexed))[:3]
            extra = sorted(set(indexed) - trial_ids)[:3]
            raise ValueError(f"{name} coverage mismatch: missing={missing}, extra={extra}")

    output_rows: list[dict[str, Any]] = []
    for trial_id in sorted(trial_ids):
        direct_row = direct[trial_id]
        grounded_row = grounded[trial_id]
        token_row = tokens[trial_id]
        candidate_row = candidates[trial_id]
        manifest_row = manifest[trial_id]
        duration = grounded_row["output_num_samples"] / grounded_row["output_sample_rate"]
        cosine = selected_enrollment_cosine(candidate_row)
        flip_rate = float(token_row["token_flip_rate_vs_evidence"])

        decisions = {
            "pool_d_direct": ("direct", (), TOKEN_ONLY_POLICY),
            "fixed_csg": ("grounded", (), TOKEN_ONLY_POLICY),
        }
        for policy_name, policy in POLICIES.items():
            decision = choose_content_preserving_output(
                evidence_enrollment_cosine=cosine,
                token_flip_rate=flip_rate,
                output_duration_seconds=duration,
                grounded_transcript=grounded_row.get("output_text"),
                grounded_dnsmos_p808=grounded_row.get("dnsmos_p808"),
                policy=policy,
            )
            decisions[policy_name] = (decision.choice, decision.reasons, policy)

        for system, (chosen_arm, reasons, policy) in decisions.items():
            metric = grounded_row if chosen_arm == "grounded" else direct_row
            output_rows.append(
                {
                    "trial_id": trial_id,
                    "mixture_id": manifest_row["mixture_id"],
                    "scope": manifest_row["benchmark"],
                    "snr_db": manifest_row.get("snr_db"),
                    "system": system,
                    "chosen_arm": chosen_arm,
                    "gate_reasons": list(reasons),
                    "evidence_enrollment_cosine": cosine,
                    "token_flip_rate": flip_rate,
                    "policy": asdict(policy),
                    "target_wer": float(metric["target_WER"]),
                    "target_word_distance": int(metric["target_word_distance"]),
                    "reference_words": len(metric["target_text"].split()),
                    "acoustic_speaker_switch": int(metric["acoustic_speaker_switch"]),
                    "content_switch": int(metric["content_switch"]),
                    "unrelated_short_output": int(metric["unrelated_short_output"]),
                    "dnsmos_p808": metric.get("dnsmos_p808"),
                    "selected_output_wav": metric["output_wav"],
                    "test_used": False,
                }
            )

    systems = ["pool_d_direct", "fixed_csg", "token_only_gate", "strong_output_guard"]
    summaries = [summarize(output_rows, system, "all") for system in systems]
    summaries += [summarize(output_rows, system, "natural") for system in systems]

    conditions: list[dict[str, Any]] = []
    for system in systems:
        system_rows = [row for row in output_rows if row["system"] == system]
        labels = sorted(
            {"natural" if row["scope"] == "natural" else f"snr_{row['snr_db']:+g}" for row in system_rows}
        )
        for label in labels:
            group = [
                row
                for row in system_rows
                if ("natural" if row["scope"] == "natural" else f"snr_{row['snr_db']:+g}")
                == label
            ]
            refs = sum(row["reference_words"] for row in group)
            conditions.append(
                {
                    "system": system,
                    "condition": label,
                    "trials": len(group),
                    "grounded_share": mean(row["chosen_arm"] == "grounded" for row in group),
                    "macro_wer": mean(row["target_wer"] for row in group),
                    "corpus_wer": sum(row["target_word_distance"] for row in group) / refs,
                    "acoustic_switch_rate": mean(row["acoustic_speaker_switch"] for row in group),
                    "content_switch_rate": mean(row["content_switch"] for row in group),
                    "dnsmos_p808": mean(row["dnsmos_p808"] for row in group),
                }
            )

    bootstrap = [
        cluster_bootstrap(
            output_rows,
            policy,
            draws=args.bootstrap_draws,
            seed=args.seed,
        )
        for policy in POLICIES
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_trial_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "by_condition.csv", conditions)
    payload = {
        "status": "COMPLETE",
        "scope": "NOISY_DEV_ONLY",
        "test_used": False,
        "trials": len(trial_ids),
        "policies": {name: asdict(policy) for name, policy in POLICIES.items()},
        "summaries": summaries,
        "bootstrap": bootstrap,
        "input_files": {
            "direct": str(args.direct),
            "grounded": str(args.grounded),
            "tokens": str(args.tokens),
            "candidates": str(args.candidates),
            "manifest": str(args.manifest),
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
