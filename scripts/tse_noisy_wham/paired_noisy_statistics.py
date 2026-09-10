#!/usr/bin/env python3
"""Registered paired bootstrap and exact McNemar tests for noisy TSE."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
CONTINUOUS = (
    "target_WER_raw", "lps", "speechbertscore", "speaker_margin",
    "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovl", "utmos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1986)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def finite(value: Any) -> bool:
    return value is not None and not isinstance(value, bool) and math.isfinite(float(value))


def holm(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["cohort"], row["family"], row["metric"]), []).append(row)
    for values in groups.values():
        ordered = sorted(values, key=lambda row: row["p_value"])
        running = 0.0
        m = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (m - index) * row["p_value"])
            running = max(running, adjusted)
            row["holm_adjusted_p"] = running


def main() -> int:
    args = parse_args()
    compiled = ROOT / f"results/noisy_wham/{args.split}/compiled"
    gnr = json.loads((
        ROOT / "analysis/noisy_wham/dev/full/gnr_selection.json"
    ).read_text())
    slugs = {
        "D0": "primary_wesep", "D3": "pool_d_selected",
        "G2": "pool_d_qfull_ud", "G3": "pool_d_fixed_csg",
        "G5": "pool_d_adaptive_csg",
        "G6": gnr["selected_dev_slug"] if args.split == "dev" else "pool_d_adaptive_csg_gnr",
    }
    systems = {
        code: {row["trial_id"]: row for row in read_jsonl(compiled / f"{slug}.jsonl")}
        for code, slug in slugs.items()
        if (compiled / f"{slug}.jsonl").is_file()
    }
    best = json.loads((
        ROOT / "analysis/noisy_wham/dev/full/best_generative_selection.json"
    ).read_text())["selected_code"]
    comparisons = (
        ("D0_vs_D3", "D0", "D3"),
        ("G2_vs_G3", "G2", "G3"),
        ("G3_vs_G5", "G3", "G5"),
        ("G5_vs_G6", "G5", "G6"),
        ("D3_vs_BEST_GENERATIVE", "D3", best),
    )
    results = []
    rng = np.random.default_rng(args.seed)
    for comparison, base_code, new_code in comparisons:
        cohort = "natural"
        ids = sorted(
            trial_id for trial_id in set(systems[base_code]) & set(systems[new_code])
            if systems[base_code][trial_id]["benchmark"] == cohort
        )
        if not ids:
            raise ValueError(f"empty paired cohort: {comparison}/{cohort}")
        # Shared bootstrap counts for every metric in this comparison. Reusing
        # resamples preserves pairing and avoids materializing full logits/audio.
        counts = rng.multinomial(
            len(ids), np.full(len(ids), 1.0 / len(ids)), size=args.resamples
        ).astype(np.float32)
        counts /= len(ids)
        for metric in CONTINUOUS:
            valid_ids = [
                trial_id for trial_id in ids
                if finite(systems[base_code][trial_id].get(metric))
                and finite(systems[new_code][trial_id].get(metric))
            ]
            if not valid_ids:
                continue
            if len(valid_ids) == len(ids):
                weights = counts
            else:
                # Missing metrics use complete pairs only, with their own exact
                # registered-size bootstrap and explicit coverage.
                weights = rng.multinomial(
                    len(valid_ids), np.full(len(valid_ids), 1.0 / len(valid_ids)),
                    size=args.resamples,
                ).astype(np.float32) / len(valid_ids)
            differences = np.asarray([
                float(systems[new_code][trial_id][metric])
                - float(systems[base_code][trial_id][metric])
                for trial_id in valid_ids
            ], dtype=np.float32)
            boot = weights @ differences
            p = min(1.0, 2.0 * min(
                (np.count_nonzero(boot <= 0) + 1) / (args.resamples + 1),
                (np.count_nonzero(boot >= 0) + 1) / (args.resamples + 1),
            ))
            results.append({
                "split": args.split, "cohort": cohort,
                "comparison": comparison, "base": base_code, "new": new_code,
                "family": "continuous", "metric": metric,
                "n_pairs": len(valid_ids), "coverage": len(valid_ids) / len(ids),
                "mean_base": float(np.mean([
                    systems[base_code][trial_id][metric] for trial_id in valid_ids
                ])),
                "mean_new": float(np.mean([
                    systems[new_code][trial_id][metric] for trial_id in valid_ids
                ])),
                "mean_difference_new_minus_base": float(np.mean(differences)),
                "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "p_value": float(p), "resamples": args.resamples,
            })
        del counts
        binary_specs = [
            ("content_switch", "content_switch", ids, False, "natural"),
            ("acoustic_speaker_switch", "acoustic_speaker_switch", ids, False, "natural"),
            ("high_error", "high_error", ids, False, "natural"),
        ]
        swap_ids = [trial_id for trial_id in ids
                    if systems[base_code][trial_id]["noisy_primary_swap"]]
        control_ids = [trial_id for trial_id in ids
                       if systems[base_code][trial_id]["noisy_primary_correct_control"]]
        binary_specs.extend([
            ("joint_success", "joint_speaker_content_recovery", swap_ids, False,
             "noisy_primary_swaps"),
            ("control_regression", "speaker_binding_correct", control_ids, True,
             "noisy_primary_correct_controls"),
        ])
        if comparison == "D0_vs_D3":
            binary_specs.append((
                "candidate_recovery", "speaker_binding_correct", swap_ids, False,
                "noisy_primary_swaps",
            ))
        for metric, field, metric_ids, invert, binary_cohort in binary_specs:
            valid_ids = [
                trial_id for trial_id in metric_ids
                if systems[base_code][trial_id].get(field) is not None
                and systems[new_code][trial_id].get(field) is not None
            ]
            base = np.asarray([
                bool(systems[base_code][trial_id][field]) for trial_id in valid_ids
            ])
            new = np.asarray([
                bool(systems[new_code][trial_id][field]) for trial_id in valid_ids
            ])
            if invert:
                base, new = ~base, ~new
            base_only = int(np.count_nonzero(base & ~new))
            new_only = int(np.count_nonzero(~base & new))
            discordant = base_only + new_only
            p = float(binomtest(base_only, discordant, 0.5).pvalue) if discordant else 1.0
            results.append({
                "split": args.split, "cohort": binary_cohort,
                "comparison": comparison, "base": base_code, "new": new_code,
                "family": "binary", "metric": metric,
                "n_pairs": len(valid_ids),
                "coverage": len(valid_ids) / len(metric_ids) if metric_ids else None,
                "rate_base": float(np.mean(base)), "rate_new": float(np.mean(new)),
                "difference_new_minus_base": float(np.mean(new) - np.mean(base)),
                "base_true_new_false": base_only,
                "base_false_new_true": new_only,
                "p_value": p, "test": "two-sided exact McNemar/binomial",
            })
    holm(results)
    output = ROOT / f"results/noisy_wham/{args.split}/paired_statistics.json"
    atomic(output, {
        "status": "COMPLETE", "split": args.split,
        "bootstrap_seed": args.seed, "bootstrap_resamples": args.resamples,
        "results": results, "test_used": args.split == "test",
    })
    print(json.dumps({
        "status": "COMPLETE", "split": args.split, "tests": len(results),
        "bootstrap_resamples": args.resamples, "test_used": args.split == "test",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
