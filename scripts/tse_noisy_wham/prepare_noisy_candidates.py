#!/usr/bin/env python3
"""Prepare leak-free noisy candidate inputs and assemble frozen Pool B/D choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/noisy_wham"
MANIFESTS = ROOT / "manifests/noisy_wham"
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
POOLS = {
    "pool_full": ("full",),
    "pool_b": ("full", "tfmap_context_full"),
    "pool_d": ORDER,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--split", choices=("dev", "test"), required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--split", choices=("dev", "test"), required=True)
    assemble.add_argument("--scope", choices=("unit20", "smoke100", "full"), required=True)
    assemble.add_argument("--candidate-paths", type=Path, action="append", required=True)
    assemble.add_argument("--acoustic-metrics", type=Path, required=True)
    assemble.add_argument("--speaker-metrics", type=Path, required=True)
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


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def clean_view_source(split: str) -> dict[str, dict[str, Any]]:
    if split == "dev":
        path = ROOT / "analysis/selected_evidence/all_dev_candidate_inputs.jsonl"
    else:
        path = ROOT / "analysis/selected_evidence_test/candidate_input_manifest.jsonl"
    rows = read_jsonl(path)
    result = {row["trial_id"]: row for row in rows}
    if len(rows) != 6000 or len(result) != 6000:
        raise ValueError(f"frozen clean enrollment-view source is incomplete: {split}")
    return result


def clean_cohorts(split: str) -> dict[str, str]:
    rows = read_jsonl(ROOT / f"analysis/wesep_speaker_selection/{split}_all_trials.jsonl")
    result = {
        row["trial_id"]: (
            "clean_defined_primary_swap"
            if bool(row["high_confidence_wrong"])
            else "clean_defined_primary_correct"
        )
        for row in rows
    }
    expected = 405 if split == "dev" else 398
    if len(result) != 6000 or Counter(result.values())["clean_defined_primary_swap"] != expected:
        raise ValueError(f"frozen clean cohort mismatch: {split}")
    return result


def normalize_trial(row: dict[str, Any], views: dict[str, dict[str, Any]], cohorts: dict[str, str]) -> dict[str, Any]:
    base_id = row.get("base_trial_id", row["trial_id"])
    source = views[base_id]
    mixture_id = base_id.split(":")[1]
    return {
        "trial_id": row["trial_id"],
        "base_trial_id": base_id,
        "split": row["split"],
        "benchmark": "controlled" if row.get("noisy_dataset") == "controlled_wham_lufs" else "natural",
        "cohort": cohorts[base_id],
        "mixture_id": mixture_id,
        "snr_db": row.get("construction_snr_db_evaluation_only"),
        "gender_cohort": row.get("gender_cohort"),
        "mixture_wav": row["mixture_wav"],
        "clean_mixture_wav": row["clean_mixture_wav"],
        "noise_wav": row["noise_wav"],
        "target_wav": row["target_wav"],
        "interferer_wav": row["interferer_wavs"][0],
        "target_speaker": row["target_speaker"],
        "interferer_speaker": row["interferer_speakers"][0],
        "enrollment_wav": row["enrollment_wav"],
        # DEV's older candidate-input artifact uses speaker_embedding_path;
        # TEST's frozen artifact names the same target enrollment embedding
        # explicitly. No embedding is recomputed here.
        "speaker_embedding_path": source.get(
            "speaker_embedding_path", source.get("target_enrollment_embedding_path")
        ),
        "target_enrollment_embedding_path": source.get(
            "speaker_embedding_path", source.get("target_enrollment_embedding_path")
        ),
        "enrollment_views": source["enrollment_views"],
    }


def add_counterpart_embeddings(rows: list[dict[str, Any]]) -> None:
    embeddings: dict[tuple[str, str], str] = {}
    for row in rows:
        embeddings[(row["mixture_id"], str(row["target_speaker"]))] = row[
            "target_enrollment_embedding_path"
        ]
    for row in rows:
        key = (row["mixture_id"], str(row["interferer_speaker"]))
        if key not in embeddings:
            raise ValueError(f"paired interferer embedding missing: {row['trial_id']}")
        row["interferer_enrollment_embedding_path"] = embeddings[key]


def choose(rows: list[dict[str, Any]], rng: random.Random, count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"cannot sample {count} from {len(rows)}")
    chosen_ids = set(rng.sample(sorted(row["trial_id"] for row in rows), count))
    return [row for row in rows if row["trial_id"] in chosen_ids]


def frozen_subsets(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(1986)
    natural = [row for row in rows if row["benchmark"] == "natural"]
    controlled = [row for row in rows if row["benchmark"] == "controlled"]
    by_nat = defaultdict(list)
    for row in natural:
        by_nat[row["cohort"]].append(row)
    by_ctl = defaultdict(list)
    for row in controlled:
        by_ctl[(int(row["snr_db"]), row["cohort"])].append(row)

    unit = []
    smoke = []
    for cohort in ("clean_defined_primary_swap", "clean_defined_primary_correct"):
        unit.extend(choose(by_nat[cohort], rng, 5))
        smoke.extend(choose(by_nat[cohort], rng, 25))
    for snr in (-5, 0, 5, 10, 15):
        for cohort in ("clean_defined_primary_swap", "clean_defined_primary_correct"):
            unit.extend(choose(by_ctl[(snr, cohort)], rng, 1))
            smoke.extend(choose(by_ctl[(snr, cohort)], rng, 5))
    unit_ids = {row["trial_id"] for row in unit}
    smoke_by_id = {row["trial_id"]: row for row in smoke}
    # Ensure the 20-trial audit is a subset of the 100-trial smoke while
    # preserving the registered cell counts. Replace within the same cell.
    for row in unit:
        if row["trial_id"] in smoke_by_id:
            continue
        cell = (row["benchmark"], row.get("snr_db"), row["cohort"])
        replaceable = sorted(
            value["trial_id"] for value in smoke_by_id.values()
            if (value["benchmark"], value.get("snr_db"), value["cohort"]) == cell
            and value["trial_id"] not in unit_ids
        )
        if not replaceable:
            raise ValueError(f"cannot nest unit set in smoke cell {cell}")
        del smoke_by_id[replaceable[-1]]
        smoke_by_id[row["trial_id"]] = row
    unit = sorted(unit, key=lambda row: row["trial_id"])
    smoke = sorted(smoke_by_id.values(), key=lambda row: row["trial_id"])
    if len(unit) != 20 or len(smoke) != 100 or not unit_ids <= set(smoke_by_id):
        raise ValueError("unit/smoke subset coverage failure")
    registration = {
        "status": "LOCKED_BEFORE_NOISY_MODEL_OUTPUT",
        "seed": 1986,
        "unit20_trial_ids": [row["trial_id"] for row in unit],
        "smoke100_trial_ids": [row["trial_id"] for row in smoke],
        "unit20_is_subset_of_smoke100": True,
        "selection": {
            "unit20": "natural 5 swap + 5 correct; controlled 1 swap + 1 correct per SNR",
            "smoke100": "natural 25 swap + 25 correct; controlled 5 swap + 5 correct per SNR",
        },
        "test_used": False,
    }
    return unit, smoke, registration


def prepare(split: str) -> int:
    views = clean_view_source(split)
    cohorts = clean_cohorts(split)
    natural = read_jsonl(MANIFESTS / f"natural_{split}.jsonl")
    controlled = read_jsonl(MANIFESTS / f"controlled_{split}.jsonl")
    rows = [normalize_trial(row, views, cohorts) for row in natural + controlled]
    add_counterpart_embeddings(rows)
    expected = 8400
    if len(rows) != expected or len({row["trial_id"] for row in rows}) != expected:
        raise ValueError(f"expected {expected} unique noisy {split} target trials")
    split_dir = ANALYSIS / split
    atomic_jsonl(split_dir / "candidate_inputs_full.jsonl", rows)
    if split == "dev":
        unit, smoke, registration = frozen_subsets(rows)
        atomic_jsonl(split_dir / "candidate_inputs_unit20.jsonl", unit)
        atomic_jsonl(split_dir / "candidate_inputs_smoke100.jsonl", smoke)
        registration_path = ANALYSIS / "preregistered_unit_smoke_subsets.json"
        atomic_json(registration_path, registration)
        atomic_text(
            registration_path.with_suffix(".sha256"),
            f"{sha256(registration_path)}  {registration_path.name}\n",
        )
    summary = {
        "status": "COMPLETE",
        "split": split,
        "trials": len(rows),
        "natural_trials": len(natural),
        "controlled_trials": len(controlled),
        "candidate_order": list(ORDER),
        "cohorts": dict(Counter(row["cohort"] for row in rows)),
        "noisy_test_model_results_used": False,
    }
    atomic_json(split_dir / "candidate_input_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def assemble(args: argparse.Namespace) -> int:
    split_dir = ANALYSIS / args.split
    inputs = read_jsonl(split_dir / f"candidate_inputs_{args.scope}.jsonl")
    expected = {"unit20": 20, "smoke100": 100, "full": 8400}[args.scope]
    if len(inputs) != expected:
        raise ValueError(f"scope {args.scope} expected {expected} rows")
    by_id = {row["trial_id"]: row for row in inputs}
    paths: dict[str, dict[str, str]] = {key: {} for key in by_id}
    for source in args.candidate_paths:
        for row in read_jsonl(source):
            if row["trial_id"] not in by_id:
                continue
            for name, path in row["candidate_paths"].items():
                if name in paths[row["trial_id"]]:
                    raise ValueError(f"duplicate candidate {name}: {row['trial_id']}")
                paths[row["trial_id"]][name] = path
    if any(tuple(value) != ORDER for value in paths.values()):
        raise ValueError("candidate order/coverage changed")
    acoustic = {row["trial_id"]: row for row in read_jsonl(args.acoustic_metrics)}
    speaker_rows = read_jsonl(args.speaker_metrics)
    speaker = {(row["trial_id"], row["candidate"]): row for row in speaker_rows}
    if set(acoustic) != set(by_id) or len(speaker) != expected * len(ORDER):
        raise ValueError("candidate metric coverage mismatch")

    deployment = []
    evaluation = []
    for source in inputs:
        trial_id = source["trial_id"]
        candidates = {}
        evaluation_candidates = {}
        for name in ORDER:
            s = speaker[(trial_id, name)]
            a = acoustic[trial_id]["candidates"][name]
            if Path(a["output_wav"]).resolve() != Path(paths[trial_id][name]).resolve():
                raise ValueError(f"candidate acoustic path mismatch: {trial_id} {name}")
            candidates[name] = {
                "waveform_path": paths[trial_id][name],
                "enrollment_cosine": float(s["speaker_similarity_to_enrollment"]),
            }
            evaluation_candidates[name] = dict(a) | {
                "speaker_similarity_to_enrollment": s["speaker_similarity_to_enrollment"],
                "speaker_similarity_to_interferer_enrollment": s[
                    "speaker_similarity_to_interferer_enrollment"
                ],
                "speaker_embedding_margin": s["speaker_embedding_margin"],
            }
        selected = {
            pool: max(
                names,
                key=lambda name: (candidates[name]["enrollment_cosine"], -ORDER.index(name)),
            )
            for pool, names in POOLS.items()
        }
        deployment.append({
            "trial_id": trial_id,
            "split": args.split,
            "mixture_wav": source["mixture_wav"],
            "enrollment_wav": source["enrollment_wav"],
            "speaker_embedding_path": source["speaker_embedding_path"],
            "candidates": candidates,
            "selected": selected,
        })
        evaluation.append(source | {
            "candidates": evaluation_candidates,
            "selected": selected,
            "pool_b_selected_waveform": candidates[selected["pool_b"]]["waveform_path"],
            "pool_d_selected_waveform": candidates[selected["pool_d"]]["waveform_path"],
            "pool_full_selected_waveform": candidates["full"]["waveform_path"],
        })
    output = split_dir / args.scope
    atomic_jsonl(output / "candidates_deployment.jsonl", deployment)
    atomic_jsonl(output / "candidates_evaluation.jsonl", evaluation)
    summary = {
        "status": "COMPLETE",
        "split": args.split,
        "scope": args.scope,
        "trials": len(deployment),
        "selection_counts": {
            pool: dict(Counter(row["selected"][pool] for row in deployment))
            for pool in POOLS
        },
        "selection_uses_evaluation_fields": False,
        "test_used": args.split == "test",
    }
    atomic_json(output / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args.split)
    return assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
