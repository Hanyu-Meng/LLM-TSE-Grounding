#!/usr/bin/env python3
"""Build the frozen TEST candidate, selection, and evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/selected_evidence_test"
PREPARED = ROOT / "manifests/tse_test_prepared.jsonl"
AUDIT = ROOT / "analysis/wesep_speaker_selection/test_all_trials.jsonl"
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
POOLS = {
    "primary": ("full",),
    "cdcs2": ("full", "tfmap_context_full"),
    "cdcs5": ORDER,
}
EXPECTED_COHORTS = Counter({
    "natural_primary_swap": 398,
    "primary_correct_control": 5588,
    "ambiguous_primary_wrong": 14,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--acoustic-metrics", type=Path, required=True)
    assemble.add_argument("--speaker-metrics", type=Path, required=True)
    assemble.add_argument("--asr-metrics", type=Path)
    sub.add_parser("finalize")
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2) + "\n")


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in trial_id)
    return f"{safe[:100]}-{digest}"


def cohort(row: dict[str, Any]) -> str:
    if bool(row["high_confidence_wrong"]):
        return "natural_primary_swap"
    if bool(row["wrong_margin_0"]):
        return "ambiguous_primary_wrong"
    return "primary_correct_control"


def prepare() -> int:
    prepared_rows = read_jsonl(PREPARED)
    audit_rows = read_jsonl(AUDIT)
    if len(prepared_rows) != 6000 or len(audit_rows) != 6000:
        raise ValueError("TEST sources must each contain exactly 6,000 rows")
    if any(row.get("split") != "test" for row in prepared_rows + audit_rows):
        raise ValueError("non-TEST row rejected")
    prepared = {row["trial_id"]: row for row in prepared_rows}
    audit = {row["trial_id"]: row for row in audit_rows}
    if len(prepared) != 6000 or set(prepared) != set(audit):
        raise ValueError("TEST prepared/audit ID mismatch")

    gate_scripts = ROOT / "scripts/tse_candidate_gate"
    sys.path.insert(0, str(gate_scripts))
    from build_candidate_gate_inputs import materialize_views  # noqa: PLC0415

    by_mixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared_rows:
        by_mixture[Path(row["mixture_wav"]).stem].append(row)
    output: list[dict[str, Any]] = []
    view_index: list[dict[str, Any]] = []
    for index, row in enumerate(prepared_rows, 1):
        trial_id = row["trial_id"]
        label = cohort(audit[trial_id])
        details = materialize_views(
            row["enrollment_wav"], ANALYSIS / "enrollment_views",
            segment_seconds=2.0, grid_seconds=0.01,
        )
        mixture_id = Path(row["mixture_wav"]).stem
        peers = [item for item in by_mixture[mixture_id] if item["trial_id"] != trial_id]
        if len(peers) != 1:
            raise ValueError(f"expected one paired target view: {trial_id}")
        counterpart = peers[0]
        primary = Path(row["evidence_wav"])
        if not primary.is_file():
            raise FileNotFoundError(primary)
        item = {
            "trial_id": trial_id,
            "split": "test",
            "cohort": label,
            "mixture_id": mixture_id,
            "mixture_wav": str(row["mixture_wav"]),
            "target_wav": str(row["target_wav"]),
            "interferer_wav": str(row["interferer_wavs"][0]),
            "target_speaker": str(row["target_speaker"]),
            "interferer_speaker": str(row["interferer_speakers"][0]),
            "enrollment_wav": str(row["enrollment_wav"]),
            "target_enrollment_embedding_path": str(row["speaker_embedding_path"]),
            "interferer_enrollment_embedding_path": str(counterpart["speaker_embedding_path"]),
            "enrollment_views": {
                key: str(value) for key, value in details["view_paths"].items()
            },
            "enrollment_view_start_seconds": details["start_seconds"],
            "enrollment_short_fallback": details["short_utterance_fallback"],
            "primary_output_wav": str(primary.resolve()),
            "frozen_primary_high_confidence_wrong": bool(audit[trial_id]["high_confidence_wrong"]),
            "frozen_primary_margin_db": float(audit[trial_id]["speaker_selection_margin"]),
        }
        output.append(item)
        view_index.append({
            "trial_id": trial_id,
            "split": "test",
            "enrollment_wav": row["enrollment_wav"],
            "view_paths": item["enrollment_views"],
            "start_seconds": details["start_seconds"],
            "short_fallback": details["short_utterance_fallback"],
        })
        if index == 1 or index % 500 == 0:
            print(f"prepared={index}/6000", flush=True)
    counts = Counter(row["cohort"] for row in output)
    if counts != EXPECTED_COHORTS or len({row["trial_id"] for row in output}) != 6000:
        raise ValueError(f"frozen TEST cohort mismatch: {counts}")
    write_jsonl(ANALYSIS / "candidate_input_manifest.jsonl", output)
    write_jsonl(ANALYSIS / "enrollment_view_index.jsonl", view_index)

    # Fixed solely from trial IDs/cohorts before any selected-evidence result is read.
    rng = random.Random(1986)
    by_cohort: dict[str, list[str]] = defaultdict(list)
    for row in output:
        by_cohort[row["cohort"]].append(row["trial_id"])
    swaps = rng.sample(sorted(by_cohort["natural_primary_swap"]), 25)
    controls = rng.sample(sorted(by_cohort["primary_correct_control"]), 25)
    ambiguous = sorted(by_cohort["ambiguous_primary_wrong"])
    used = set(swaps + controls + ambiguous)
    remaining = sorted(set(prepared) - used)
    random_ids = rng.sample(remaining, 100 - len(used))
    smoke_ids = swaps + controls + ambiguous + random_ids
    smoke = {
        "status": "LOCKED_BEFORE_SELECTED_EVIDENCE_TEST_RESULTS",
        "seed": 1986,
        "selection_basis": "frozen primary cohort labels and TEST trial IDs only",
        "components": {
            "natural_primary_swap": swaps,
            "primary_correct_control": controls,
            "ambiguous_primary_wrong": ambiguous,
            "deterministic_random_from_remaining": random_ids,
        },
        "ordered_trial_ids": smoke_ids,
        "count": 100,
        "test_used": True,
    }
    write_json(ANALYSIS / "preregistered_smoke_subset.json", smoke)
    by_id = {row["trial_id"]: row for row in output}
    write_jsonl(ANALYSIS / "smoke_candidate_inputs.jsonl", [by_id[value] for value in smoke_ids])
    summary = {
        "status": "COMPLETE", "split": "test", "trials": 6000,
        "cohorts": dict(counts), "segment_seconds": 2.0, "grid_seconds": 0.01,
        "test_used": True, "clean_reference_used_to_construct_candidates": False,
    }
    write_json(ANALYSIS / "candidate_input_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def assemble(args: argparse.Namespace) -> int:
    inputs = read_jsonl(ANALYSIS / "candidate_input_manifest.jsonl")
    acoustic_rows = read_jsonl(args.acoustic_metrics)
    speaker_rows = read_jsonl(args.speaker_metrics)
    if len(inputs) != 6000 or len(acoustic_rows) != 6000 or len(speaker_rows) != 30000:
        raise ValueError("candidate assembly coverage mismatch")
    input_by_id = {row["trial_id"]: row for row in inputs}
    acoustic_by_id = {row["trial_id"]: row for row in acoustic_rows}
    speaker = {(row["trial_id"], row["candidate"]): row for row in speaker_rows}
    asr: dict[tuple[str, str], dict[str, Any]] = {}
    if args.asr_metrics:
        asr_rows = read_jsonl(args.asr_metrics)
        if len(asr_rows) != 30000:
            raise ValueError("candidate ASR must contain 30,000 rows")
        asr = {(row["trial_id"], row["candidate"]): row for row in asr_rows}
    if not (set(input_by_id) == set(acoustic_by_id)):
        raise ValueError("candidate acoustic IDs mismatch")

    deployment: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    for source in inputs:
        trial_id = source["trial_id"]
        acoustic = acoustic_by_id[trial_id]
        if tuple(acoustic["candidates"]) != ORDER:
            raise ValueError(f"candidate order changed: {trial_id}")
        candidates: dict[str, dict[str, Any]] = {}
        full_metrics: dict[str, dict[str, Any]] = {}
        for name in ORDER:
            a = dict(acoustic["candidates"][name])
            s = speaker[(trial_id, name)]
            candidates[name] = {
                "waveform_path": a["output_wav"],
                "enrollment_cosine": float(s["speaker_similarity_to_enrollment"]),
            }
            a.update({
                "candidate_embedding_path": s["candidate_embedding_path"],
                "speaker_similarity_to_enrollment": s["speaker_similarity_to_enrollment"],
                "speaker_similarity_to_interferer_enrollment": s["speaker_similarity_to_interferer_enrollment"],
                "speaker_embedding_margin": s["speaker_embedding_margin"],
            })
            if asr:
                c = asr[(trial_id, name)]
                a.update({key: c[key] for key in (
                    "target_text", "interferer_text", "output_text", "target_WER",
                    "target_word_distance", "interferer_WER", "interferer_word_distance",
                    "content_switch",
                )})
            full_metrics[name] = a
        selected = {
            pool: max(
                names,
                key=lambda name: (candidates[name]["enrollment_cosine"], -ORDER.index(name)),
            )
            for pool, names in POOLS.items()
        }
        deployment.append({
            "trial_id": trial_id, "split": "test", "cohort": source["cohort"],
            "mixture_wav": source["mixture_wav"],
            "enrollment_wav": source["enrollment_wav"],
            "speaker_embedding_path": source["target_enrollment_embedding_path"],
            "candidates": candidates, "selected": selected,
        })
        diagnostic.append({
            "trial_id": trial_id, "split": "test", "cohort": source["cohort"],
            "target_speaker": source["target_speaker"],
            "interferer_speaker": source["interferer_speaker"],
            "target_wav": source["target_wav"],
            "interferer_wav": source["interferer_wav"],
            "target_enrollment_embedding_path": source["target_enrollment_embedding_path"],
            "interferer_enrollment_embedding_path": source["interferer_enrollment_embedding_path"],
            "candidates": full_metrics, "selected": selected,
        })
    write_jsonl(ANALYSIS / "full_test_candidates.jsonl", deployment)
    write_jsonl(ANALYSIS / "full_test_candidate_metrics.jsonl", diagnostic)
    smoke_ids = read_json(ANALYSIS / "preregistered_smoke_subset.json")["ordered_trial_ids"]
    by_id = {row["trial_id"]: row for row in deployment}
    write_jsonl(ANALYSIS / "smoke_candidates.jsonl", [by_id[value] for value in smoke_ids])
    summary = {
        "status": "COMPLETE", "split": "test", "trials": 6000,
        "candidate_order": list(ORDER), "pools": {key: list(value) for key, value in POOLS.items()},
        "selection": "frozen enrollment-only ECAPA cosine argmax",
        "selection_counts": {
            pool: dict(Counter(row["selected"][pool] for row in deployment)) for pool in POOLS
        },
        "clean_reference_used_for_selection": False, "test_used": True,
    }
    write_json(ANALYSIS / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_token_cache(pool: str, rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    token_dir = ANALYSIS / "evidence" / pool / "tokens"
    wav_dir = ANALYSIS / "evidence" / pool / "wav"
    token_paths: dict[str, str] = {}
    wav_paths: dict[str, str] = {}
    for row in rows:
        stem = safe_name(row["trial_id"])
        token = token_dir / f"{stem}.npy"
        wav = wav_dir / f"{stem}.wav"
        mixture = sf.info(row["mixture_wav"])
        aligned = sf.info(wav)
        values = np.load(token, allow_pickle=False).reshape(-1)
        expected = math.ceil(int(mixture.frames) / 640)
        if not (
            int(mixture.samplerate) == int(aligned.samplerate) == 16000
            and int(mixture.frames) == int(aligned.frames)
            and values.size == expected and values.size > 0
            and np.issubdtype(values.dtype, np.integer)
            and int(values.min()) >= 0 and int(values.max()) < 6561
        ):
            raise ValueError(f"invalid {pool} cache: {row['trial_id']}")
        token_paths[row["trial_id"]] = str(token.resolve())
        wav_paths[row["trial_id"]] = str(wav.resolve())
    return token_paths, wav_paths


def finalize() -> int:
    rows = read_jsonl(ANALYSIS / "full_test_candidates.jsonl")
    prepared_rows = read_jsonl(PREPARED)
    prepared = {row["trial_id"]: row for row in prepared_rows}
    ids = [row["trial_id"] for row in rows]
    if len(rows) != 6000 or len(set(ids)) != 6000 or set(ids) != set(prepared):
        raise ValueError("TEST selection/prepared coverage mismatch")
    caches = {pool: validate_token_cache(pool, rows) for pool in POOLS}
    inference: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    for row in rows:
        trial_id = row["trial_id"]
        inference.append({
            "trial_id": trial_id, "split": "test",
            "mixture_wav": row["mixture_wav"],
            "enrollment_wav": row["enrollment_wav"],
            "speaker_embedding_path": row["speaker_embedding_path"],
            **{f"{pool}_evidence_token_path": caches[pool][0][trial_id] for pool in POOLS},
            **{f"{pool}_selected_candidate": row["selected"][pool] for pool in POOLS},
        })
        evaluation.append(dict(prepared[trial_id]) | {
            "cohort": row["cohort"],
            "cdcs2_direct_candidate": row["selected"]["cdcs2"],
            "cdcs5_direct_candidate": row["selected"]["cdcs5"],
            "cdcs2_direct_waveform": caches["cdcs2"][1][trial_id],
            "cdcs5_direct_waveform": caches["cdcs5"][1][trial_id],
        })
        for pool in POOLS:
            alignment.append({
                "trial_id": trial_id, "split": "test", "pool": pool,
                "selected_candidate": row["selected"][pool],
                "aligned_waveform": caches[pool][1][trial_id],
                "evidence_token_path": caches[pool][0][trial_id],
                "mixture_num_samples": int(sf.info(row["mixture_wav"]).frames),
                "target_length_used": False, "test_used": True,
            })
    smoke_ids = read_json(ANALYSIS / "preregistered_smoke_subset.json")["ordered_trial_ids"]
    infer_by_id = {row["trial_id"]: row for row in inference}
    eval_by_id = {row["trial_id"]: row for row in evaluation}
    write_jsonl(ROOT / "manifests/selected_evidence_test_inference.jsonl", inference)
    write_jsonl(ROOT / "manifests/selected_evidence_test_evaluation.jsonl", evaluation)
    write_jsonl(ROOT / "manifests/selected_evidence_test_smoke_inference.jsonl", [infer_by_id[value] for value in smoke_ids])
    write_jsonl(ROOT / "manifests/selected_evidence_test_smoke_evaluation.jsonl", [eval_by_id[value] for value in smoke_ids])
    write_jsonl(ANALYSIS / "evidence_alignment.jsonl", alignment)
    for pool in POOLS:
        write_jsonl(ANALYSIS / "evidence" / pool / "token_records.jsonl", [
            {"trial_id": trial_id, "token_path": caches[pool][0][trial_id]}
            for trial_id in ids
        ])
    summary = {
        "status": "COMPLETE", "split": "test", "expected": 6000,
        "unique": 6000, "missing": 0, "duplicate": 0, "failed": 0,
        "cohorts": dict(Counter(row["cohort"] for row in rows)),
        "candidate_order": list(ORDER), "pools": {key: list(value) for key, value in POOLS.items()},
        "selection": "frozen enrollment-only ECAPA cosine argmax",
        "clean_reference_used_for_inference": False,
        "target_length_used": False, "test_used": True,
    }
    write_json(ANALYSIS / "evidence_preparation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "assemble":
        return assemble(args)
    if args.command == "finalize":
        return finalize()
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
