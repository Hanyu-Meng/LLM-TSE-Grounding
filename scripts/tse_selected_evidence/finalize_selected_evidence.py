#!/usr/bin/env python3
"""Validate completed selected-evidence caches and build frozen DEV manifests.

This step never runs an extractor or tokenizer.  It only accepts exact,
complete CDCS-2 and CDCS-5 caches and then materializes the leak-free inference
manifest plus the separate evaluation-only manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/selected_evidence"
CANDIDATES = ANALYSIS / "full_dev_candidates.jsonl"
PREPARED = ROOT / "manifests/tse_dev_prepared.jsonl"
SMOKE = ANALYSIS / "preregistered_smoke_subset.json"
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
POOLS = {
    "cdcs2": ("full", "tfmap_context_full"),
    "cdcs5": ORDER,
}
FORBIDDEN = (
    "target", "interferer", "transcript", "sisdr", "si_sdr", "qc",
    "reference", "label",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    readable = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in trial_id
    )
    return f"{readable[:100]}-{digest}"


def validate_cache(
    rows: list[dict[str, Any]], pool: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    token_dir = ANALYSIS / "evidence" / pool / "tokens"
    wav_dir = ANALYSIS / "evidence" / pool / "wav"
    expected_token_names = {f"{safe_name(row['trial_id'])}.npy" for row in rows}
    expected_wav_names = {f"{safe_name(row['trial_id'])}.wav" for row in rows}
    actual_token_names = {path.name for path in token_dir.glob("*.npy")}
    actual_wav_names = {path.name for path in wav_dir.glob("*.wav")}
    if actual_token_names != expected_token_names:
        raise ValueError(
            f"{pool} token filename mismatch missing="
            f"{len(expected_token_names - actual_token_names)} extra="
            f"{len(actual_token_names - expected_token_names)}"
        )
    if actual_wav_names != expected_wav_names:
        raise ValueError(
            f"{pool} WAV filename mismatch missing="
            f"{len(expected_wav_names - actual_wav_names)} extra="
            f"{len(actual_wav_names - expected_wav_names)}"
        )

    token_paths: dict[str, str] = {}
    wav_paths: dict[str, str] = {}
    token_lengths: list[int] = []
    for index, row in enumerate(rows, 1):
        trial_id = row["trial_id"]
        stem = safe_name(trial_id)
        token_path = token_dir / f"{stem}.npy"
        wav_path = wav_dir / f"{stem}.wav"
        mixture = sf.info(row["mixture_wav"])
        aligned = sf.info(wav_path)
        tokens = np.load(token_path, allow_pickle=False).reshape(-1)
        expected_tokens = math.ceil(int(mixture.frames) / 640)
        valid = (
            int(mixture.samplerate) == int(aligned.samplerate) == 16000
            and int(mixture.frames) == int(aligned.frames)
            and tokens.size == expected_tokens
            and tokens.size > 0
            and np.issubdtype(tokens.dtype, np.integer)
            and int(tokens.min()) >= 0
            and int(tokens.max()) < 6561
        )
        if not valid:
            raise ValueError(f"invalid {pool} cache at row {index}: {trial_id}")
        token_paths[trial_id] = str(token_path.resolve())
        wav_paths[trial_id] = str(wav_path.resolve())
        token_lengths.append(int(tokens.size))
    return token_paths, wav_paths, {
        "count": len(token_paths),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
    }


def main() -> int:
    candidates = read_jsonl(CANDIDATES)
    prepared_rows = read_jsonl(PREPARED)
    prepared = {row["trial_id"]: row for row in prepared_rows}
    ids = [row["trial_id"] for row in candidates]
    if len(candidates) != 6000 or len(set(ids)) != 6000:
        raise ValueError("candidate manifest is not exactly 6,000 unique trials")
    if len(prepared) != 6000 or set(prepared) != set(ids):
        raise ValueError("prepared DEV and selected candidates do not match exactly")
    cohorts = Counter(row["cohort"] for row in candidates)
    if cohorts != Counter({
        "natural_primary_swap": 405,
        "primary_correct_control": 5586,
        "ambiguous_primary_wrong": 9,
    }):
        raise ValueError(f"frozen cohort mismatch: {cohorts}")
    for row in candidates:
        if row.get("split") != "dev" or tuple(row["candidates"]) != ORDER:
            raise ValueError(f"candidate protocol mismatch: {row['trial_id']}")
        for pool, allowed in POOLS.items():
            if row["selected"][pool] not in allowed:
                raise ValueError(f"selection outside {pool}: {row['trial_id']}")

    cache: dict[str, tuple[dict[str, str], dict[str, str], dict[str, int]]] = {}
    for pool in POOLS:
        cache[pool] = validate_cache(candidates, pool)

    inference: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    for row in candidates:
        trial_id = row["trial_id"]
        inference_row = {
            "trial_id": trial_id,
            "split": "dev",
            "mixture_wav": row["mixture_wav"],
            "enrollment_wav": row["enrollment_wav"],
            "speaker_embedding_path": row["speaker_embedding_path"],
            "cdcs2_evidence_token_path": cache["cdcs2"][0][trial_id],
            "cdcs5_evidence_token_path": cache["cdcs5"][0][trial_id],
            "cdcs2_direct_candidate": row["selected"]["cdcs2"],
            "cdcs5_direct_candidate": row["selected"]["cdcs5"],
        }
        leaked = sorted(
            key for key in inference_row
            if any(part in key.lower() for part in FORBIDDEN)
        )
        if leaked:
            raise ValueError(f"inference manifest leaks evaluation fields: {leaked}")
        inference.append(inference_row)
        evaluation.append(dict(prepared[trial_id]) | {
            "cohort": row["cohort"],
            "cdcs2_direct_candidate": row["selected"]["cdcs2"],
            "cdcs5_direct_candidate": row["selected"]["cdcs5"],
            "cdcs2_direct_waveform": cache["cdcs2"][1][trial_id],
            "cdcs5_direct_waveform": cache["cdcs5"][1][trial_id],
        })
        for pool in POOLS:
            alignment.append({
                "trial_id": trial_id,
                "split": "dev",
                "pool": pool,
                "selected_candidate": row["selected"][pool],
                "aligned_waveform": cache[pool][1][trial_id],
                "evidence_token_path": cache[pool][0][trial_id],
                "mixture_num_samples": int(sf.info(row["mixture_wav"]).frames),
                "target_length_used": False,
                "test_used": False,
            })

    smoke_ids = json.loads(SMOKE.read_text(encoding="utf-8"))["ordered_trial_ids"]
    if len(smoke_ids) != 100 or len(set(smoke_ids)) != 100:
        raise ValueError("smoke subset is not exactly 100 unique trials")
    inference_by_id = {row["trial_id"]: row for row in inference}
    evaluation_by_id = {row["trial_id"]: row for row in evaluation}

    atomic_jsonl(ROOT / "manifests/selected_evidence_dev_inference.jsonl", inference)
    atomic_jsonl(ROOT / "manifests/selected_evidence_dev_evaluation.jsonl", evaluation)
    atomic_jsonl(
        ROOT / "manifests/selected_evidence_dev_smoke_inference.jsonl",
        [inference_by_id[trial_id] for trial_id in smoke_ids],
    )
    atomic_jsonl(
        ROOT / "manifests/selected_evidence_dev_smoke_evaluation.jsonl",
        [evaluation_by_id[trial_id] for trial_id in smoke_ids],
    )
    atomic_jsonl(ANALYSIS / "evidence_alignment.jsonl", alignment)
    for pool in POOLS:
        atomic_jsonl(
            ANALYSIS / "evidence" / pool / "token_records.jsonl",
            [{
                "trial_id": trial_id,
                "token_path": cache[pool][0][trial_id],
            } for trial_id in ids],
        )

    summary = {
        "status": "COMPLETE",
        "expected": 6000,
        "unique": len(set(ids)),
        "missing": 0,
        "duplicate": 0,
        "failed": 0,
        "cohorts": dict(cohorts),
        "candidate_order": list(ORDER),
        "pools": {pool: list(names) for pool, names in POOLS.items()},
        "selection": "frozen enrollment-only ECAPA cosine argmax",
        "cache_validation": {pool: cache[pool][2] for pool in POOLS},
        "inference_fields": sorted(inference[0]),
        "clean_reference_used_for_inference": False,
        "target_length_used": False,
        "test_used": False,
    }
    atomic_json(ANALYSIS / "evidence_preparation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
