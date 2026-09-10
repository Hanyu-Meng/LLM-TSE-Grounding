#!/usr/bin/env python3
"""Validate noisy S3 evidence caches and build leak-free decode manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/noisy_wham"
MANIFESTS = ROOT / "manifests/noisy_wham"
POOLS = ("primary", "cdcs2", "cdcs5")
FORBIDDEN = ("target", "interferer", "transcript", "sisdr", "si_sdr", "qc", "reference", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--scope", choices=("unit20", "smoke100", "full"), required=True)
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


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2) + "\n")


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    readable = "".join(value if value.isalnum() or value in "-_" else "_" for value in trial_id)
    return f"{readable[:100]}-{digest}"


def validate_pool(root: Path, rows: list[dict[str, Any]], pool: str) -> tuple[dict[str, str], dict[str, str]]:
    token_dir = root / "evidence" / pool / "tokens"
    wav_dir = root / "evidence" / pool / "wav"
    tokens = {}
    waves = {}
    for row in rows:
        trial_id = row["trial_id"]
        stem = safe_name(trial_id)
        token_path = token_dir / f"{stem}.npy"
        wav_path = wav_dir / f"{stem}.wav"
        if not token_path.is_file() or not wav_path.is_file():
            raise FileNotFoundError(f"missing {pool} evidence: {trial_id}")
        mixture = sf.info(row["mixture_wav"])
        aligned = sf.info(wav_path)
        values = np.load(token_path, allow_pickle=False).reshape(-1)
        expected = math.ceil(int(mixture.frames) / 640)
        if not (
            int(mixture.samplerate) == int(aligned.samplerate) == 16000
            and int(mixture.frames) == int(aligned.frames)
            and values.size == expected
            and values.size > 0
            and np.issubdtype(values.dtype, np.integer)
            and int(values.min()) >= 0
            and int(values.max()) < 6561
        ):
            raise ValueError(f"invalid {pool} evidence cache: {trial_id}")
        tokens[trial_id] = str(token_path.resolve())
        waves[trial_id] = str(wav_path.resolve())
    return tokens, waves


def main() -> int:
    args = parse_args()
    scope_dir = ANALYSIS / args.split / args.scope
    deployment = read_jsonl(scope_dir / "candidates_deployment.jsonl")
    evaluation = read_jsonl(scope_dir / "candidates_evaluation.jsonl")
    expected = {"unit20": 20, "smoke100": 100, "full": 8400}[args.scope]
    deployment_by_id = {row["trial_id"]: row for row in deployment}
    evaluation_by_id = {row["trial_id"]: row for row in evaluation}
    if (
        len(deployment) != expected
        or len(deployment_by_id) != expected
        or set(deployment_by_id) != set(evaluation_by_id)
    ):
        raise ValueError("candidate deployment/evaluation coverage mismatch")
    # Unit, smoke, and full runs share one resume-safe cache so validated early
    # trials are never recomputed. The scope still controls which rows enter the
    # manifests and audits.
    cache_root = ANALYSIS / args.split / "full"
    caches = {pool: validate_pool(cache_root, deployment, pool) for pool in POOLS}

    # The requested scientific order is controlled DEV/TEST before natural.
    # Reordering here is inference-safe: it uses only the public benchmark tag,
    # never a target/reference metric, and all caches remain keyed by trial_id.
    original_index = {row["trial_id"]: index for index, row in enumerate(deployment)}
    deployment = sorted(
        deployment,
        key=lambda row: (
            0 if evaluation_by_id[row["trial_id"]].get("benchmark") == "controlled" else 1,
            original_index[row["trial_id"]],
        ),
    )
    inference_rows = []
    evaluation_rows = []
    for row in deployment:
        trial_id = row["trial_id"]
        inference = {
            "trial_id": trial_id,
            "split": args.split,
            "mixture_wav": row["mixture_wav"],
            "enrollment_wav": row["enrollment_wav"],
            "speaker_embedding_path": row["speaker_embedding_path"],
        }
        for pool in POOLS:
            inference[f"{pool}_evidence_token_path"] = caches[pool][0][trial_id]
            inference[f"{pool}_evidence_waveform"] = caches[pool][1][trial_id]
            inference[f"{pool}_selected_candidate"] = row["selected"][pool]
        leaked = [
            key for key in inference
            if any(value in key.lower() for value in FORBIDDEN)
        ]
        if leaked:
            raise ValueError(f"inference manifest leakage: {leaked}")
        inference_rows.append(inference)
        evaluation_rows.append(evaluation_by_id[trial_id] | {
            "tfmap_context_waveform": evaluation_by_id[trial_id]["candidates"][
                "tfmap_context_full"
            ]["output_wav"],
        } | {
            f"{pool}_aligned_evidence_waveform": caches[pool][1][trial_id]
            for pool in POOLS
        })

    prefix = f"{args.split}_{args.scope}"
    inference_path = MANIFESTS / f"{prefix}_inference.jsonl"
    evaluation_path = MANIFESTS / f"{prefix}_evaluation.jsonl"
    atomic_jsonl(inference_path, inference_rows)
    atomic_jsonl(evaluation_path, evaluation_rows)
    for pool in POOLS:
        atomic_jsonl(scope_dir / "evidence" / pool / "token_records.jsonl", [
            {"trial_id": row["trial_id"], "token_path": caches[pool][0][row["trial_id"]]}
            for row in deployment
        ])
    summary = {
        "status": "COMPLETE",
        "split": args.split,
        "scope": args.scope,
        "trials": expected,
        "pools": list(POOLS),
        "inference_fields": sorted(inference_rows[0]),
        "clean_reference_used_for_inference": False,
        "target_length_used": False,
        "execution_order": "controlled_then_natural",
        "test_used": args.split == "test",
    }
    atomic_json(scope_dir / "evidence_preparation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
