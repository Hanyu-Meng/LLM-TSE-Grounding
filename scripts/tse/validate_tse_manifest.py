#!/usr/bin/env python3
"""Strictly validate target-speaker trial manifests."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from se_align.data.store import read_manifest


REQUIRED = (
    "trial_id",
    "split",
    "mixture_wav",
    "enrollment_wav",
    "target_wav",
    "interferer_wavs",
    "target_speaker",
    "interferer_speakers",
    "target_utterance",
    "enrollment_utterance",
    "enrollment_source",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-trials", type=int)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.expected_trials is not None and len(rows) != args.expected_trials:
        raise ValueError(f"expected {args.expected_trials} rows, found {len(rows)}")

    trial_ids: set[str] = set()
    mixtures: dict[str, list[dict]] = defaultdict(list)
    sources: Counter[str] = Counter()
    for index, row in enumerate(rows):
        absent = [key for key in REQUIRED if key not in row]
        if absent:
            raise ValueError(f"row {index} missing fields: {absent}")
        if row["trial_id"] in trial_ids:
            raise ValueError(f"duplicate trial_id: {row['trial_id']}")
        trial_ids.add(row["trial_id"])
        for field in ("mixture_wav", "enrollment_wav", "target_wav"):
            path = Path(row[field])
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        for value in row["interferer_wavs"]:
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        target_speaker = row["target_utterance"].split("-", 1)[0]
        enrollment_speaker = row["enrollment_utterance"].split("-", 1)[0]
        if target_speaker != row["target_speaker"] or enrollment_speaker != target_speaker:
            raise ValueError(f"speaker mismatch: {row['trial_id']}")
        if row["enrollment_utterance"] == row["target_utterance"]:
            raise ValueError(f"enrollment leakage: {row['trial_id']}")
        mixture_id = Path(row["mixture_wav"]).stem
        mixtures[mixture_id].append(row)
        sources[row["enrollment_source"]] += 1

    for mixture_id, trials in mixtures.items():
        targets = {row["target_utterance"] for row in trials}
        if len(trials) != 2 or len(targets) != 2:
            raise ValueError(
                f"{mixture_id}: expected two distinct target trials, got {len(trials)}"
            )
    print(
        f"VALID manifest={args.manifest} trials={len(rows)} "
        f"mixtures={len(mixtures)} enrollment_sources={dict(sources)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
