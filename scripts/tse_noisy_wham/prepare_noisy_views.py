#!/usr/bin/env python3
"""Create exact controlled/natural DEV or TEST views keyed only by benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--token-records", type=Path)
    parser.add_argument("--token-output", type=Path)
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def atomic(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    prefix = ROOT / f"manifests/noisy_wham/{args.split}_full"
    evaluation_path = prefix.with_name(prefix.name + "_evaluation.jsonl")
    evaluation = rows(evaluation_path)
    benchmark = {row["trial_id"]: row["benchmark"] for row in evaluation}
    if len(evaluation) != 8400 or len(benchmark) != 8400:
        raise ValueError("full evaluation coverage failure")
    if args.token_records:
        if args.token_output is None:
            raise ValueError("--token-output is required with --token-records")
        source = rows(args.token_records)
        selected = [row for row in source if benchmark.get(row["trial_id"]) == "controlled"]
        if len(selected) != 2400 or len({row["trial_id"] for row in selected}) != 2400:
            raise ValueError("controlled token-record view coverage failure")
        atomic(args.token_output, selected)
        print(json.dumps({"status": "COMPLETE", "rows": 2400,
                          "output": str(args.token_output)}, indent=2))
        return 0
    analysis = ROOT / f"analysis/noisy_wham/{args.split}/full"
    qc_path = analysis / "qc_metrics.jsonl"
    sources = {
        "inference": prefix.with_name(prefix.name + "_inference.jsonl"),
        "evaluation": evaluation_path,
        "qc": qc_path,
    }
    counts = {}
    for name, path in sources.items():
        source = rows(path)
        if len(source) != 8400 or len({row["trial_id"] for row in source}) != 8400:
            raise ValueError(f"full {name} coverage failure")
        for cohort, expected in (("controlled", 2400), ("natural", 6000)):
            selected = [row for row in source if benchmark.get(row["trial_id"]) == cohort]
            if len(selected) != expected:
                raise ValueError(f"{name}/{cohort} expected {expected}, got {len(selected)}")
            output = analysis / "views" / f"{cohort}_{name}.jsonl"
            atomic(output, selected)
            counts[f"{cohort}_{name}"] = len(selected)
    print(json.dumps({"status": "COMPLETE", "split": args.split,
                      "counts": counts, "test_used": args.split == "test"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
