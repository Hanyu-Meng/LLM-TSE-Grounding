#!/usr/bin/env python3
"""Validate the frozen 100-trial selected-evidence smoke and memory gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "analysis/selected_evidence/smoke"
SYSTEMS = (
    "qwen_tse_ud_cdcs2", "qwen_tse_fixed_csg_cdcs2",
    "qwen_tse_ud_cdcs5", "qwen_tse_fixed_csg_cdcs5",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_summary(path: Path, count_keys: tuple[str, str, str]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_key, complete_key, failed_key = count_keys
    if not (
        value.get("status") == "COMPLETE"
        and int(value.get(expected_key, -1)) == 100
        and int(value.get(complete_key, -1)) == 100
        and int(value.get(failed_key, -1)) == 0
    ):
        raise ValueError(f"incomplete smoke summary: {path}")
    return value


def memory_check(rows: list[dict[str, Any]], system: str, stage: str) -> dict[str, Any]:
    if len(rows) != 100 or len({row["trial_id"] for row in rows}) != 100:
        raise ValueError(f"{system} {stage} is not exactly 100 unique rows")
    output: dict[str, Any] = {"system": system, "stage": stage}
    limits = {
        "process_rss_bytes": 1024 ** 3,
        "cuda_reserved_bytes": 512 * 1024 ** 2,
    }
    passed = True
    for key, tolerance in limits.items():
        values = np.asarray([float(row[key]) for row in rows])
        first = float(np.mean(values[:20]))
        last = float(np.mean(values[-20:]))
        delta = last - first
        output[key] = {
            "first20_mean": first,
            "last20_mean": last,
            "last_minus_first": delta,
            "allowed_growth": tolerance,
            "pass": delta <= tolerance,
        }
        passed = passed and delta <= tolerance
    output["pass"] = passed
    if not passed:
        raise ValueError(f"smoke memory gate failed: {system} {stage}: {output}")
    return output


def main() -> int:
    checks = []
    for system in SYSTEMS:
        directory = BASE / system
        exact_summary(
            directory / "token_summary.json",
            ("expected_trials", "decoded_trials", "failed_trials"),
        )
        exact_summary(
            directory / "audio_summary.json",
            ("expected", "decoded", "failed"),
        )
        exact_summary(
            directory / "summary.json",
            ("expected", "decoded", "failed"),
        )
        token_rows = read_jsonl(directory / "per_trial_tokens.jsonl")
        audio_rows = read_jsonl(directory / "audio_metrics.jsonl")
        metric_rows = read_jsonl(directory / "per_trial_metrics.jsonl")
        if len(metric_rows) != 100 or len({row["trial_id"] for row in metric_rows}) != 100:
            raise ValueError(f"{system} final metrics are not exactly 100 unique rows")
        checks.append(memory_check(token_rows, system, "qwen_tse_decode"))
        checks.append(memory_check(audio_rows, system, "cosyvoice_audio"))
    result = {
        "status": "PASS",
        "systems": list(SYSTEMS),
        "expected_per_system": 100,
        "memory_checks": checks,
        "full_dev_authorized": True,
        "test_used": False,
    }
    destination = BASE / "smoke_validation.json"
    temporary = destination.with_suffix(".tmp.json")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
