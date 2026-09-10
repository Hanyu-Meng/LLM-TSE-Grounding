#!/usr/bin/env python3
"""Select K/R on natural DEV under the pre-locked quality/content gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--k20-metrics", type=Path, required=True)
    parser.add_argument("--k50-metrics", type=Path, required=True)
    parser.add_argument("--k20-utmos", type=Path, required=True)
    parser.add_argument("--k50-utmos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def keyed(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    result = {row["trial_id"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"duplicate IDs: {path}")
    return result


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def summarize(metrics: dict[str, dict], utmos: dict[str, dict], ids: list[str]) -> dict:
    wers = np.asarray([float(metrics[x]["target_WER"]) for x in ids])
    return {
        "trials": len(ids),
        "raw_wer": float(np.mean(wers)),
        "content_switch_rate": float(np.mean([bool(metrics[x]["content_switch"]) for x in ids])),
        "dnsmos_p808": float(np.mean([float(metrics[x]["dnsmos_p808"]) for x in ids])),
        "utmos": float(np.mean([float(utmos[x]["utmos"]) for x in ids])),
    }


def main() -> int:
    args = parse_args()
    evaluation = keyed(args.evaluation)
    ids = sorted(x for x, row in evaluation.items() if row.get("benchmark") == "natural")
    if len(ids) != 6000:
        raise ValueError("expected 6000 natural DEV trials")
    k20m, k50m = keyed(args.k20_metrics), keyed(args.k50_metrics)
    k20u, k50u = keyed(args.k20_utmos), keyed(args.k50_utmos)
    if any(set(source) != set(evaluation) for source in (k20m, k50m, k20u, k50u)):
        raise ValueError("GNR DEV coverage mismatch")
    a = summarize(k20m, k20u, ids)
    b = summarize(k50m, k50u, ids)
    quality_support = (
        b["dnsmos_p808"] >= a["dnsmos_p808"] + 0.005
        or b["utmos"] >= a["utmos"] + 0.005
    )
    choose_b = (
        quality_support
        and b["raw_wer"] <= a["raw_wer"] + 0.005
        and b["content_switch_rate"] <= a["content_switch_rate"] + 0.0025
    )
    result = {
        "status": "FROZEN_ON_DEV",
        "gnr_a_k20_r2": a,
        "gnr_b_k50_r3": b,
        "selected": "GNR-B" if choose_b else "GNR-A",
        "K": 50 if choose_b else 20,
        "R": 3 if choose_b else 2,
        "selected_dev_slug": (
            "qwen_tse_adaptive_csg_gnr_cdcs5_k50_r3" if choose_b
            else "qwen_tse_adaptive_csg_gnr_cdcs5_k20_r2"
        ),
        "selection_rule": (
            "choose GNR-B if P808 or UTMOS improves >=0.005, raw WER rise "
            "<=0.005, and content-switch rise <=0.0025; otherwise GNR-A"
        ),
        "test_used": False,
    }
    atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
