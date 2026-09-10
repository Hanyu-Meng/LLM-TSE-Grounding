#!/usr/bin/env python3
"""Freeze the best-generative comparison arm on natural noisy DEV."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ("G2", "G3", "G5", "G6")


def main() -> int:
    source = ROOT / "results/noisy_wham/dev/compiled/all_system_summaries.json"
    output = ROOT / "analysis/noisy_wham/dev/full/best_generative_selection.json"
    value = json.loads(source.read_text())
    systems = value["systems"]
    ranked = []
    for order, code in enumerate(CANDIDATES):
        row = systems[code]["natural"]
        objective = (
            float(row["target_WER_raw"])
            + 0.5 * float(row["content_switch_rate"])
            + 0.25 * float(row["high_error_rate"])
        )
        ranked.append({
            "system_code": code,
            "system_slug": systems[code]["system_slug"],
            "system_name": systems[code]["system_name"],
            "reliability_objective": objective,
            "raw_wer": row["target_WER_raw"],
            "content_switch_rate": row["content_switch_rate"],
            "high_error_rate": row["high_error_rate"],
            "dnsmos_p808": row["dnsmos_p808"],
            "utmos": row["utmos"],
            "fixed_order": order,
        })
    ranked.sort(key=lambda row: (
        row["reliability_objective"],
        -float(row["dnsmos_p808"]),
        -float(row["utmos"]),
        row["fixed_order"],
    ))
    result = {
        "status": "FROZEN_ON_NATURAL_NOISY_DEV",
        "selected_code": ranked[0]["system_code"],
        "selected_slug": ranked[0]["system_slug"],
        "selected_name": ranked[0]["system_name"],
        "ranking": ranked,
        "rule": (
            "minimum mean(raw WER + 0.5*content-switch + 0.25*high-error); "
            "tie higher P808, higher UTMOS, fixed order"
        ),
        "test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
