#!/usr/bin/env python3
"""Final cross-artifact QA for the frozen noisy TSE study."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/noisy_tse"
CODES = ("D0", "D3", "G2", "G3", "G5", "G6")
PRIMARY_METRICS = (
    "target_WER_raw", "speaker_margin", "lps", "speechbertscore",
    "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovl", "utmos",
)


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    issues = []
    checks = []
    for split in ("dev", "test"):
        compiled = json.loads((
            ROOT / f"results/noisy_wham/{split}/compiled/all_system_summaries.json"
        ).read_text())
        if compiled.get("status") != "COMPLETE":
            issues.append(f"{split} compiled status is not COMPLETE")
        systems = compiled["systems"]
        for code in CODES:
            if code not in systems:
                issues.append(f"{split} missing {code}")
                continue
            if systems[code]["all"]["trials"] != 8400:
                issues.append(f"{split}/{code} does not have 8400 trials")
            for metric in PRIMARY_METRICS:
                coverage = systems[code]["natural"].get(f"{metric}_coverage")
                if coverage is None or not math.isfinite(float(coverage)) or coverage < 0.99:
                    issues.append(f"{split}/{code}/{metric} coverage={coverage}")
        checks.append(f"{split}: six systems × 8400 and primary coverage >=99%")
    csv_expected = {
        "NOISY_MAIN_TABLE.csv": 12,
        "CONTROLLED_SNR_TABLE.csv": 50,
        "TAIL_ROBUSTNESS_TABLE.csv": 8,
        "CONFUSION_REPAIR_TABLE.csv": 12,
    }
    for name, expected in csv_expected.items():
        path = OUT / name
        count = len(rows(path)) if path.is_file() else -1
        if count != expected:
            issues.append(f"{name}: expected {expected} rows, got {count}")
        checks.append(f"{name}: {count} rows")
    required = [
        *(ROOT / "docs" / name for name in (
            "NOISY_TSE_MAIN_REPORT.md", "NOISY_CDCS5_ANALYSIS.md",
            "NOISY_GROUNDING_ANALYSIS.md", "GNR_LLM_TSE_ANALYSIS.md",
            "NOISY_TSE_COAUTHOR_SUMMARY.md", "GNR_TSE_IMPLEMENTATION_AUDIT.md",
        )),
        *(OUT / name for name in (
            "noisy_main_table.tex", "controlled_snr_table.tex", "tail_table.tex",
            "noisy_results_for_paper.tex", "scientific_verdict.json",
            "artifact.json", "report.html", "report_delivery.json",
        )),
    ]
    figure_stems = (
        "confusion_robustness_vs_snr_dev", "confusion_robustness_vs_snr_test",
        "content_wer_vs_snr", "perceptual_quality_p808", "perceptual_quality_utmos",
        "perceptual_quality_bak", "reliability_quality_tradeoff",
        "tail_robustness", "adaptive_grounding_behavior",
    )
    required.extend(
        ROOT / "figures/noisy_tse" / f"{stem}.{suffix}"
        for stem in figure_stems for suffix in ("pdf", "png")
    )
    missing = [str(path.relative_to(ROOT)) for path in required
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        issues.append("missing/empty artifacts: " + ", ".join(missing))
    delivery_path = OUT / "report_delivery.json"
    if delivery_path.is_file():
        delivery = json.loads(delivery_path.read_text())
        if delivery.get("status") not in ("PASS", "PASS_WITH_STRUCTURAL_BROWSER_FALLBACK"):
            issues.append(f"invalid portable report delivery: {delivery.get('status')}")
        checks.append(
            "portable report: " + str(delivery.get("status", "UNKNOWN"))
        )
    ledger = json.loads((ROOT / "analysis/noisy_wham/NOISY_TEST_EXECUTION.json").read_text())
    if ledger.get("execution_count") != 1 or ledger.get("post_test_tuning") is not False:
        issues.append("invalid TEST execution ledger")
    status = "READY_TO_SHARE" if not issues else "NEEDS_REVISION"
    result = {"status": status, "issues": issues, "checks": checks,
              "paper_ready": not issues, "post_test_tuning": False}
    atomic(OUT / "validation.json", json.dumps(result, indent=2) + "\n")
    report = f"""# Validation Report

## Overall Assessment: {'Ready to share' if not issues else 'Needs revision'}

Question: are the frozen noisy DEV/TEST results complete, internally consistent, and ready for the ICASSP report?

## Methodology and calculation checks

""" + "\n".join(f"- {check}" for check in checks) + """

## Issues found

""" + ("- None." if not issues else "\n".join(f"- {issue}" for issue in issues)) + """

## Required caveats

- WER is frozen ASR consistency; DNSMOS/UTMOS are model estimates.
- The residual difficulty includes interferer, noise, and extraction error.
- No post-TEST tuning or second noisy TEST run is allowed.
"""
    atomic(OUT / "VALIDATION_REPORT.md", report)
    print(json.dumps(result, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
