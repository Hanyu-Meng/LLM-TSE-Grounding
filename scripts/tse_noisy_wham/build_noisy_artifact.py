#!/usr/bin/env python3
"""Create the canonical portable-report artifact from validated noisy tables."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/noisy_tse"


def read_csv(name: str) -> list[dict]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "WER", "Raw_WER", "Content_Switch", "Spk_Switch", "Spk_Margin",
        "LPS", "SpeechBERTScore", "P808", "SIG", "BAK", "OVRL", "UTMOS",
        "SNR_dB", "p95_Raw_WER", "High_Error", "Mean_Lambda",
        "CSG_Modified_Rate", "GNR_Edit_Rate", "Primary_Swap_Count",
        "Primary_Swap_Rate", "Pool_D_Availability", "Pool_D_Oracle_Recovery",
        "Pool_D_Cosine_Recovery", "Selector_Oracle_Gap", "Control_Regression",
        "p50_Raw_WER", "p90_Raw_WER", "p99_Raw_WER",
        "Worst_50_Mean_Raw_WER", "Worst_100_Mean_Raw_WER",
        "Worst_200_Mean_Raw_WER", "Worst_10pct_Mean_Raw_WER",
    }
    for row in rows:
        for key in numeric & set(row):
            row[key] = None if row[key] in ("", "None") else float(row[key])
    return rows


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def source(source_id: str, label: str, path: str, description: str) -> dict:
    return {
        "id": source_id, "label": label, "path": path,
        "query": {
            "description": description, "language": "SQL", "engine": "DuckDB",
            "sql": f"SELECT * FROM read_csv_auto('{path}')",
            "executed_at": datetime.now(ZoneInfo("Australia/Sydney")).isoformat(),
            "filters": ["natural or controlled split exactly as labeled", "six frozen main systems"],
            "metric_definitions": [
                "WER is frozen Whisper-small.en ASR-consistency word error rate",
                "Content switch means interferer WER is lower than target WER",
                "Speaker switch means CAMPPlus target cosine is below interferer cosine",
                "P808/SIG/BAK/OVRL are DNSMOS outputs; UTMOS is reference-free",
            ],
            "tables_used": [path],
        },
    }


def main() -> int:
    main_rows = read_csv("NOISY_MAIN_TABLE.csv")
    controlled = read_csv("CONTROLLED_SNR_TABLE.csv")
    confusion = read_csv("CONFUSION_REPAIR_TABLE.csv")
    tail = read_csv("TAIL_ROBUSTNESS_TABLE.csv")
    test_main = [row for row in main_rows if row["Split"] == "TEST"]
    test_controlled = [row for row in controlled if row["Split"] == "TEST"]
    test_confusion = [row for row in confusion if row["Split"] == "TEST"]
    test_tail = [row for row in tail if row["Split"] == "TEST"]
    cand = json.loads((
        ROOT / "analysis/noisy_wham/test/full/candidate_analysis/candidate_analysis.json"
    ).read_text())["natural"]
    verdict = json.loads((OUT / "scientific_verdict.json").read_text())["verdicts"]
    by_code = {row["Code"]: row for row in test_main}
    headline = [{
        "pool_d_swap_recovery": cand["pool_d_selected_recovery_rate_on_primary_swaps"],
        "pool_d_oracle": cand["pool_d_oracle_recovery_rate_on_primary_swaps"],
        "pool_d_control_regression": cand["pool_d_control_regression_rate"],
        "primary_wer": by_code["D0"]["WER"],
        "pool_d_wer": by_code["D3"]["WER"],
        "qfull_wer": by_code["G2"]["WER"],
        "adaptive_wer": by_code["G5"]["WER"],
        "gnr_wer": by_code["G6"]["WER"],
    }]
    now = datetime.now(ZoneInfo("Australia/Sydney")).isoformat()
    sources = [
        source("main_source", "Frozen noisy main table",
               "results/noisy_tse/NOISY_MAIN_TABLE.csv",
               "Compiled natural DEV/TEST metrics for the six frozen systems."),
        source("controlled_source", "Controlled-SNR table",
               "results/noisy_tse/CONTROLLED_SNR_TABLE.csv",
               "Compiled controlled -5/0/5/10/15 dB results."),
        source("confusion_source", "Confusion repair table",
               "results/noisy_tse/CONFUSION_REPAIR_TABLE.csv",
               "Candidate availability, selector recovery, complementarity, and regression."),
        source("tail_source", "Frozen tail table",
               "results/noisy_tse/TAIL_ROBUSTNESS_TABLE.csv",
               "Raw-WER tails using Q-Full-UD-ranked common subsets."),
    ]
    cards = [
        {"id": "repair_card", "dataset": "headline", "sourceId": "confusion_source",
         "description": "Share of noisy primary swaps corrected by frozen Pool D cosine selection.",
         "metrics": [{"label": "Pool D swap recovery", "field": "pool_d_swap_recovery", "format": "percent"},
                     {"label": "Oracle availability", "field": "pool_d_oracle", "format": "percent"}]},
        {"id": "regression_card", "dataset": "headline", "sourceId": "confusion_source",
         "description": "Wrong-target regression among noisy-primary-correct controls.",
         "metrics": [{"label": "Control regression", "field": "pool_d_control_regression", "format": "percent"}]},
        {"id": "wer_card", "dataset": "headline", "sourceId": "main_source",
         "description": "Natural TEST target WER for primary and direct Pool D outputs.",
         "metrics": [{"label": "Pool D direct WER", "field": "pool_d_wer", "format": "percent"},
                     {"label": "Primary WER", "field": "primary_wer", "format": "percent"}]},
        {"id": "grounding_card", "dataset": "headline", "sourceId": "main_source",
         "description": "Natural TEST WER across unrestricted, adaptive, and local-refinement generation.",
         "metrics": [{"label": "Adaptive CSG WER", "field": "adaptive_wer", "format": "percent"},
                     {"label": "Q-Full UD", "field": "qfull_wer", "format": "percent"},
                     {"label": "GNR-LLM", "field": "gnr_wer", "format": "percent"}]},
    ]
    charts = [
        {"id": "wer_snr", "title": "Target WER across controlled SNR",
         "subtitle": "All points use the same frozen TEST configurations.",
         "type": "line", "dataset": "test_controlled", "sourceId": "controlled_source",
         "encodings": {"x": {"field": "SNR_dB", "type": "quantitative", "label": "SNR (dB)"},
                       "y": {"field": "WER", "type": "quantitative", "format": "percent", "label": "Target WER"},
                       "color": {"field": "Method", "type": "nominal", "label": "Method"}},
         "layout": "full", "maxRows": 25},
        {"id": "quality_tradeoff", "title": "Reliability–quality operating points",
         "subtitle": "Each label is one frozen method–SNR condition on TEST.",
         "type": "scatter", "dataset": "test_controlled", "sourceId": "controlled_source",
         "encodings": {"x": {"field": "P808", "type": "quantitative", "label": "DNSMOS P.808"},
                       "y": {"field": "WER", "type": "quantitative", "format": "percent", "label": "Target WER"},
                       "color": {"field": "Method", "type": "nominal", "label": "Method"},
                       "label": {"field": "SNR_dB", "type": "text", "label": "SNR"}},
         "layout": "full", "maxRows": 25},
    ]
    tables = [
        {"id": "main_table", "title": "Natural noisy TEST — six-system result",
         "dataset": "test_main", "sourceId": "main_source", "layout": "full",
         "defaultSort": {"field": "WER", "direction": "asc"}, "density": "dense",
         "columns": [
             {"field": "Method", "label": "Method", "type": "text"},
             {"field": "WER", "label": "WER", "format": "percent"},
             {"field": "Content_Switch", "label": "Content switch", "format": "percent"},
             {"field": "Spk_Switch", "label": "Speaker switch", "format": "percent"},
             {"field": "Spk_Margin", "label": "Speaker margin", "format": "number"},
             {"field": "P808", "label": "P808", "format": "number"},
             {"field": "BAK", "label": "BAK", "format": "number"},
             {"field": "UTMOS", "label": "UTMOS", "format": "number"},
         ]},
        {"id": "tail_table", "title": "Common Q-Full-UD-ranked reliability tails",
         "dataset": "test_tail", "sourceId": "tail_source", "layout": "full",
         "defaultSort": {"field": "Worst_100_Mean_Raw_WER", "direction": "asc"}, "density": "dense",
         "columns": [
             {"field": "Method", "label": "Method", "type": "text"},
             {"field": "Worst_50_Mean_Raw_WER", "label": "Worst 50", "format": "number"},
             {"field": "Worst_100_Mean_Raw_WER", "label": "Worst 100", "format": "number"},
             {"field": "Worst_200_Mean_Raw_WER", "label": "Worst 200", "format": "number"},
             {"field": "Worst_10pct_Mean_Raw_WER", "label": "Worst 10%", "format": "number"},
             {"field": "p95_Raw_WER", "label": "p95", "format": "number"},
         ]},
    ]
    blocks = [
        {"id": "title", "type": "markdown", "layout": "full",
         "body": "# Repair Before Grounding: Noisy TSE Results\n\nFrozen ICASSP 2027 robustness study."},
        {"id": "answer", "type": "markdown", "layout": "full",
         "body": ("## Answer first\n\nPool D repair: **" + verdict["NOISY_POOL_D_REPAIR"]
                  + "**. Fixed/adaptive grounding: **" + verdict["FIXED_CSG_CONTENT_CONTROL"]
                  + "/" + verdict["ADAPTIVE_CSG_VALUE"] + "**. GNR quality/content: **"
                  + verdict["GNR_QUALITY_RECOVERY"] + "/" + verdict["GNR_CONTENT_PRESERVATION"]
                  + "**. TEST was run once; post-TEST tuning is NO.")},
        {"id": "metrics", "type": "metric-strip", "layout": "full",
         "cardIds": ["repair_card", "regression_card", "wer_card", "grounding_card"]},
        {"id": "reliability_heading", "type": "markdown", "layout": "full",
         "body": "## Reliability across acoustic difficulty\n\nControlled SNR reveals where unrestricted generation drifts and whether grounding limits that drift."},
        {"id": "wer_chart", "type": "chart", "chartId": "wer_snr", "layout": "full"},
        {"id": "tradeoff_heading", "type": "markdown", "layout": "full",
         "body": "## Reliability–quality trade-off\n\nQuality is not treated as a win when content reliability materially regresses."},
        {"id": "quality_chart", "type": "chart", "chartId": "quality_tradeoff", "layout": "full"},
        {"id": "main_heading", "type": "markdown", "layout": "full",
         "body": "## Natural noisy TEST result\n\nThe six rows are the complete paper mainline."},
        {"id": "main", "type": "table", "tableId": "main_table", "layout": "full"},
        {"id": "tail_heading", "type": "markdown", "layout": "full",
         "body": "## Tail robustness\n\nEvery method is evaluated on the same Q-Full-UD-ranked worst-case IDs."},
        {"id": "tail", "type": "table", "tableId": "tail_table", "layout": "full"},
        {"id": "caveats", "type": "markdown", "layout": "full",
         "body": "## Caveats\n\nWER is frozen ASR consistency. DNSMOS and UTMOS are model-based estimates. The deployable difficulty residual contains interferer speech, ambient noise, and extraction error. STOI/ESTOI/PESQ and generative SI-SDR are diagnostic only."},
    ]
    manifest = {
        "version": 1, "surface": "report",
        "title": "Repair Before Grounding: Noisy TSE Results",
        "description": "Frozen DEV and one-run TEST evidence for confusion-resilient TSE.",
        "generatedAt": now, "cards": cards, "charts": charts, "tables": tables,
        "sources": sources, "blocks": blocks,
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {"version": 1, "generatedAt": now, "status": "ready",
                     "datasets": {"headline": headline, "test_main": test_main,
                                  "test_controlled": test_controlled,
                                  "test_confusion": test_confusion, "test_tail": test_tail}},
        "sources": sources,
    }
    atomic(OUT / "artifact.json", artifact)
    print(json.dumps({"status": "COMPLETE", "datasets": {
        key: len(value) for key, value in artifact["snapshot"]["datasets"].items()
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
