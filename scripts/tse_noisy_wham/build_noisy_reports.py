#!/usr/bin/env python3
"""Build the exact ICASSP noisy-TSE tables, figures, reports, and verdicts."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/noisy_tse"
FIG = ROOT / "figures/noisy_tse"
DOCS = ROOT / "docs"
CODES = ("D0", "D3", "G2", "G3", "G5", "G6")
GROUNDING = ("D3", "G2", "G3", "G5", "G6")
TAIL_CODES = ("G2", "G3", "G5", "G6")
LABELS = {
    "D0": "Primary WeSep", "D3": "CDCS-5 direct",
    "G2": "Qwen-TSE UD", "G3": "Qwen-TSE fixed CSG",
    "G5": "Qwen-TSE adaptive CSG", "G6": "Qwen-TSE adaptive CSG+GNR",
}
COLORS = {
    "D0": "#475569", "D3": "#0F766E", "G2": "#94A3B8",
    "G3": "#7C3AED", "G5": "#0284C7", "G6": "#E76F51",
}
MARKERS = {"D0": "o", "D3": "s", "G2": "^", "G3": "D", "G5": "P", "G6": "X"}
SNRS = (-5, 0, 5, 10, 15)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def summaries(split: str) -> dict[str, Any]:
    value = read_json(ROOT / f"results/noisy_wham/{split}/compiled/all_system_summaries.json")
    if value.get("status") != "COMPLETE" or value.get("split") != split:
        raise ValueError(f"compiled {split} is incomplete")
    if set(CODES) - set(value["systems"]):
        raise ValueError(f"compiled {split} is missing a main system")
    return value["systems"]


def candidate(split: str) -> dict[str, Any]:
    return read_json(
        ROOT / f"analysis/noisy_wham/{split}/full/candidate_analysis/candidate_analysis.json"
    )


def stat_rows(split: str) -> list[dict[str, Any]]:
    return read_json(ROOT / f"results/noisy_wham/{split}/paired_statistics.json")["results"]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{100 * float(value):.{digits}f}%"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def significant(stats: list[dict[str, Any]], comparison: str, metric: str,
                favorable: str) -> bool:
    matches = [row for row in stats if row["comparison"] == comparison
               and row["family"] == "continuous" and row["metric"] == metric]
    if not matches:
        return False
    low, high = matches[0]["ci95"]
    return high < 0 if favorable == "lower" else low > 0


def reliability(row: dict[str, Any]) -> float:
    return (float(row["target_WER_raw"])
            + 0.5 * float(row["content_switch_rate"])
            + 0.25 * float(row["high_error_rate"]))


def decisions(test: dict[str, Any], cand_test: dict[str, Any],
              stats: list[dict[str, Any]]) -> dict[str, str]:
    c = cand_test["natural"]
    recovery = float(c["cdcs5_direct_recovery_rate_on_primary_swaps"])
    gap = float(c["cdcs5_selector_gap_on_primary_swaps"])
    regression = float(c["cdcs5_control_regression_rate"])
    if recovery >= 0.50 and gap <= 0.15 and regression <= 0.01:
        pool = "STRONG"
    elif recovery >= 0.25 and regression <= 0.05:
        pool = "PARTIAL"
    else:
        pool = "FAILED"

    def content_value(base: str, new: str) -> str:
        a, b = test[base]["natural"], test[new]["natural"]
        better = reliability(b) < reliability(a)
        worse = (float(b["target_WER_raw"]) > float(a["target_WER_raw"]) + 0.005
                 or float(b["content_switch_rate"]) > float(a["content_switch_rate"]) + 0.0025)
        if better and not worse:
            return "YES"
        if not better and worse:
            return "NO"
        return "MIXED"

    fixed = content_value("G2", "G3")
    adaptive = content_value("G3", "G5")
    a, b = test["G5"]["natural"], test["G6"]["natural"]
    quality_metrics = ("dnsmos_p808", "dnsmos_bak", "utmos")
    quality_gain = any(significant(stats, "G5_vs_G6", m, "higher") for m in quality_metrics)
    quality_loss = any(float(b[m]) < float(a[m]) - 0.01 for m in quality_metrics)
    gnr_quality = "YES" if quality_gain and not quality_loss else "NO" if quality_loss and not quality_gain else "MIXED"
    content_preserved = (
        float(b["target_WER_raw"]) <= float(a["target_WER_raw"]) + 0.005
        and float(b["content_switch_rate"]) <= float(a["content_switch_rate"]) + 0.0025
        and not significant(stats, "G5_vs_G6", "target_WER_raw", "higher")
    )
    gnr_content = "YES" if content_preserved else (
        "NO" if float(b["target_WER_raw"]) > float(a["target_WER_raw"]) + 0.01 else "MIXED"
    )
    best_code = read_json(
        ROOT / "analysis/noisy_wham/dev/full/best_generative_selection.json"
    )["selected_code"]
    direct, generative = test["D3"]["natural"], test[best_code]["natural"]
    llm_quality = any(significant(stats, "D3_vs_BEST_GENERATIVE", m, "higher")
                       for m in quality_metrics)
    material_content_loss = reliability(generative) > reliability(direct) + 0.01
    llm = "YES" if llm_quality and not material_content_loss else "MIXED" if llm_quality else "NO"
    return {
        "NOISY_CDCS5_REPAIR": pool,
        "FIXED_CSG_CONTENT_CONTROL": fixed,
        "ADAPTIVE_CSG_VALUE": adaptive,
        "GNR_QUALITY_RECOVERY": gnr_quality,
        "GNR_CONTENT_PRESERVATION": gnr_content,
        "LLM_NOISY_ADVANTAGE": llm,
    }


def make_main_table(dev: dict[str, Any], test: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, systems in (("DEV", dev), ("TEST", test)):
        for code in CODES:
            row = systems[code]["natural"]
            rows.append({
                "Split": split, "Code": code, "Method": LABELS[code],
                "WER": row["target_WER_capped"], "Raw_WER": row["target_WER_raw"],
                "Content_Switch": row["content_switch_rate"],
                "Target_Preference": row["target_content_preference_rate"],
                "High_Error": row["high_error_rate"], "Empty": row["empty_output_rate"],
                "Short": row["short_output_rate"],
                "Spk_Switch": row["acoustic_speaker_switch_rate"],
                "Target_Cosine": row["sim_target"], "Interferer_Cosine": row["sim_interferer"],
                "Spk_Margin": row["speaker_margin"],
                "Joint_Success": row["joint_speaker_content_recovery_rate"],
                "LPS": row["lps"], "SpeechBERTScore": row["speechbertscore"],
                "P808": row["dnsmos_p808"], "SIG": row["dnsmos_sig"],
                "BAK": row["dnsmos_bak"], "OVRL": row["dnsmos_ovl"],
                "UTMOS": row["utmos"],
            })
    write_csv(OUT / "NOISY_MAIN_TABLE.csv", rows)
    return rows


def make_controlled_table(dev: dict[str, Any], test: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, systems in (("DEV", dev), ("TEST", test)):
        for snr in SNRS:
            for code in GROUNDING:
                row = systems[code]["controlled_by_snr"][str(snr)]
                rows.append({
                    "Split": split, "SNR_dB": snr, "Code": code, "Method": LABELS[code],
                    "WER": row["target_WER_capped"], "Raw_WER": row["target_WER_raw"],
                    "Content_Switch": row["content_switch_rate"],
                    "Spk_Switch": row["acoustic_speaker_switch_rate"],
                    "P808": row["dnsmos_p808"], "BAK": row["dnsmos_bak"],
                    "UTMOS": row["utmos"], "p95_Raw_WER": row["target_WER_raw_p95"],
                    "High_Error": row["high_error_rate"],
                    "Mean_Lambda": row["selected_lambda"],
                    "CSG_Modified_Rate": row["modified_token_rate"],
                    "GNR_Edit_Rate": row["gnr_edit_rate"],
                })
    write_csv(OUT / "CONTROLLED_SNR_TABLE.csv", rows)
    return rows


def make_tail_table(dev: dict[str, Any], test: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, systems in (("DEV", dev), ("TEST", test)):
        for code in TAIL_CODES:
            row = systems[code]["natural"]
            rows.append({
                "Split": split, "Code": code, "Method": LABELS[code],
                "p50_Raw_WER": row["target_WER_raw_p50"],
                "p90_Raw_WER": row["target_WER_raw_p90"],
                "p95_Raw_WER": row["target_WER_raw_p95"],
                "p99_Raw_WER": row["target_WER_raw_p99"],
                "High_Error": row["high_error_rate"],
                "Worst_50_Mean_Raw_WER": row["frozen_worst_50_mean_raw_WER"],
                "Worst_100_Mean_Raw_WER": row["frozen_worst_100_mean_raw_WER"],
                "Worst_200_Mean_Raw_WER": row["frozen_worst_200_mean_raw_WER"],
                "Worst_10pct_Mean_Raw_WER": row["frozen_worst_10_percent_mean_raw_WER"],
            })
    write_csv(OUT / "TAIL_ROBUSTNESS_TABLE.csv", rows)
    return rows


def confusion_row(split: str, cohort: str, row: dict[str, Any],
                  joint: float | None) -> dict[str, Any]:
    return {
        "Split": split, "Cohort": cohort,
        "Primary_Swap_Count": row["primary_noisy_swap_count"],
        "Primary_Swap_Rate": row["primary_noisy_swap_rate"],
        "CDCS5_Availability": row["cdcs5_oracle_recovery_rate_on_primary_swaps"],
        "CDCS5_Oracle_Recovery": row["cdcs5_oracle_recovery_rate_on_primary_swaps"],
        "CDCS5_Cosine_Recovery": row["cdcs5_direct_recovery_rate_on_primary_swaps"],
        "Selector_Oracle_Gap": row["cdcs5_selector_gap_on_primary_swaps"],
        "TFmap_Only_Recovery": row["cdcs5_tfmap_only_recovery_rate_on_primary_swaps"],
        "Segment_Only_Recovery": row["cdcs5_segment_only_recovery_rate_on_primary_swaps"],
        "TFmap_Segment_Overlap": row["cdcs5_tfmap_segment_overlap_rate_on_primary_swaps"],
        "Control_Regression": row["cdcs5_control_regression_rate"],
        "Joint_Recovery": joint,
    }


def make_confusion_table(dev: dict[str, Any], test: dict[str, Any],
                         cdev: dict[str, Any], ctest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, systems, cand in (("DEV", dev, cdev), ("TEST", test, ctest)):
        rows.append(confusion_row(
            split, "natural", cand["natural"],
            systems["D3"]["noisy_primary_swaps"]["joint_speaker_content_recovery_rate"],
        ))
        for snr in SNRS:
            rows.append(confusion_row(split, f"controlled_{snr:+d}dB",
                                      cand["controlled_by_snr"][str(snr)], None))
    write_csv(OUT / "CONFUSION_REPAIR_TABLE.csv", rows)
    return rows


def tex_table(path: Path, rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> None:
    lines = [r"\begin{tabular}{" + "l" * len(fields) + "}", r"\toprule",
             " & ".join(label for _, label in fields) + " \\\\", r"\midrule"]
    for row in rows:
        values = [str(row[key]) if isinstance(row[key], str) else fmt(row[key])
                  for key, _ in fields]
        lines.append(" & ".join(values) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    atomic_text(path, "\n".join(lines) + "\n")


def configure_plots() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "semibold", "axes.labelcolor": "#1E293B",
        "xtick.color": "#334155", "ytick.color": "#334155",
    })


def save(fig: Any, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def figures(dev: dict[str, Any], test: dict[str, Any],
            cdev: dict[str, Any], ctest: dict[str, Any]) -> None:
    configure_plots()
    for split, cand in (("dev", cdev), ("test", ctest)):
        fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
        series = {
            "Primary swap": [cand["controlled_by_snr"][str(s)]["primary_noisy_swap_rate"] for s in SNRS],
            "CDCS-5 selected wrong": [1-cand["controlled_by_snr"][str(s)]["cdcs5_direct_target_correct_rate"] for s in SNRS],
            "CDCS-5 oracle wrong": [1-cand["controlled_by_snr"][str(s)]["cdcs5_oracle_availability_rate"] for s in SNRS],
        }
        for (label, values), color, marker in zip(series.items(),
                ("#475569", "#0F766E", "#E76F51"), ("o", "s", "^")):
            ax.plot(SNRS, 100*np.asarray(values), label=label, color=color,
                    marker=marker, linewidth=2)
        ax.set(xlabel="Speech-mixture-to-noise SNR (dB)", ylabel="Wrong-speaker / swap rate (%)",
               xticks=SNRS, title=f"Confusion robustness — {split.upper()}")
        ax.grid(axis="y", color="#CBD5E1", alpha=.7); ax.legend(frameon=False)
        save(fig, f"confusion_robustness_vs_snr_{split}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, (split, systems) in zip(axes, (("DEV", dev), ("TEST", test))):
        for code in GROUNDING:
            values = [100*systems[code]["controlled_by_snr"][str(s)]["target_WER_capped"] for s in SNRS]
            ax.plot(SNRS, values, color=COLORS[code], marker=MARKERS[code], label=LABELS[code])
        ax.set(xlabel="SNR (dB)", ylabel="Target WER (%)", xticks=SNRS, title=split)
        ax.grid(axis="y", alpha=.3)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "content_wer_vs_snr")

    for metric, stem, ylabel in (("dnsmos_p808", "perceptual_quality_p808", "DNSMOS P.808"),
                                  ("utmos", "perceptual_quality_utmos", "UTMOS"),
                                  ("dnsmos_bak", "perceptual_quality_bak", "DNSMOS BAK")):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        for ax, (split, systems) in zip(axes, (("DEV", dev), ("TEST", test))):
            for code in GROUNDING:
                values = [systems[code]["controlled_by_snr"][str(s)][metric] for s in SNRS]
                ax.plot(SNRS, values, color=COLORS[code], marker=MARKERS[code], label=LABELS[code])
            ax.set(xlabel="SNR (dB)", ylabel=ylabel, xticks=SNRS, title=split)
            ax.grid(axis="y", alpha=.3)
        axes[1].legend(frameon=False, fontsize=8)
        save(fig, stem)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for ax, (split, systems) in zip(axes, (("DEV", dev), ("TEST", test))):
        for code in GROUNDING:
            for snr in SNRS:
                row = systems[code]["controlled_by_snr"][str(snr)]
                ax.scatter(row["dnsmos_p808"], 100*row["target_WER_capped"],
                           color=COLORS[code], marker=MARKERS[code], s=38)
                ax.annotate(str(snr), (row["dnsmos_p808"], 100*row["target_WER_capped"]),
                            xytext=(3, 2), textcoords="offset points", fontsize=6.5)
        ax.set(xlabel="DNSMOS P.808", ylabel="Target WER (%)", title=split)
        ax.grid(alpha=.25)
    save(fig, "reliability_quality_tradeoff")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    keys = ("frozen_worst_50_mean_raw_WER", "frozen_worst_100_mean_raw_WER",
            "frozen_worst_200_mean_raw_WER", "target_WER_raw_p95",
            "target_WER_raw_p99", "high_error_rate")
    labels = ("Worst 50", "Worst 100", "Worst 200", "p95", "p99", ">50%")
    x = np.arange(len(keys)); width = .18
    for ax, (split, systems) in zip(axes, (("DEV", dev), ("TEST", test))):
        for offset, code in enumerate(TAIL_CODES):
            vals = [systems[code]["natural"][key] for key in keys]
            ax.bar(x+(offset-1.5)*width, vals, width, color=COLORS[code], label=LABELS[code])
        ax.set_xticks(x, labels, rotation=20); ax.set_title(split); ax.set_ylabel("Raw WER / event rate")
        ax.grid(axis="y", alpha=.25)
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, "tail_robustness")

    fig, ax1 = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    lam = [dev["G5"]["controlled_by_snr"][str(s)]["selected_lambda"] for s in SNRS]
    mod = [dev["G5"]["controlled_by_snr"][str(s)]["modified_token_rate"] for s in SNRS]
    edit = [dev["G6"]["controlled_by_snr"][str(s)]["gnr_edit_rate"] for s in SNRS]
    ax1.plot(SNRS, lam, color=COLORS["G5"], marker="o", label="Mean selected λ")
    ax1.set(xlabel="SNR (dB)", ylabel="Mean selected λ", xticks=SNRS)
    ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(SNRS, mod, color="#7C3AED", marker="s", label="CSG modified")
    ax2.plot(SNRS, edit, color=COLORS["G6"], marker="^", label="GNR edits")
    ax2.set_ylabel("Position rate")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [x.get_label() for x in lines], frameon=False)
    ax1.grid(axis="y", alpha=.3)
    save(fig, "adaptive_grounding_behavior")


def reports(dev: dict[str, Any], test: dict[str, Any], ctest: dict[str, Any],
            verdict: dict[str, str]) -> None:
    cal = read_json(ROOT / "analysis/noisy_wham/dev/full/adaptive/residual_snr_calibration.json")
    adaptive = read_json(ROOT / "analysis/noisy_wham/dev/full/adaptive/frozen_adaptive_dev_choice.json")
    gnr = read_json(ROOT / "analysis/noisy_wham/dev/full/gnr_selection.json")
    mech = read_json(ROOT / "analysis/noisy_wham/dev/full/gnr_mechanism/summary.json")
    best = read_json(ROOT / "analysis/noisy_wham/dev/full/best_generative_selection.json")
    c = ctest["natural"]
    headline_rows = []
    for code in CODES:
        row = test[code]["natural"]
        headline_rows.append([LABELS[code], pct(row["target_WER_capped"]),
                              pct(row["content_switch_rate"]), pct(row["acoustic_speaker_switch_rate"]),
                              fmt(row["dnsmos_p808"]), fmt(row["dnsmos_bak"]), fmt(row["utmos"])])
    headline = md_table(["Method", "WER↓", "Content switch↓", "Spk switch↓", "P808↑", "BAK↑", "UTMOS↑"], headline_rows)
    main = f"""# Noisy TSE Main Report

## Answer first

The frozen noisy study completed natural and controlled-SNR DEV plus one frozen TEST run. CDCS-5 repair is **{verdict['NOISY_CDCS5_REPAIR']}**; fixed CSG content control is **{verdict['FIXED_CSG_CONTENT_CONTROL']}**; adaptive CSG value is **{verdict['ADAPTIVE_CSG_VALUE']}**; GNR quality recovery/content preservation are **{verdict['GNR_QUALITY_RECOVERY']} / {verdict['GNR_CONTENT_PRESERVATION']}**. Post-TEST tuning: **NO**.

{headline}

## Data and protocol

Natural DEV/TEST each contain 3,000 Libri2Mix/WHAM mixtures evaluated in both target directions (6,000 trials). Controlled DEV/TEST each add 2,400 trials at −5/0/5/10/15 dB. Enrollment remains clean. All candidate construction, Qwen-TSE, fixed CSG, adaptive mapping, GNR K/R, model hashes, and metric code were frozen before noisy TEST. TEST was executed once.

## Main findings

- Natural TEST primary swap count: {c['primary_noisy_swap_count']}; CDCS-5 cosine recovery {pct(c['cdcs5_direct_recovery_rate_on_primary_swaps'])}, oracle {pct(c['cdcs5_oracle_recovery_rate_on_primary_swaps'])}, gap {pct(c['cdcs5_selector_gap_on_primary_swaps'])}, control regression {pct(c['cdcs5_control_regression_rate'])}.
- Adaptive policy: **{adaptive['selected_policy']}**, temporal tolerance `w={adaptive['selected_temporal_tolerance']}`. Residual calibration Spearman {fmt(cal['spearman'])}, MAE {fmt(cal['mae_db'])} dB.
- GNR: {gnr['selected']} (`K={gnr['K']}`, `R={gnr['R']}`), frozen on DEV. Best-generative comparison arm: {best['selected_name']}.
- STOI/ESTOI/PESQ and generative SI-SDR remain diagnostic only and are excluded from the noisy main table.

## Statistics and limitations

Continuous metrics use 10,000 paired bootstrap resamples and binary metrics use exact McNemar. The residual contains interferer speech, ambient noise, and extraction error. DNSMOS/UTMOS are model estimates; WER is frozen ASR consistency. No post-TEST threshold, lambda, K/R, candidate, or metric change is permitted.

## Verdicts

""" + "\n".join(f"- `{key} = {value}`" for key, value in verdict.items()) + "\n"
    atomic_text(DOCS / "NOISY_TSE_MAIN_REPORT.md", main)

    pool = f"""# Noisy CDCS-5 Analysis

On natural TEST, CDCS-5 recovers {pct(c['cdcs5_direct_recovery_rate_on_primary_swaps'])} of {c['primary_noisy_swap_count']} noisy primary swaps. Oracle availability is {pct(c['cdcs5_oracle_recovery_rate_on_primary_swaps'])}; the selector gap is {pct(c['cdcs5_selector_gap_on_primary_swaps'])}. Correct-case regression is {pct(c['cdcs5_control_regression_rate'])}. Verdict: **{verdict['NOISY_CDCS5_REPAIR']}**.

- TF-map-only recoverability: {pct(c['cdcs5_tfmap_only_recovery_rate_on_primary_swaps'])}
- Segmented-view-only recoverability: {pct(c['cdcs5_segment_only_recovery_rate_on_primary_swaps'])}
- TF-map/segmented overlap: {pct(c['cdcs5_tfmap_segment_overlap_rate_on_primary_swaps'])}

Candidates are full, first, middle, final, and TF-map/context full, all from the same clean enrollment. Selection uses frozen enrollment ECAPA cosine only. Clean targets, interferers, transcripts, WER, SI-SDR, SNR, and swap labels are evaluation-only.
"""
    atomic_text(DOCS / "NOISY_CDCS5_ANALYSIS.md", pool)

    grounding = f"""# Noisy Grounding Analysis

Qwen-TSE unrestricted decoding is compared with fixed CSG (`λ=1,w=0`) and **{adaptive['selected_policy']}** (`w={adaptive['selected_temporal_tolerance']}`). Source-threshold and TSE-DEV-calibrated policies were compared on controlled DEV; TEST reused the frozen winner. Observation blending is never used because it would reintroduce the competing speaker.

Residual calibration: Pearson {fmt(cal['pearson'])}, Spearman {fmt(cal['spearman'])}, MAE {fmt(cal['mae_db'])} dB, RMSE {fmt(cal['rmse_db'])} dB. Fixed CSG content-control verdict: **{verdict['FIXED_CSG_CONTENT_CONTROL']}**. Adaptive value: **{verdict['ADAPTIVE_CSG_VALUE']}**.
"""
    atomic_text(DOCS / "NOISY_GROUNDING_ANALYSIS.md", grounding)

    gnr_report = f"""# GNR-LLM TSE Analysis

GNR uses one frozen Qwen-TSE teacher-forced pass over the immutable complete adaptive anchor. Each position chooses the highest-scoring token in `TopK ∩ HammingBall_R`, union the anchor; refined tokens never feed later histories. DEV selected **{gnr['selected']}** (`K={gnr['K']}, R={gnr['R']}`).

- Median accepted LLM score margin: {fmt(mech['median_accepted_llm_score_margin'], 4)}
- Mean unchanged-position fraction: {pct(mech['mean_fraction_positions_unchanged'])}
- Accepted edit positions: {mech['accepted_edit_positions']}
- Quality recovery: **{verdict['GNR_QUALITY_RECOVERY']}**
- Content preservation: **{verdict['GNR_CONTENT_PRESERVATION']}**

Clean target tokens were used only for the labeled DEV mechanism analysis, never for decoding or selection. Implementation correctness is in `docs/GNR_TSE_IMPLEMENTATION_AUDIT.md`.
"""
    atomic_text(DOCS / "GNR_LLM_TSE_ANALYSIS.md", gnr_report)

    coauthor = f"""# Noisy TSE Coauthor Summary

1. **Does noise increase wrong-speaker extraction?** Natural TEST contains {c['primary_noisy_swap_count']} high-confidence primary swaps; compare carefully with clean because definitions differ.
2. **Does CDCS-5 fix it?** {pct(c['cdcs5_direct_recovery_rate_on_primary_swaps'])}; verdict **{verdict['NOISY_CDCS5_REPAIR']}**.
3. **How close is selection to oracle?** Oracle {pct(c['cdcs5_oracle_recovery_rate_on_primary_swaps'])}; gap {pct(c['cdcs5_selector_gap_on_primary_swaps'])}.
4. **Does Qwen-TSE drift at low SNR?** See `content_wer_vs_snr.pdf`; the plot reports the data without forcing the hypothesis.
5. **Does fixed CSG help?** **{verdict['FIXED_CSG_CONTENT_CONTROL']}**.
6. **Does adaptive grounding help?** **{verdict['ADAPTIVE_CSG_VALUE']}**.
7. **Does GNR improve quality?** Quality **{verdict['GNR_QUALITY_RECOVERY']}**, content preservation **{verdict['GNR_CONTENT_PRESERVATION']}**.
8. **Strongest generative system?** {best['selected_name']} on DEV; CDCS-5 direct remains the deterministic reference.
9. **ICASSP claim:** repair target identity before grounding, constrain generation to control drift, and treat GNR only as local quality refinement.
"""
    atomic_text(DOCS / "NOISY_TSE_COAUTHOR_SUMMARY.md", coauthor)


def paper_tex(main_rows: list[dict[str, Any]], controlled_rows: list[dict[str, Any]],
              tail_rows: list[dict[str, Any]], test: dict[str, Any],
              ctest: dict[str, Any], verdict: dict[str, str]) -> None:
    tex_table(OUT / "noisy_main_table.tex", main_rows,
              [("Split", "Split"), ("Method", "Method"), ("WER", "WER"),
               ("Content_Switch", "CSw"), ("Spk_Switch", "SSw"),
               ("Spk_Margin", "SpkM"), ("LPS", "LPS"),
               ("SpeechBERTScore", "SpB"), ("P808", "P808"),
               ("SIG", "SIG"), ("BAK", "BAK"), ("UTMOS", "UTMOS")])
    tex_table(OUT / "controlled_snr_table.tex", controlled_rows,
              [("Split", "Split"), ("SNR_dB", "SNR"), ("Method", "Method"),
               ("WER", "WER"), ("Content_Switch", "CSw"),
               ("Spk_Switch", "SSw"), ("P808", "P808"),
               ("BAK", "BAK"), ("UTMOS", "UTMOS"),
               ("p95_Raw_WER", "p95"), ("High_Error", "HighErr")])
    tex_table(OUT / "tail_table.tex", tail_rows,
              [("Split", "Split"), ("Method", "Method"),
               ("Worst_50_Mean_Raw_WER", "W50"),
               ("Worst_100_Mean_Raw_WER", "W100"),
               ("Worst_200_Mean_Raw_WER", "W200"),
               ("Worst_10pct_Mean_Raw_WER", "W10pct"),
               ("p95_Raw_WER", "p95"), ("p99_Raw_WER", "p99")])
    c = ctest["natural"]
    g2, g3, g5, g6 = (test[x]["natural"] for x in ("G2", "G3", "G5", "G6"))
    paragraph = (
        f"Ambient noise produced {c['primary_noisy_swap_count']} high-confidence primary speaker swaps on natural TEST. "
        f"CDCS-5 recovered {pct(c['cdcs5_direct_recovery_rate_on_primary_swaps'])} versus {pct(c['cdcs5_oracle_recovery_rate_on_primary_swaps'])} oracle availability, with {pct(c['cdcs5_control_regression_rate'])} control regression. "
        f"Target WER changed from {pct(g2['target_WER_capped'])} for unrestricted Qwen-TSE to {pct(g3['target_WER_capped'])} with fixed CSG and {pct(g5['target_WER_capped'])} with frozen adaptive CSG. "
        f"GNR changed P808 from {fmt(g5['dnsmos_p808'])} to {fmt(g6['dnsmos_p808'])} and UTMOS from {fmt(g5['utmos'])} to {fmt(g6['utmos'])}, yielding quality/content verdicts {verdict['GNR_QUALITY_RECOVERY']}/{verdict['GNR_CONTENT_PRESERVATION']}. "
        "Overall, the results support repairing target identity before language-model grounding while treating generative quality gains and content drift as separate paired outcomes."
    )
    if len(paragraph.split()) > 170:
        raise ValueError("ICASSP paragraph exceeds 170 words")
    combined = (
        "% Auto-generated frozen noisy-TSE results.\n"
        "\\input{results/noisy_tse/noisy_main_table.tex}\n"
        "\\input{results/noisy_tse/controlled_snr_table.tex}\n"
        "\\input{results/noisy_tse/tail_table.tex}\n\n"
        "\\paragraph{Noisy robustness results.} " + paragraph + "\n"
    )
    atomic_text(OUT / "noisy_results_for_paper.tex", combined)


def main() -> int:
    dev, test = summaries("dev"), summaries("test")
    cdev, ctest = candidate("dev"), candidate("test")
    verdict = decisions(test, ctest, stat_rows("test"))
    main_rows = make_main_table(dev, test)
    controlled_rows = make_controlled_table(dev, test)
    tail_rows = make_tail_table(dev, test)
    make_confusion_table(dev, test, cdev, ctest)
    paper_tex(main_rows, controlled_rows, tail_rows, test, ctest, verdict)
    figures(dev, test, cdev, ctest)
    reports(dev, test, ctest, verdict)
    scientific = {
        "status": "COMPLETE", "verdicts": verdict,
        "post_test_tuning": False, "test_execution_count": 1,
        "main_systems": list(CODES), "test_used": True,
    }
    atomic_text(OUT / "scientific_verdict.json", json.dumps(scientific, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "verdicts": verdict,
                      "post_test_tuning": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
