#!/usr/bin/env python3
"""Build compact, paper-ready clean/noisy TEST tables from frozen outputs."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/paper_tables"
CLEAN_SOURCE = ROOT / "results/selected_evidence_test/MAIN_ICASSP_TEST_TABLE.csv"
CLEAN_GNR_SOURCE = (
    ROOT
    / "results/selected_evidence_test/systems/qwen_tse_gnr_cdcs5_k20_r2/summary.json"
)
CLEAN_GNR_PROTOCOL = ROOT / "analysis/clean_table_completion/clean_gnr_protocol.json"
NOISY_SOURCE = ROOT / "results/noisy_tse/NOISY_MAIN_TABLE.csv"
NOISY_VALIDATION = ROOT / "results/noisy_tse/validation.json"
NOISY_EXECUTION = ROOT / "analysis/noisy_wham/NOISY_TEST_EXECUTION.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def finite_row(row: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(math.isfinite(float(row[key])) for key in keys)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index < 2 else "---:" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("→", r"$\rightarrow$")
    )


def build_clean_rows() -> list[dict[str, Any]]:
    source_rows = {
        row["system"]: row
        for row in read_csv(CLEAN_SOURCE)
        if row["cohort"] == "full_test"
    }
    selected = [
        ("D0", "Primary WeSep", "Primary WeSep", "pre-registered clean TEST"),
        ("D3", "CDCS-5 direct", "CDCS-5 direct", "pre-registered clean TEST"),
        ("G2", "Qwen-TSE UD", "Qwen-TSE UD (CDCS-5 evidence)", "pre-registered clean TEST"),
        ("G3", "Qwen-TSE fixed CSG", "Qwen-TSE fixed CSG (CDCS-5 evidence)", "pre-registered clean TEST"),
    ]
    rows: list[dict[str, Any]] = []
    for code, label, source_name, protocol in selected:
        source = source_rows[source_name]
        rows.append({
            "Code": code,
            "Method": label,
            "WER": float(source["target_wer"]),
            "Content_Switch": float(source["content_switch_rate"]),
            "Spk_Switch": float(source["acoustic_switch_rate"]),
            "Spk_Margin": float(source["speaker_margin"]),
            "P808": float(source["dnsmos_p808"]),
            "Short_Output": float(source["short_output_rate"]),
            "Protocol_Status": protocol,
        })

    gnr = read_json(CLEAN_GNR_SOURCE)
    protocol = read_json(CLEAN_GNR_PROTOCOL)
    if not (
        gnr.get("status") == "COMPLETE"
        and gnr.get("expected") == gnr.get("unique") == 6000
        and gnr.get("failed") == 0
        and protocol.get("status") == "COMPLETE"
        and protocol.get("execution_count") == 1
        and protocol.get("clean_test_used_for_configuration") is False
        and protocol.get("configuration") == {
            "K": 20,
            "R": 2,
            "anchor": "qwen_tse_fixed_csg_cdcs5_lambda_1",
        }
    ):
        raise RuntimeError("Clean GNR completion/provenance gate failed")
    full = gnr["full_test"]
    rows.append({
        "Code": "G4*",
        "Method": "Qwen-TSE GNR (K=20, R=2)",
        "WER": float(full["target_WER"]),
        "Content_Switch": float(full["content_switch_rate"]),
        "Spk_Switch": float(full["acoustic_speaker_switch_rate"]),
        "Spk_Margin": float(full["speaker_margin"]),
        "P808": float(full["dnsmos_p808"]),
        "Short_Output": float(full["short_output_rate"]),
        "Protocol_Status": "post-hoc clean readback; K/R fixed on noisy DEV; no clean TEST tuning",
    })
    return rows


def build_noisy_rows() -> list[dict[str, Any]]:
    validation = read_json(NOISY_VALIDATION)
    execution = read_json(NOISY_EXECUTION)
    if not (
        validation.get("status") == "READY_TO_SHARE"
        and validation.get("paper_ready") is True
        and validation.get("post_test_tuning") is False
        and execution.get("status") == "COMPLETE"
        and execution.get("execution_count") == 1
        and execution.get("post_test_tuning") is False
    ):
        raise RuntimeError("Noisy TEST validation/provenance gate failed")
    rows = [row for row in read_csv(NOISY_SOURCE) if row["Split"] == "TEST"]
    if len(rows) != 6:
        raise RuntimeError(f"Expected six noisy TEST systems, found {len(rows)}")
    return [{
        "Code": row["Code"],
        "Method": row["Method"],
        "WER": float(row["WER"]),
        "Content_Switch": float(row["Content_Switch"]),
        "Spk_Switch": float(row["Spk_Switch"]),
        "Spk_Margin": float(row["Spk_Margin"]),
        "P808": float(row["P808"]),
        "BAK": float(row["BAK"]),
        "UTMOS": float(row["UTMOS"]),
        "Short_Output": float(row["Short"]),
        "Protocol_Status": "single frozen noisy TEST",
    } for row in rows]


def build_markdown(clean: list[dict[str, Any]], noisy: list[dict[str, Any]]) -> str:
    clean_rows = [[
        str(row["Code"]), str(row["Method"]), percent(row["WER"]),
        percent(row["Content_Switch"]), percent(row["Spk_Switch"]),
        f"{row['Spk_Margin']:.3f}", f"{row['P808']:.3f}",
        percent(row["Short_Output"]),
    ] for row in clean]
    noisy_rows = [[
        str(row["Code"]), str(row["Method"]), percent(row["WER"]),
        percent(row["Content_Switch"]), percent(row["Spk_Switch"]),
        f"{row['Spk_Margin']:.3f}", f"{row['P808']:.3f}",
        f"{row['BAK']:.3f}", f"{row['UTMOS']:.3f}",
    ] for row in noisy]
    csg = clean[3]
    gnr = clean[4]
    adaptive = next(row for row in noisy if row["Code"] == "G5")
    noisy_gnr = next(row for row in noisy if row["Code"] == "G6")
    return "\n".join([
        "# ICASSP 2027 Final Paper Tables",
        "",
        "## Clean natural TEST (6,000 trials)",
        "",
        markdown_table(
            ["Code", "Method", "WER ↓", "Content switch ↓", "Spk switch ↓", "Spk margin ↑", "P808 ↑", "Short ↓"],
            clean_rows,
        ),
        "",
        "`G4*` is a post-hoc clean readback of the K=20/R=2 GNR configuration selected on noisy DEV. It used no clean TEST tuning and does not redefine the original 10-system clean confirmatory test.",
        "",
        f"Clean GNR changes CSG WER by **{100 * (gnr['WER'] - csg['WER']):+.2f} pp** and P808 by **{gnr['P808'] - csg['P808']:+.3f}**. Direct CDCS-5 remains the clean endpoint.",
        "",
        "## Natural noisy TEST (6,000 trials)",
        "",
        markdown_table(
            ["Code", "Method", "WER ↓", "Content switch ↓", "Spk switch ↓", "Spk margin ↑", "P808 ↑", "BAK ↑", "UTMOS ↑"],
            noisy_rows,
        ),
        "",
        f"Noisy GNR changes adaptive-CSG WER by **{100 * (noisy_gnr['WER'] - adaptive['WER']):+.2f} pp**, P808 by **{noisy_gnr['P808'] - adaptive['P808']:+.3f}**, and UTMOS by **{noisy_gnr['UTMOS'] - adaptive['UTMOS']:+.3f}**. This is quality recovery without content preservation.",
        "",
        "## Final reading",
        "",
        "- Evidence repair is the robust result: CDCS-5 sharply reduces speaker/content switches in clean and noisy speech.",
        "- CSG is useful inside the grounded generative branch because it reduces unrestricted-decoder drift.",
        "- GNR slightly improves predicted quality but worsens WER, so it is an analysis arm rather than the final endpoint.",
        "- Clean and noisy values are different conditions and must not be averaged or treated as a single pooled benchmark.",
        "",
    ])


def build_tex(clean: list[dict[str, Any]], noisy: list[dict[str, Any]]) -> str:
    def clean_line(row: dict[str, Any]) -> str:
        return (
            f"{tex_escape(str(row['Code']))} & {tex_escape(str(row['Method']))} & "
            f"{row['WER']:.3f} & {row['Content_Switch']:.3f} & {row['Spk_Switch']:.3f} & "
            f"{row['Spk_Margin']:.3f} & {row['P808']:.3f} & {row['Short_Output']:.3f} \\\\"
        )

    def noisy_line(row: dict[str, Any]) -> str:
        return (
            f"{tex_escape(str(row['Code']))} & {tex_escape(str(row['Method']))} & "
            f"{row['WER']:.3f} & {row['Content_Switch']:.3f} & {row['Spk_Switch']:.3f} & "
            f"{row['Spk_Margin']:.3f} & {row['P808']:.3f} & {row['BAK']:.3f} & {row['UTMOS']:.3f} \\\\"
        )

    return "\n".join([
        "% Auto-generated by scripts/tse_selected_evidence/build_final_paper_tables.py",
        "% G4* is post-hoc clean readback with K/R fixed on noisy DEV; no clean TEST tuning.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Clean natural TEST results (6,000 trials). Lower is better for WER and switch rates; higher is better otherwise.}",
        r"\label{tab:clean_test_main}",
        r"\begin{tabular}{clrrrrrr}",
        r"\toprule",
        r"Code & Method & WER$\downarrow$ & CSw$\downarrow$ & SSw$\downarrow$ & SpkM$\uparrow$ & P808$\uparrow$ & Short$\downarrow$ \\",
        r"\midrule",
        *[clean_line(row) for row in clean],
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.98\linewidth}\footnotesize G4* is a post-hoc clean readback of K=20/R=2 selected on noisy DEV. It used no clean TEST tuning and is not part of the original clean confirmatory conjunction.\end{minipage}",
        r"\end{table*}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Natural noisy TEST results (6,000 trials) under the single frozen protocol.}",
        r"\label{tab:noisy_test_main}",
        r"\begin{tabular}{clrrrrrrr}",
        r"\toprule",
        r"Code & Method & WER$\downarrow$ & CSw$\downarrow$ & SSw$\downarrow$ & SpkM$\uparrow$ & P808$\uparrow$ & BAK$\uparrow$ & UTMOS$\uparrow$ \\",
        r"\midrule",
        *[noisy_line(row) for row in noisy],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])


def main() -> int:
    clean = build_clean_rows()
    noisy = build_noisy_rows()
    clean_keys = ("WER", "Content_Switch", "Spk_Switch", "Spk_Margin", "P808", "Short_Output")
    noisy_keys = clean_keys + ("BAK", "UTMOS")
    checks = {
        "clean_rows_exactly_five": len(clean) == 5,
        "noisy_rows_exactly_six": len(noisy) == 6,
        "clean_metrics_finite": all(finite_row(row, clean_keys) for row in clean),
        "noisy_metrics_finite": all(finite_row(row, noisy_keys) for row in noisy),
        "clean_gnr_zero_failures": read_json(CLEAN_GNR_SOURCE).get("failed") == 0,
        "clean_gnr_no_clean_test_tuning": read_json(CLEAN_GNR_PROTOCOL).get("clean_test_used_for_configuration") is False,
        "noisy_test_once": read_json(NOISY_EXECUTION).get("execution_count") == 1,
        "no_post_test_tuning": read_json(NOISY_EXECUTION).get("post_test_tuning") is False,
        "clean_direct_cdcs5_is_best_wer": min(clean, key=lambda row: row["WER"])["Code"] == "D3",
        "clean_gnr_improves_p808_but_worsens_wer": clean[4]["P808"] > clean[3]["P808"] and clean[4]["WER"] > clean[3]["WER"],
        "noisy_gnr_improves_p808_utmos_but_worsens_wer": (
            noisy[-1]["P808"] > noisy[-2]["P808"]
            and noisy[-1]["UTMOS"] > noisy[-2]["UTMOS"]
            and noisy[-1]["WER"] > noisy[-2]["WER"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Final table validation failed: {checks}")

    write_csv(OUTPUT / "CLEAN_TEST_CORE_TABLE.csv", clean)
    write_csv(OUTPUT / "NOISY_TEST_CORE_TABLE.csv", noisy)
    atomic_text(OUTPUT / "FINAL_PAPER_TABLES.md", build_markdown(clean, noisy))
    atomic_text(OUTPUT / "icassp2027_main_tables.tex", build_tex(clean, noisy))
    atomic_json(OUTPUT / "validation.json", {
        "status": "READY_TO_SHARE",
        "checks": checks,
        "sources": {
            "clean_confirmatory": str(CLEAN_SOURCE.relative_to(ROOT)),
            "clean_gnr": str(CLEAN_GNR_SOURCE.relative_to(ROOT)),
            "clean_gnr_protocol": str(CLEAN_GNR_PROTOCOL.relative_to(ROOT)),
            "noisy_main": str(NOISY_SOURCE.relative_to(ROOT)),
            "noisy_validation": str(NOISY_VALIDATION.relative_to(ROOT)),
            "noisy_execution": str(NOISY_EXECUTION.relative_to(ROOT)),
        },
        "clean_gnr_note": "Post-hoc clean readback; K/R fixed on noisy DEV; no clean TEST tuning.",
        "post_test_tuning": False,
    })
    print(json.dumps({"status": "READY_TO_SHARE", "output": str(OUTPUT.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
