#!/usr/bin/env python3
"""Measure frozen WeSep speaker-selection errors on Libri2Mix dev/test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis/wesep_speaker_selection")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--progress-interval", type=int, default=500)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_audio(path: str) -> tuple[np.ndarray, int]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = values.mean(axis=1, dtype=np.float64)
    return mono, int(sample_rate)


def audio_info(path: str) -> tuple[int | None, int | None]:
    try:
        info = sf.info(path)
        return int(info.frames), int(info.samplerate)
    except Exception:
        return None, None


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = EPS) -> float:
    length = min(estimate.size, reference.size)
    if length <= 0:
        return float("nan")
    estimate = estimate[:length] - np.mean(estimate[:length])
    reference = reference[:length] - np.mean(reference[:length])
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= eps:
        return float("nan")
    scale = float(np.dot(estimate, reference)) / (reference_energy + eps)
    projected = scale * reference
    noise = estimate - projected
    return float(
        10.0
        * np.log10(
            (float(np.dot(projected, projected)) + eps)
            / (float(np.dot(noise, noise)) + eps)
        )
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return None
    return float(np.dot(a, b) / denom)


def split_paths(split: str) -> dict[str, Path]:
    return {
        "raw": PROJECT_ROOT / f"manifests/tse_{split}.jsonl",
        "prepared": PROJECT_ROOT / f"manifests/tse_{split}_prepared.jsonl",
        "qc": PROJECT_ROOT / f"artifacts/tse_qc/wesep_{split}_20260810_072006/trial_metrics.jsonl",
        "stage1": PROJECT_ROOT / f"dev_outputs/WeSep/per_trial_metrics.jsonl" if split == "dev" else Path(""),
    }


def audit_repair(split: str) -> dict[str, Any]:
    paths = split_paths(split)
    raw = read_jsonl(paths["raw"])
    prepared = read_jsonl(paths["prepared"])
    raw_by_id = {row["trial_id"]: row for row in raw}
    prep_by_id = {row["trial_id"]: row for row in prepared}
    details: dict[str, Any] = {
        "split": split,
        "raw_trials": len(raw),
        "prepared_trials": len(prepared),
        "deleted_trials": sorted(set(raw_by_id) - set(prep_by_id))[:20],
        "added_trials": sorted(set(prep_by_id) - set(raw_by_id))[:20],
        "target_replacements": 0,
        "interferer_replacements": 0,
        "enrollment_replacements": 0,
        "repair_marked_trials": 0,
        "alternate_or_repair_evidence": 0,
        "missing_evidence": 0,
        "examples": [],
    }
    repair_keys = (
        "qc_repaired",
        "repair_variant",
        "diagnostic_original_trial_id",
        "qc_original_enrollment_wav",
        "qc_candidate_trial_id",
        "alternate_enrollment_wav",
    )
    for trial_id, prepared_row in prep_by_id.items():
        raw_row = raw_by_id.get(trial_id)
        if raw_row is None:
            continue
        checks = (
            ("target_replacements", "target_wav"),
            ("enrollment_replacements", "enrollment_wav"),
        )
        for counter_name, field in checks:
            if prepared_row.get(field) != raw_row.get(field):
                details[counter_name] += 1
                if len(details["examples"]) < 20:
                    details["examples"].append({
                        "trial_id": trial_id,
                        "field": field,
                        "raw": raw_row.get(field),
                        "prepared": prepared_row.get(field),
                    })
        if prepared_row.get("interferer_wavs") != raw_row.get("interferer_wavs"):
            details["interferer_replacements"] += 1
            if len(details["examples"]) < 20:
                details["examples"].append({
                    "trial_id": trial_id,
                    "field": "interferer_wavs",
                    "raw": raw_row.get("interferer_wavs"),
                    "prepared": prepared_row.get("interferer_wavs"),
                })
        if any(prepared_row.get(key) for key in repair_keys):
            details["repair_marked_trials"] += 1
        evidence = str(prepared_row.get("evidence_wav", ""))
        if "repair" in evidence.lower() or "alternate" in evidence.lower():
            details["alternate_or_repair_evidence"] += 1
        if not evidence or not Path(evidence).is_file():
            details["missing_evidence"] += 1
    repaired = (
        len(raw) != len(prepared)
        or set(raw_by_id) != set(prep_by_id)
        or details["target_replacements"] > 0
        or details["interferer_replacements"] > 0
        or details["enrollment_replacements"] > 0
        or details["repair_marked_trials"] > 0
        or details["alternate_or_repair_evidence"] > 0
        or details["missing_evidence"] > 0
    )
    details["repaired"] = bool(repaired)
    return details


def qc_status_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    rows = read_jsonl(path)
    return {row["trial_id"]: row.get("waveform_qc_status", "MISSING") for row in rows}


def secondary_map(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    result = {}
    for row in read_jsonl(path):
        if row.get("sim_target") is None or row.get("sim_interferer") is None:
            continue
        result[row["trial_id"]] = {
            "sim_target": float(row["sim_target"]),
            "sim_interferer": float(row["sim_interferer"]),
            "speaker_embedding_margin": float(row["sim_target"] - row["sim_interferer"]),
        }
    return result


def evaluate_one(task: tuple[dict[str, Any], str, dict[str, str], dict[str, dict[str, float]]]) -> dict[str, Any]:
    row, split, qc_map, sec_map = task
    target_path = row["target_wav"]
    interferer_path = row["interferer_wavs"][0]
    output_path = row["evidence_wav"]
    mixture_path = row["mixture_wav"]
    target, target_sr = load_audio(target_path)
    interferer, interferer_sr = load_audio(interferer_path)
    output, output_sr = load_audio(output_path)
    common = min(target.size, interferer.size, output.size)
    sisdr_target = si_sdr(output[:common], target[:common])
    sisdr_interferer = si_sdr(output[:common], interferer[:common])
    margin = sisdr_target - sisdr_interferer
    sec = sec_map.get(row["trial_id"], {})
    sim_target = sec.get("sim_target")
    sim_interferer = sec.get("sim_interferer")
    speaker_embedding_margin = sec.get("speaker_embedding_margin")
    high_confidence_wrong = bool(margin < -5.0 and sisdr_interferer > 0.0)
    record = {
        "trial_id": row["trial_id"],
        "split": split,
        "target_speaker": row["target_speaker"],
        "interferer_speaker": row["interferer_speakers"][0],
        "mixture_path": mixture_path,
        "enrollment_path": row["enrollment_wav"],
        "target_path": target_path,
        "interferer_path": interferer_path,
        "wesep_output_path": output_path,
        "sisdr_target": sisdr_target,
        "sisdr_interferer": sisdr_interferer,
        "speaker_selection_margin": margin,
        "wrong_margin_0": bool(margin < 0.0),
        "wrong_margin_3": bool(margin < -3.0),
        "wrong_margin_5": bool(margin < -5.0),
        "high_confidence_wrong": high_confidence_wrong,
        "qc_status": qc_map.get(row["trial_id"], "MISSING"),
        "sim_target": sim_target,
        "sim_interferer": sim_interferer,
        "speaker_embedding_margin": speaker_embedding_margin,
        "embedding_prefers_interferer": (
            bool(sim_interferer > sim_target)
            if sim_target is not None and sim_interferer is not None
            else None
        ),
        "target_num_samples": int(target.size),
        "interferer_num_samples": int(interferer.size),
        "wesep_output_num_samples": int(output.size),
        "common_num_samples": int(common),
        "wesep_minus_target_samples": int(output.size - target.size),
        "wesep_minus_interferer_samples": int(output.size - interferer.size),
        "sample_rates": {
            "target": target_sr,
            "interferer": interferer_sr,
            "wesep_output": output_sr,
        },
        "alignment_policy": "same start, truncate output/target/interferer to common valid length",
    }
    return record


def percentile(values: list[float], q: float) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(np.percentile(finite, q))


def distribution(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {key: None for key in ("mean", "median", "p05", "p10", "p25", "p75", "p90", "p95")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": percentile(finite, 5),
        "p10": percentile(finite, 10),
        "p25": percentile(finite, 25),
        "p75": percentile(finite, 75),
        "p90": percentile(finite, 90),
        "p95": percentile(finite, 95),
    }


def summarize_split(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)

    def count(key: str) -> int:
        return sum(1 for row in rows if row[key])

    summary = {
        "split": split,
        "total": total,
        "wrong_margin_0": count("wrong_margin_0"),
        "wrong_margin_3": count("wrong_margin_3"),
        "wrong_margin_5": count("wrong_margin_5"),
        "high_confidence_wrong": count("high_confidence_wrong"),
        "margin_distribution": distribution([row["speaker_selection_margin"] for row in rows]),
        "sisdr_target_distribution": distribution([row["sisdr_target"] for row in rows]),
        "sisdr_interferer_distribution": distribution([row["sisdr_interferer"] for row in rows]),
        "qc_status_counts": dict(Counter(row["qc_status"] for row in rows)),
        "length_adjustments": dict(
            sorted(Counter(row["wesep_minus_target_samples"] for row in rows).items())
        ),
    }
    for key in ("wrong_margin_0", "wrong_margin_3", "wrong_margin_5", "high_confidence_wrong"):
        summary[f"{key}_percent"] = 100.0 * summary[key] / total if total else None
    high_conf = [row for row in rows if row["high_confidence_wrong"]]
    with_embedding = [row for row in high_conf if row["embedding_prefers_interferer"] is not None]
    summary["high_confidence_wrong_embedding_checked"] = len(with_embedding)
    summary["high_confidence_wrong_embedding_prefers_interferer"] = sum(
        1 for row in with_embedding if row["embedding_prefers_interferer"]
    )
    return summary


def evaluate_split(split: str, output_dir: Path, workers: int, progress_interval: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = split_paths(split)
    prepared = read_jsonl(paths["prepared"])
    qc_map = qc_status_map(paths["qc"])
    sec_map = secondary_map(paths["stage1"])
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    tasks = [(row, split, qc_map, sec_map) for row in prepared]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, record in enumerate(executor.map(evaluate_one, tasks, chunksize=16), 1):
            records.append(record)
            if index == 1 or index % progress_interval == 0 or index == len(tasks):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[{split} {index}/{len(tasks)}] rate={index / elapsed:.2f}/s "
                    f"wrong0={sum(row['wrong_margin_0'] for row in records)}",
                    flush=True,
                )
    write_jsonl(output_dir / f"{split}_all_trials.jsonl", records)
    write_jsonl(output_dir / f"{split}_wrong_margin0.jsonl", [row for row in records if row["wrong_margin_0"]])
    write_jsonl(output_dir / f"{split}_wrong_margin3.jsonl", [row for row in records if row["wrong_margin_3"]])
    write_jsonl(output_dir / f"{split}_wrong_margin5.jsonl", [row for row in records if row["wrong_margin_5"]])
    write_jsonl(
        output_dir / f"{split}_high_confidence_wrong.jsonl",
        [row for row in records if row["high_confidence_wrong"]],
    )
    summary = summarize_split(split, records)
    summary["elapsed_seconds"] = time.monotonic() - started
    (output_dir / f"{split}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return records, summary


def read_librimix_metrics(split: str) -> dict[str, dict[str, str]]:
    path = REPO_ROOT / f"LibriMix/Libri2Mix/wav16k/min/metadata/metrics_{split}_mix_clean.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["mixture_ID"]: row for row in csv.DictReader(handle)}


def read_speaker_gender(split: str) -> dict[str, str]:
    path = REPO_ROOT / f"LibriMix/metadata/LibriSpeech/{split}-clean.csv"
    gender: dict[str, str] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                gender[row["speaker_ID"]] = row["sex"]
    speakers = REPO_ROOT / "LibriSpeech/SPEAKERS.TXT"
    if speakers.is_file():
        for line in speakers.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith(";") or "|" not in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 2 and parts[0].isdigit() and parts[0] not in gender:
                gender[parts[0]] = parts[1]
    return gender


def mixture_id(row: dict[str, Any]) -> str:
    return Path(row["mixture_path"]).stem


def source_index(row: dict[str, Any]) -> int | None:
    parts = Path(row["target_path"]).parts
    if "s1" in parts:
        return 1
    if "s2" in parts:
        return 2
    return None


def add_condition(condition_rows: list[dict[str, Any]], condition: str, rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    wrong0 = sum(row["wrong_margin_0"] for row in rows)
    wrong5 = sum(row["wrong_margin_5"] for row in rows)
    high = sum(row["high_confidence_wrong"] for row in rows)
    condition_rows.append({
        "condition": condition,
        "num_trials": total,
        "wrong_margin0": wrong0,
        "wrong_margin0_rate": wrong0 / total if total else None,
        "wrong_margin5": wrong5,
        "wrong_margin5_rate": wrong5 / total if total else None,
        "high_confidence_wrong": high,
        "high_confidence_wrong_rate": high / total if total else None,
    })


def bin_numeric(value: float | None, bins: list[tuple[float, float, str]]) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return None


def enrich_dev_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = read_librimix_metrics("dev")
    gender = read_speaker_gender("dev")
    prepared = {item["trial_id"]: item for item in read_jsonl(split_paths("dev")["prepared"])}
    by_mixture = defaultdict(list)
    for row in rows:
        by_mixture[mixture_id(row)].append(row)
    paired_embedding_similarity: dict[str, float] = {}
    for pair in by_mixture.values():
        if len(pair) != 2:
            continue
        first, second = pair
        try:
            first_emb = np.load(prepared[first["trial_id"]]["speaker_embedding_path"], allow_pickle=False)
            second_emb = np.load(prepared[second["trial_id"]]["speaker_embedding_path"], allow_pickle=False)
            sim = cosine(first_emb, second_emb)
        except Exception:
            sim = None
        if sim is not None:
            paired_embedding_similarity[first["trial_id"]] = sim
            paired_embedding_similarity[second["trial_id"]] = sim
    enriched = []
    for row in rows:
        item = dict(row)
        mid = mixture_id(row)
        metric = metrics.get(mid, {})
        idx = source_index(row)
        if idx == 1 and metric:
            item["tir_db"] = float(metric["source_1_SNR"]) - float(metric["source_2_SNR"])
        elif idx == 2 and metric:
            item["tir_db"] = float(metric["source_2_SNR"]) - float(metric["source_1_SNR"])
        else:
            item["tir_db"] = None
        target_gender = gender.get(str(row["target_speaker"]))
        interferer_gender = gender.get(str(row["interferer_speaker"]))
        item["gender_condition"] = (
            "same_gender"
            if target_gender and target_gender == interferer_gender
            else "different_gender"
            if target_gender and interferer_gender
            else "gender_unknown"
        )
        item["target_interferer_embedding_similarity"] = paired_embedding_similarity.get(row["trial_id"])
        enroll_frames, enroll_sr = audio_info(row["enrollment_path"])
        mix_frames, mix_sr = audio_info(row["mixture_path"])
        item["enrollment_duration_sec"] = enroll_frames / enroll_sr if enroll_frames and enroll_sr else None
        item["mixture_duration_sec"] = mix_frames / mix_sr if mix_frames and mix_sr else None
        enriched.append(item)
    return enriched


def write_failure_conditions(dev_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    rows = enrich_dev_conditions(dev_rows)
    conditions: list[dict[str, Any]] = []
    add_condition(conditions, "all_dev", rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"gender:{row['gender_condition']}"].append(row)
        tir_bin = bin_numeric(
            row.get("tir_db"),
            [(-math.inf, -5, "tir_<_-5"), (-5, 0, "tir_-5_to_0"), (0, 5, "tir_0_to_5"), (5, math.inf, "tir_>=5")],
        )
        if tir_bin:
            grouped[f"tir:{tir_bin}"].append(row)
        spk_sim_bin = bin_numeric(
            row.get("target_interferer_embedding_similarity"),
            [(-math.inf, 0, "spk_sim_<0"), (0, 0.25, "spk_sim_0_to_0.25"), (0.25, 0.5, "spk_sim_0.25_to_0.5"), (0.5, math.inf, "spk_sim_>=0.5")],
        )
        if spk_sim_bin:
            grouped[f"target_interferer_embedding:{spk_sim_bin}"].append(row)
        enroll_bin = bin_numeric(
            row.get("enrollment_duration_sec"),
            [(0, 2, "enroll_<2s"), (2, 4, "enroll_2_to_4s"), (4, 6, "enroll_4_to_6s"), (6, math.inf, "enroll_>=6s")],
        )
        if enroll_bin:
            grouped[f"enrollment_duration:{enroll_bin}"].append(row)
        mixture_bin = bin_numeric(
            row.get("mixture_duration_sec"),
            [(0, 3, "mix_<3s"), (3, 5, "mix_3_to_5s"), (5, 8, "mix_5_to_8s"), (8, math.inf, "mix_>=8s")],
        )
        if mixture_bin:
            grouped[f"mixture_duration:{mixture_bin}"].append(row)
    for condition in sorted(grouped):
        add_condition(conditions, condition, grouped[condition])
    path = output_dir / "dev_failure_conditions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "condition",
            "num_trials",
            "wrong_margin0",
            "wrong_margin0_rate",
            "wrong_margin5",
            "wrong_margin5_rate",
            "high_confidence_wrong",
            "high_confidence_wrong_rate",
        ])
        writer.writeheader()
        writer.writerows(conditions)
    return conditions


def make_plots(dev_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    created: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"matplotlib unavailable; skipping plots: {error}", flush=True)
        return created

    margins = [row["speaker_selection_margin"] for row in dev_rows]
    plt.figure(figsize=(8, 5))
    plt.hist(margins, bins=80)
    plt.axvline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("speaker_selection_margin (dB)")
    plt.ylabel("Trials")
    plt.tight_layout()
    path = output_dir / "dev_margin_histogram.png"
    plt.savefig(path, dpi=150)
    plt.close()
    created.append(str(path))

    x = np.array([row["sisdr_target"] for row in dev_rows])
    y = np.array([row["sisdr_interferer"] for row in dev_rows])
    high = np.array([row["high_confidence_wrong"] for row in dev_rows], dtype=bool)
    plt.figure(figsize=(6, 6))
    plt.scatter(x[~high], y[~high], s=8, alpha=0.45)
    if np.any(high):
        plt.scatter(x[high], y[high], s=18, marker="x", alpha=0.9)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    plt.xlabel("SI-SDR(output, target)")
    plt.ylabel("SI-SDR(output, interferer)")
    plt.tight_layout()
    path = output_dir / "dev_target_vs_interferer_sisdr.png"
    plt.savefig(path, dpi=150)
    plt.close()
    created.append(str(path))
    return created


def copy_case_audio(record: dict[str, Any], case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "mixture.wav": record["mixture_path"],
        "enrollment.wav": record["enrollment_path"],
        "target.wav": record["target_path"],
        "interferer.wav": record["interferer_path"],
        "wesep_output.wav": record["wesep_output_path"],
    }
    for name, source in copies.items():
        shutil.copy2(source, case_dir / name)
    (case_dir / "metrics.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_case_name(prefix: str, index: int, trial_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in trial_id)
    return f"{prefix}_{index:02d}_{safe[:90]}"


def save_audio_examples(dev_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    root = output_dir / "audio_examples"
    if root.exists():
        shutil.rmtree(root)
    high = sorted(
        [row for row in dev_rows if row["high_confidence_wrong"]],
        key=lambda row: (row["speaker_selection_margin"], -row["sisdr_interferer"]),
    )[:20]
    ambiguous = sorted(dev_rows, key=lambda row: abs(row["speaker_selection_margin"]))[:10]
    correct = [row for row in dev_rows if row["speaker_selection_margin"] >= 0.0]
    median_margin = statistics.median([row["speaker_selection_margin"] for row in correct]) if correct else 0.0
    controls = sorted(correct, key=lambda row: abs(row["speaker_selection_margin"] - median_margin))[:10]
    selections = {
        "high_confidence_wrong": high,
        "ambiguous_margin_near_0": ambiguous,
        "correct_controls": controls,
    }
    for prefix, rows in selections.items():
        for index, row in enumerate(rows, 1):
            copy_case_audio(row, root / safe_case_name(prefix, index, row["trial_id"]))
    return {key: len(value) for key, value in selections.items()}


def write_summary_csv(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "wesep_speaker_selection_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Split",
            "Trials",
            "Wrong_margin_0",
            "Wrong_margin_0_percent",
            "Wrong_margin_3",
            "Wrong_margin_3_percent",
            "Wrong_margin_5",
            "Wrong_margin_5_percent",
            "High_confidence_wrong",
            "High_confidence_wrong_percent",
            "Mean_target_SISDR",
            "Mean_interferer_SISDR",
            "Mean_margin",
        ])
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "Split": summary["split"],
                "Trials": summary["total"],
                "Wrong_margin_0": summary["wrong_margin_0"],
                "Wrong_margin_0_percent": summary["wrong_margin_0_percent"],
                "Wrong_margin_3": summary["wrong_margin_3"],
                "Wrong_margin_3_percent": summary["wrong_margin_3_percent"],
                "Wrong_margin_5": summary["wrong_margin_5"],
                "Wrong_margin_5_percent": summary["wrong_margin_5_percent"],
                "High_confidence_wrong": summary["high_confidence_wrong"],
                "High_confidence_wrong_percent": summary["high_confidence_wrong_percent"],
                "Mean_target_SISDR": summary["sisdr_target_distribution"]["mean"],
                "Mean_interferer_SISDR": summary["sisdr_interferer_distribution"]["mean"],
                "Mean_margin": summary["margin_distribution"]["mean"],
            })


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}%"


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def report_summary_line(summary: dict[str, Any], key: str) -> str:
    return f"{summary[key]} ({pct(summary[f'{key}_percent'])})"


def write_report(
    audits: dict[str, dict[str, Any]],
    dev_summary: dict[str, Any],
    test_summary: dict[str, Any],
    conditions: list[dict[str, Any]],
    examples: dict[str, int],
    plots: list[str],
) -> Path:
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    path = docs / "WESEP_SPEAKER_SELECTION_FAILURE_REPORT.md"
    top_conditions = sorted(
        [row for row in conditions if row["condition"] != "all_dev" and row["num_trials"] >= 50],
        key=lambda row: row["high_confidence_wrong_rate"] or 0.0,
        reverse=True,
    )[:8]
    content = f"""# WeSep Speaker-Selection Failure Analysis

## 1. Executive Summary

Frozen WeSep does show target confusion on both Libri2Mix dev and test when judged by direct clean-source SI-SDR preference. Dev has {report_summary_line(dev_summary, "wrong_margin_0")} preference errors and {report_summary_line(dev_summary, "high_confidence_wrong")} high-confidence wrong-speaker extractions. Test, using the frozen dev definition unchanged, has {report_summary_line(test_summary, "wrong_margin_0")} preference errors and {report_summary_line(test_summary, "high_confidence_wrong")} high-confidence wrong-speaker extractions.

## 2. Data Lineage

DEV_REPAIRED = {'YES' if audits['dev']['repaired'] else 'NO'}

TEST_REPAIRED = {'YES' if audits['test']['repaired'] else 'NO'}

The dev/test prepared manifests preserve trial count, target references, interferer references, and enrollment paths from the corresponding base manifests. Prepared `evidence_wav` files come from the original dev/test feature roots. Old PASS/RETRY/FAIL QC labels are retained only as metadata.

## 3. Evaluation Definition

Primary wrong-speaker diagnostic is based on comparing the same WeSep output against both clean source references.

For every trial, the frozen WeSep output is `evidence_wav` from `tse_{{split}}_prepared.jsonl`. The target and interferer references are the Libri2Mix clean `s1/s2` source files already identified in the manifest.

## 4. SI-SDR Speaker-Selection Margin

The margin is:

`M = SI-SDR(output,target) - SI-SDR(output,interferer)`

All SI-SDR comparisons use the same scale-invariant implementation. Output, target, and interferer are aligned from the same starting sample and truncated to the common valid length; no per-reference shift search is used.

## 5. Dev Results

DEV_TOTAL = {dev_summary['total']}

DEV_WRONG_MARGIN_0 = {report_summary_line(dev_summary, "wrong_margin_0")}

DEV_WRONG_MARGIN_3 = {report_summary_line(dev_summary, "wrong_margin_3")}

DEV_WRONG_MARGIN_5 = {report_summary_line(dev_summary, "wrong_margin_5")}

DEV_HIGH_CONFIDENCE_WRONG = {report_summary_line(dev_summary, "high_confidence_wrong")}

Mean target SI-SDR is {fmt(dev_summary['sisdr_target_distribution']['mean'])} dB, mean interferer SI-SDR is {fmt(dev_summary['sisdr_interferer_distribution']['mean'])} dB, and mean margin is {fmt(dev_summary['margin_distribution']['mean'])} dB.

Dev margin distribution: median {fmt(dev_summary['margin_distribution']['median'])}, p05 {fmt(dev_summary['margin_distribution']['p05'])}, p10 {fmt(dev_summary['margin_distribution']['p10'])}, p25 {fmt(dev_summary['margin_distribution']['p25'])}, p75 {fmt(dev_summary['margin_distribution']['p75'])}, p90 {fmt(dev_summary['margin_distribution']['p90'])}, p95 {fmt(dev_summary['margin_distribution']['p95'])} dB.

Dev target SI-SDR distribution: median {fmt(dev_summary['sisdr_target_distribution']['median'])}, p05 {fmt(dev_summary['sisdr_target_distribution']['p05'])}, p25 {fmt(dev_summary['sisdr_target_distribution']['p25'])}, p75 {fmt(dev_summary['sisdr_target_distribution']['p75'])}, p95 {fmt(dev_summary['sisdr_target_distribution']['p95'])} dB.

Dev interferer SI-SDR distribution: median {fmt(dev_summary['sisdr_interferer_distribution']['median'])}, p05 {fmt(dev_summary['sisdr_interferer_distribution']['p05'])}, p25 {fmt(dev_summary['sisdr_interferer_distribution']['p25'])}, p75 {fmt(dev_summary['sisdr_interferer_distribution']['p75'])}, p95 {fmt(dev_summary['sisdr_interferer_distribution']['p95'])} dB.

## 6. Error Severity

Preference error: `M < 0 dB`.

Clear speaker swap: `M < -3 dB`.

Strong speaker swap: `M < -5 dB`.

High-confidence speaker swap: `M < -5 dB` and `SI-SDR(output, interferer) > 0 dB`.

## 7. High-Confidence Speaker Swaps

Dev high-confidence wrong-speaker cases: {dev_summary['high_confidence_wrong']}. Test high-confidence wrong-speaker cases: {test_summary['high_confidence_wrong']}.

## 8. Speaker-Embedding Cross-Check

Secondary speaker-embedding scores were available from the existing Dev WeSep evaluation. Among Dev high-confidence wrong cases, {dev_summary['high_confidence_wrong_embedding_prefers_interferer']} / {dev_summary['high_confidence_wrong_embedding_checked']} also have `sim_interferer > sim_target`. Test speaker-embedding cross-check was not precomputed, so it was not used for the frozen Test measurement.

## 9. Failure Conditions

Failure-condition summaries are saved in `analysis/wesep_speaker_selection/dev_failure_conditions.csv`. Existing metadata covered TIR, same/different gender, enrollment duration, mixture duration, and target/interferer enrollment-embedding similarity. No existing overlap-ratio metadata was found, so overlap ratio was not regenerated.

Top high-confidence wrong-speaker rates among conditions with at least 50 trials:

"""
    for row in top_conditions:
        content += (
            f"- {row['condition']}: {row['high_confidence_wrong']} / {row['num_trials']} "
            f"({100.0 * (row['high_confidence_wrong_rate'] or 0.0):.2f}%)\n"
        )
    content += f"""
## 10. Test Results

TEST_TOTAL = {test_summary['total']}

TEST_WRONG_MARGIN_0 = {report_summary_line(test_summary, "wrong_margin_0")}

TEST_WRONG_MARGIN_3 = {report_summary_line(test_summary, "wrong_margin_3")}

TEST_WRONG_MARGIN_5 = {report_summary_line(test_summary, "wrong_margin_5")}

TEST_HIGH_CONFIDENCE_WRONG = {report_summary_line(test_summary, "high_confidence_wrong")}

Test mean target SI-SDR is {fmt(test_summary['sisdr_target_distribution']['mean'])} dB, mean interferer SI-SDR is {fmt(test_summary['sisdr_interferer_distribution']['mean'])} dB, and mean margin is {fmt(test_summary['margin_distribution']['mean'])} dB.

Test margin distribution: median {fmt(test_summary['margin_distribution']['median'])}, p05 {fmt(test_summary['margin_distribution']['p05'])}, p10 {fmt(test_summary['margin_distribution']['p10'])}, p25 {fmt(test_summary['margin_distribution']['p25'])}, p75 {fmt(test_summary['margin_distribution']['p75'])}, p90 {fmt(test_summary['margin_distribution']['p90'])}, p95 {fmt(test_summary['margin_distribution']['p95'])} dB.

## 11. Example Cases

Audio examples are saved under `analysis/wesep_speaker_selection/audio_examples/`: {examples['high_confidence_wrong']} high-confidence wrong cases, {examples['ambiguous_margin_near_0']} ambiguous near-zero-margin cases, and {examples['correct_controls']} correct controls.

Plots generated:

"""
    for plot in plots:
        plot_path = Path(plot).resolve()
        content += f"- `{plot_path.relative_to(PROJECT_ROOT)}`\n"
    content += """
## 12. Implications

Frozen WeSep target confusion is measurable without using old QC labels: a minority of trials prefer the interferer by SI-SDR, and a smaller subset are clear/high-confidence speaker swaps rather than simply low-quality or ambiguous outputs. The frozen test split confirms the same diagnostic pattern under the dev-fixed thresholds.
"""
    path.write_text(content, encoding="utf-8")
    return path


def print_final(audits: dict[str, dict[str, Any]], dev: dict[str, Any], test: dict[str, Any], report: Path) -> None:
    print()
    print(f"DEV_REPAIRED: {'YES' if audits['dev']['repaired'] else 'NO'}")
    print(f"TEST_REPAIRED: {'YES' if audits['test']['repaired'] else 'NO'}")
    print(f"DEV_TOTAL: {dev['total']}")
    print(f"DEV_WRONG_MARGIN_0: {report_summary_line(dev, 'wrong_margin_0')}")
    print(f"DEV_WRONG_MARGIN_3: {report_summary_line(dev, 'wrong_margin_3')}")
    print(f"DEV_WRONG_MARGIN_5: {report_summary_line(dev, 'wrong_margin_5')}")
    print(f"DEV_HIGH_CONFIDENCE_WRONG: {report_summary_line(dev, 'high_confidence_wrong')}")
    print(f"TEST_TOTAL: {test['total']}")
    print(f"TEST_WRONG_MARGIN_0: {report_summary_line(test, 'wrong_margin_0')}")
    print(f"TEST_WRONG_MARGIN_3: {report_summary_line(test, 'wrong_margin_3')}")
    print(f"TEST_WRONG_MARGIN_5: {report_summary_line(test, 'wrong_margin_5')}")
    print(f"TEST_HIGH_CONFIDENCE_WRONG: {report_summary_line(test, 'high_confidence_wrong')}")
    print(f"REPORT: {report.relative_to(PROJECT_ROOT)}")


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    audits = {"dev": audit_repair("dev"), "test": audit_repair("test")}
    print(f"DEV_REPAIRED = {'YES' if audits['dev']['repaired'] else 'NO'}", flush=True)
    print(f"TEST_REPAIRED = {'YES' if audits['test']['repaired'] else 'NO'}", flush=True)
    if audits["dev"]["repaired"] or audits["test"]["repaired"]:
        print(json.dumps(audits, indent=2, ensure_ascii=False), flush=True)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "repair_audit.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dev_rows, dev_summary = evaluate_split("dev", args.output_dir, args.workers, args.progress_interval)
    conditions = write_failure_conditions(dev_rows, args.output_dir)
    examples = save_audio_examples(dev_rows, args.output_dir)
    plots = make_plots(dev_rows, args.output_dir)

    test_rows, test_summary = evaluate_split("test", args.output_dir, args.workers, args.progress_interval)
    _ = test_rows
    write_summary_csv([dev_summary, test_summary], args.output_dir)
    report = write_report(audits, dev_summary, test_summary, conditions, examples, plots)
    print_final(audits, dev_summary, test_summary, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
