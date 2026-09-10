#!/usr/bin/env python3
"""Build and audit the frozen natural and controlled-SNR WHAM datasets.

The natural builder consumes the already-frozen official Libri2Mix metadata.
It uses the official LibriMix transformation functions but writes only the
missing ``noise`` and ``mix_both`` products, leaving clean results untouched.
All work is serial, atomic, fsync'ed, and resume-safe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
LIBRIMIX = WORKSPACE / "LibriMix"
LIBRISPEECH = WORKSPACE / "LibriSpeech"
WHAM = WORKSPACE / "wham_noise"
WAV_ROOT = LIBRIMIX / "Libri2Mix/wav16k/min"
OFFICIAL_MD = LIBRIMIX / "metadata/Libri2Mix"
ANALYSIS = ROOT / "analysis/noisy_wham"
MANIFESTS = ROOT / "manifests/noisy_wham"
STATUS = ANALYSIS / "STATUS.md"
RATE = 16_000
EPS = 1e-10
SNRS = (-5, 0, 5, 10, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("natural", "controlled"):
        child = sub.add_parser(name)
        child.add_argument("--split", choices=("dev", "test"), required=True)
        child.add_argument("--status-every", type=int, default=100)
    sub.add_parser("gender-metadata")
    sub.add_parser("audit")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_snr(value: int) -> str:
    return f"m{abs(value)}" if value < 0 else f"p{value}"


def write_wav(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.wav")
    sf.write(temporary, waveform, RATE, subtype="PCM_16", format="WAV")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_mono(path: Path, dtype: str = "float32") -> np.ndarray:
    values, rate = sf.read(path, dtype=dtype, always_2d=True)
    if int(rate) != RATE:
        raise ValueError(f"expected 16 kHz: {path}")
    waveform = values.mean(axis=1, dtype=np.float64).astype(dtype)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"empty or non-finite waveform: {path}")
    return waveform


def update_status(phase: str, completed: int, total: int, detail: str = "") -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_text(
        STATUS,
        "# Noisy WHAM status\n\n"
        f"- Updated: `{stamp}`\n"
        f"- Phase: `{phase}`\n"
        f"- Progress: `{completed}/{total}`\n"
        f"- Detail: {detail or 'none'}\n"
        "- Heavy models resident: `none`\n"
        "- Noisy TEST model inference: `not started`\n",
    )


def load_official_module():
    path = LIBRIMIX / "scripts/create_librimix_from_metadata.py"
    spec = importlib.util.spec_from_file_location("official_librimix_generation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def natural_manifest_path(split: str) -> Path:
    return MANIFESTS / f"natural_{split}.jsonl"


def source_manifest(split: str) -> Path:
    return ROOT / "manifests" / f"tse_{split}.jsonl"


def official_gender_cohorts(split: str) -> dict[str, str]:
    info = pd.read_csv(OFFICIAL_MD / f"libri2mix_{split}-clean_info.csv")
    genders = {
        str(row["mixture_ID"]): (
            "same" if str(row["speaker_1_sex"]) == str(row["speaker_2_sex"])
            else "different"
        )
        for _, row in info.iterrows()
    }
    if len(genders) != 3000:
        raise ValueError(f"official {split} gender metadata is not 3,000 mixtures")
    return genders


def repair_gender_metadata() -> int:
    """Add official mixture-level gender cohorts without touching waveforms."""
    result = {"status": "COMPLETE", "splits": {}, "test_model_results_used": False}
    for split in ("dev", "test"):
        path = natural_manifest_path(split)
        rows = read_jsonl(path)
        genders = official_gender_cohorts(split)
        mixture_ids = [row["trial_id"].split(":")[1] for row in rows]
        if (
            len(rows) != 6000 or len(set(mixture_ids)) != 3000
            or set(mixture_ids) != set(genders)
        ):
            raise ValueError(f"natural {split} manifest/gender ID mismatch")
        for row, mixture_id in zip(rows, mixture_ids):
            row["gender_cohort"] = genders[mixture_id]
        counts = Counter(row["gender_cohort"] for row in rows)
        if sum(counts.values()) != 6000 or set(counts) != {"same", "different"}:
            raise ValueError(f"natural {split} gender coverage failure: {counts}")
        atomic_jsonl(path, rows)
        result["splits"][split] = {
            "target_trials": len(rows),
            "gender_counts": dict(counts),
            "trial_manifest_sha256": sha256(path),
        }
    atomic_json(ANALYSIS / "natural_gender_metadata_repair.json", result)
    print(json.dumps(result, indent=2))
    return 0


def build_natural(split: str, status_every: int) -> int:
    official = load_official_module()
    mapping_path = OFFICIAL_MD / f"libri2mix_{split}-clean.csv"
    mapping = pd.read_csv(mapping_path, engine="python")
    if len(mapping) != 3000 or mapping["mixture_ID"].nunique() != 3000:
        raise ValueError(f"official {split} mapping is not 3,000 unique mixtures")

    split_root = WAV_ROOT / split
    mix_dir = split_root / "mix_both"
    noise_dir = split_root / "noise"
    mix_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    progress_path = ANALYSIS / "data" / f"natural_{split}_progress.jsonl"
    prior = {}
    if progress_path.is_file():
        for row in read_jsonl(progress_path):
            prior[row["mixture_id"]] = row

    records: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for index, (_, row) in enumerate(mapping.iterrows(), 1):
        mixture_id = str(row["mixture_ID"])
        mix_path = mix_dir / f"{mixture_id}.wav"
        noise_path = noise_dir / f"{mixture_id}.wav"
        saved = prior.get(mixture_id)
        if saved and mix_path.is_file() and noise_path.is_file():
            if (
                sha256(mix_path) == saved.get("mix_both_sha256")
                and sha256(noise_path) == saved.get("noise_sha256")
            ):
                records[mixture_id] = saved
                if index % status_every == 0 or index == len(mapping):
                    update_status(f"natural_{split}", index, len(mapping), "resume validation")
                continue

        mix_id, gains, sources = official.read_sources(
            row, 2, str(LIBRISPEECH), str(WHAM)
        )
        if str(mix_id) != mixture_id:
            raise ValueError(f"official ID changed: {mixture_id} -> {mix_id}")
        transformed = official.transform_sources(sources, RATE, "min", gains)
        if len(transformed) != 3 or len({len(value) for value in transformed}) != 1:
            raise ValueError(f"official transformed length mismatch: {mixture_id}")
        expected_length = int(len(transformed[0]))
        clean_s1 = load_mono(split_root / "s1" / f"{mixture_id}.wav")
        clean_s2 = load_mono(split_root / "s2" / f"{mixture_id}.wav")
        clean_mix = load_mono(split_root / "mix_clean" / f"{mixture_id}.wav")
        if not (len(clean_s1) == len(clean_s2) == len(clean_mix) == expected_length):
            raise ValueError(f"frozen clean length differs from official transform: {mixture_id}")
        quantization_tolerance = 1.1 / 32768.0
        if (
            float(np.max(np.abs(clean_s1 - transformed[0]))) > quantization_tolerance
            or float(np.max(np.abs(clean_s2 - transformed[1]))) > quantization_tolerance
        ):
            raise ValueError(f"frozen clean source is not the official transformed source: {mixture_id}")

        noisy = official.mix(transformed)
        noise = transformed[-1]
        if not np.isfinite(noisy).all() or not np.isfinite(noise).all():
            raise ValueError(f"non-finite official mixture: {mixture_id}")
        write_wav(noise_path, noise)
        write_wav(mix_path, noisy)
        scores = official.compute_snr_list(noisy, transformed)
        speech = transformed[0] + transformed[1]
        speech_to_noise_rms = float(
            10.0 * np.log10(
                (float(np.mean(speech.astype(np.float64) ** 2)) + EPS)
                / (float(np.mean(noise.astype(np.float64) ** 2)) + EPS)
            )
        )
        record = {
            "mixture_id": mixture_id,
            "split": split,
            "mix_both_path": str(mix_path.resolve()),
            "noise_path": str(noise_path.resolve()),
            "source_1_path": str((split_root / "s1" / f"{mixture_id}.wav").resolve()),
            "source_2_path": str((split_root / "s2" / f"{mixture_id}.wav").resolve()),
            "official_wham_path": str((WHAM / str(row["noise_path"])).resolve()),
            "official_noise_gain": float(row["noise_gain"]),
            "num_samples": expected_length,
            "source_1_snr_db": float(scores[0]),
            "source_2_snr_db": float(scores[1]),
            "noise_snr_db": float(scores[2]),
            "speech_to_noise_rms_db": speech_to_noise_rms,
            "mix_both_sha256": sha256(mix_path),
            "noise_sha256": sha256(noise_path),
            "official_generation_semantics": True,
        }
        append_jsonl(progress_path, record)
        records[mixture_id] = record
        if index == 1 or index % status_every == 0 or index == len(mapping):
            elapsed = max(time.monotonic() - started, 1e-6)
            update_status(f"natural_{split}", index, len(mapping), f"{index / elapsed:.2f} mixtures/s")
            print(f"natural_{split}={index}/{len(mapping)} rate={index / elapsed:.2f}/s", flush=True)

    ordered = [records[str(value)] for value in mapping["mixture_ID"]]
    atomic_jsonl(ANALYSIS / "data" / f"natural_{split}_mixtures.jsonl", ordered)
    metric_rows = [{
        "mixture_ID": row["mixture_id"],
        "source_1_SNR": row["source_1_snr_db"],
        "source_2_SNR": row["source_2_snr_db"],
        "noise_SNR": row["noise_snr_db"],
    } for row in ordered]
    mixture_rows = [{
        "mixture_ID": row["mixture_id"],
        "mixture_path": row["mix_both_path"],
        "source_1_path": row["source_1_path"],
        "source_2_path": row["source_2_path"],
        "noise_path": row["noise_path"],
        "length": row["num_samples"],
    } for row in ordered]
    atomic_csv(
        WAV_ROOT / "metadata" / f"metrics_{split}_mix_both.csv",
        ["mixture_ID", "source_1_SNR", "source_2_SNR", "noise_SNR"],
        metric_rows,
    )
    atomic_csv(
        WAV_ROOT / "metadata" / f"mixture_{split}_mix_both.csv",
        ["mixture_ID", "mixture_path", "source_1_path", "source_2_path", "noise_path", "length"],
        mixture_rows,
    )

    base_trials = read_jsonl(source_manifest(split))
    by_mix = {row["mixture_id"]: row for row in ordered}
    genders = official_gender_cohorts(split)
    if set(by_mix) != set(genders):
        raise ValueError(f"natural {split} mixture/gender ID mismatch")
    trials = []
    for trial in base_trials:
        mixture_id = trial["trial_id"].split(":")[1]
        source = by_mix[mixture_id]
        item = dict(trial)
        item["clean_mixture_wav"] = item["mixture_wav"]
        item["mixture_wav"] = source["mix_both_path"]
        item.update({
            "noise_wav": source["noise_path"],
            "official_wham_path": source["official_wham_path"],
            "official_noise_gain": source["official_noise_gain"],
            "natural_speech_to_noise_rms_db_evaluation_only": source["speech_to_noise_rms_db"],
            "gender_cohort": genders[mixture_id],
            "noisy_dataset": "official_librimix_mix_both",
        })
        trials.append(item)
    if len(trials) != 6000:
        raise ValueError("natural target manifest must have 6,000 trials")
    atomic_jsonl(natural_manifest_path(split), trials)
    atomic_json(ANALYSIS / "data" / f"natural_{split}_summary.json", {
        "status": "COMPLETE",
        "split": split,
        "mixtures": len(ordered),
        "target_trials": len(trials),
        "resumed_mixtures": sum(value in prior for value in records),
        "elapsed_seconds": time.monotonic() - started,
        "test_model_results_used": False,
    })
    return 0


def clean_mixture_labels(split: str) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    audits = read_jsonl(ROOT / f"analysis/wesep_speaker_selection/{split}_all_trials.jsonl")
    wrong: dict[str, list[bool]] = defaultdict(list)
    for row in audits:
        wrong[row["trial_id"].split(":")[1]].append(bool(row["high_confidence_wrong"]))
    if len(wrong) != 3000 or any(len(value) != 2 for value in wrong.values()):
        raise ValueError(f"invalid frozen clean cohort source: {split}")
    labels = {key: ("swap" if any(value) else "correct") for key, value in wrong.items()}
    genders = official_gender_cohorts(split)
    clean_md = pd.read_csv(WAV_ROOT / "metadata" / f"mixture_{split}_mix_clean.csv")
    lengths = {str(row["mixture_ID"]): int(row["length"]) for _, row in clean_md.iterrows()}
    if not (set(labels) == set(genders) == set(lengths)):
        raise ValueError(f"clean stratum source ID mismatch: {split}")
    return labels, genders, lengths


def hamilton(counts: dict[tuple[str, int], int], total: int) -> dict[tuple[str, int], int]:
    population = sum(counts.values())
    ideal = {key: total * value / population for key, value in counts.items()}
    allocation = {key: int(math.floor(value)) for key, value in ideal.items()}
    remaining = total - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(ideal[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != total or any(allocation[key] > counts[key] for key in counts):
        raise ValueError("invalid Hamilton allocation")
    return allocation


def select_controlled_ids(split: str) -> tuple[list[str], dict[str, Any]]:
    labels, genders, lengths = clean_mixture_labels(split)
    edges = np.quantile(np.asarray(list(lengths.values()), dtype=np.float64), [0.25, 0.5, 0.75])
    bins = {
        key: int(np.searchsorted(edges, value, side="left"))
        for key, value in lengths.items()
    }
    strata: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for mixture_id in sorted(labels):
        strata[(labels[mixture_id], genders[mixture_id], bins[mixture_id])].append(mixture_id)
    rng = random.Random(1986)
    selected: list[str] = []
    allocations: dict[str, int] = {}
    for label in ("swap", "correct"):
        counts = {
            (gender, duration_bin): len(ids)
            for (source_label, gender, duration_bin), ids in strata.items()
            if source_label == label
        }
        allocation = hamilton(counts, 120)
        for key in sorted(allocation):
            count = allocation[key]
            ids = sorted(strata[(label, key[0], key[1])])
            chosen = rng.sample(ids, count)
            selected.extend(chosen)
            allocations[f"{label}:{key[0]}:duration_q{key[1] + 1}"] = count
    if len(selected) != 240 or len(set(selected)) != 240:
        raise ValueError("controlled selection is not 240 unique mixtures")
    selected = sorted(selected)
    summary = {
        "split": split,
        "seed": 1986,
        "duration_quartile_edges_samples": [float(value) for value in edges],
        "allocations": allocations,
        "label_counts": dict(Counter(labels[value] for value in selected)),
        "gender_counts": dict(Counter(genders[value] for value in selected)),
        "ordered_mixture_ids": selected,
    }
    return selected, summary


def official_noise_segment(path: Path, length: int, official: Any) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if int(rate) != RATE:
        raise ValueError(f"WHAM recording is not 16 kHz: {path}")
    noise = values[:, 0]
    if len(noise) < length:
        noise = official.extend_noise(noise, length)
    return np.asarray(noise[:length], dtype=np.float32)


def active_rms_snr(speech: np.ndarray, noise: np.ndarray) -> tuple[float, float]:
    frame = int(0.02 * RATE)
    usable = (len(speech) // frame) * frame
    if usable == 0:
        mask = np.ones(len(speech), dtype=bool)
        active_fraction = 1.0
    else:
        framed = speech[:usable].reshape(-1, frame).astype(np.float64)
        powers = np.mean(framed ** 2, axis=1)
        threshold = float(np.max(powers)) * 1e-4
        active_frames = powers >= threshold
        sample_mask = np.repeat(active_frames, frame)
        mask = np.zeros(len(speech), dtype=bool)
        mask[:usable] = sample_mask
        if not bool(mask.any()):
            mask[:] = True
        active_fraction = float(np.mean(mask))
    speech_power = float(np.mean(speech[mask].astype(np.float64) ** 2))
    noise_power = float(np.mean(noise[mask].astype(np.float64) ** 2))
    return float(10.0 * np.log10((speech_power + EPS) / (noise_power + EPS))), active_fraction


def build_controlled(split: str, status_every: int) -> int:
    official = load_official_module()
    selected, selection = select_controlled_ids(split)
    selection_path = ANALYSIS / "data" / f"controlled_{split}_selection.json"
    atomic_json(selection_path, selection)
    mapping_df = pd.read_csv(OFFICIAL_MD / f"libri2mix_{split}-clean.csv", engine="python")
    mapping = {str(row["mixture_ID"]): row for _, row in mapping_df.iterrows()}
    base_trials = read_jsonl(source_manifest(split))
    trials_by_mix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_trials:
        trials_by_mix[row["trial_id"].split(":")[1]].append(row)
    labels, genders, lengths = clean_mixture_labels(split)
    progress_path = ANALYSIS / "data" / f"controlled_{split}_progress.jsonl"
    prior: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        for row in read_jsonl(progress_path):
            prior[f"{row['mixture_id']}:{row['snr_db']}"] = row

    meter = pyln.Meter(RATE)
    records: dict[str, dict[str, Any]] = {}
    tasks = [(snr, mixture_id) for snr in SNRS for mixture_id in selected]
    started = time.monotonic()
    for index, (snr, mixture_id) in enumerate(tasks, 1):
        key = f"{mixture_id}:{snr}"
        stem_dir = WAV_ROOT / f"controlled_{split}" / f"snr_{safe_snr(snr)}"
        paths = {
            name: stem_dir / name / f"{mixture_id}.wav"
            for name in ("mix_both", "s1", "s2", "noise")
        }
        saved = prior.get(key)
        if saved and all(path.is_file() for path in paths.values()):
            if all(sha256(paths[name]) == saved["waveform_sha256"][name] for name in paths):
                records[key] = saved
                if index % status_every == 0 or index == len(tasks):
                    update_status(f"controlled_{split}", index, len(tasks), "resume validation")
                continue

        s1 = load_mono(WAV_ROOT / split / "s1" / f"{mixture_id}.wav")
        s2 = load_mono(WAV_ROOT / split / "s2" / f"{mixture_id}.wav")
        length = min(len(s1), len(s2))
        if length != lengths[mixture_id]:
            raise ValueError(f"controlled clean length mismatch: {mixture_id}")
        s1 = s1[:length]
        s2 = s2[:length]
        speech = s1 + s2
        official_row = mapping[mixture_id]
        wham_path = WHAM / str(official_row["noise_path"])
        noise = official_noise_segment(wham_path, length, official)
        speech_lufs = float(meter.integrated_loudness(speech))
        noise_lufs = float(meter.integrated_loudness(noise))
        if not math.isfinite(speech_lufs) or not math.isfinite(noise_lufs):
            raise ValueError(f"non-finite integrated loudness: {mixture_id}")
        noise_gain = float(10.0 ** ((speech_lufs - noise_lufs - snr) / 20.0))
        scaled_noise = noise * noise_gain
        noisy = speech + scaled_noise
        peak = max(
            float(np.max(np.abs(noisy))),
            float(np.max(np.abs(s1))),
            float(np.max(np.abs(s2))),
            float(np.max(np.abs(scaled_noise))),
        )
        anti_clip = min(1.0, 0.99 / peak) if peak > 0 else 1.0
        saved_values = {
            "mix_both": noisy * anti_clip,
            "s1": s1 * anti_clip,
            "s2": s2 * anti_clip,
            "noise": scaled_noise * anti_clip,
        }
        if any(not np.isfinite(value).all() for value in saved_values.values()):
            raise ValueError(f"non-finite controlled waveform: {mixture_id} {snr}")
        for name, path in paths.items():
            write_wav(path, saved_values[name])
        achieved_lufs = float(
            meter.integrated_loudness(saved_values["s1"] + saved_values["s2"])
            - meter.integrated_loudness(saved_values["noise"])
        )
        achieved_active_rms, active_fraction = active_rms_snr(
            saved_values["s1"] + saved_values["s2"], saved_values["noise"]
        )
        record = {
            "mixture_id": mixture_id,
            "split": split,
            "snr_db": snr,
            "clean_label": labels[mixture_id],
            "gender": genders[mixture_id],
            "num_samples": length,
            "speech_lufs_before_scaling": speech_lufs,
            "noise_lufs_before_scaling": noise_lufs,
            "noise_gain": noise_gain,
            "achieved_lufs_snr_db": achieved_lufs,
            "achieved_active_rms_snr_db": achieved_active_rms,
            "active_speech_sample_fraction": active_fraction,
            "active_rms_definition": "20 ms nonoverlap speech frames within 40 dB of maximum frame power",
            "official_wham_path": str(wham_path.resolve()),
            "noise_offset_samples": 0,
            "anti_clipping_gain": anti_clip,
            "waveform_paths": {name: str(path.resolve()) for name, path in paths.items()},
            "waveform_sha256": {name: sha256(path) for name, path in paths.items()},
        }
        append_jsonl(progress_path, record)
        records[key] = record
        if index == 1 or index % status_every == 0 or index == len(tasks):
            elapsed = max(time.monotonic() - started, 1e-6)
            update_status(f"controlled_{split}", index, len(tasks), f"{index / elapsed:.2f} mixtures/s")
            print(f"controlled_{split}={index}/{len(tasks)} rate={index / elapsed:.2f}/s", flush=True)

    ordered = [records[f"{mixture_id}:{snr}"] for snr in SNRS for mixture_id in selected]
    atomic_jsonl(ANALYSIS / "data" / f"controlled_{split}_mixtures.jsonl", ordered)
    controlled_trials = []
    for record in ordered:
        mixture_id = record["mixture_id"]
        for trial in trials_by_mix[mixture_id]:
            source_name = Path(trial["target_wav"]).parent.name
            if source_name not in {"s1", "s2"}:
                raise ValueError(f"cannot resolve target direction: {trial['trial_id']}")
            direction = int(source_name[-1])
            item = dict(trial)
            item.update({
                "base_trial_id": trial["trial_id"],
                "trial_id": f"controlled_{split}:{mixture_id}:{trial['target_speaker']}:snr{record['snr_db']:+d}",
                "clean_mixture_wav": trial["mixture_wav"],
                "mixture_wav": record["waveform_paths"]["mix_both"],
                "target_wav": record["waveform_paths"]["s1" if direction == 1 else "s2"],
                "interferer_wavs": [record["waveform_paths"]["s2" if direction == 1 else "s1"]],
                "noise_wav": record["waveform_paths"]["noise"],
                "official_wham_path": record["official_wham_path"],
                "noise_offset_samples": 0,
                "construction_snr_db_evaluation_only": record["snr_db"],
                "achieved_lufs_snr_db_evaluation_only": record["achieved_lufs_snr_db"],
                "achieved_active_rms_snr_db_evaluation_only": record["achieved_active_rms_snr_db"],
                "global_anti_clipping_gain": record["anti_clipping_gain"],
                "controlled_clean_label": record["clean_label"],
                "gender_cohort": record["gender"],
                "target_direction": f"s{direction}",
                "noisy_dataset": "controlled_wham_lufs",
            })
            controlled_trials.append(item)
    expected = 240 * 5 * 2
    if len(controlled_trials) != expected or len({row["trial_id"] for row in controlled_trials}) != expected:
        raise ValueError("controlled target manifest coverage failure")
    atomic_jsonl(MANIFESTS / f"controlled_{split}.jsonl", controlled_trials)
    atomic_json(ANALYSIS / "data" / f"controlled_{split}_summary.json", {
        "status": "COMPLETE",
        "split": split,
        "mixtures": 240,
        "mixture_snr_waveforms": len(ordered),
        "target_trials": len(controlled_trials),
        "snr_levels": list(SNRS),
        "selection_sha256": sha256(selection_path),
        "elapsed_seconds": time.monotonic() - started,
        "test_model_results_used": False,
    })
    return 0


def audit_wave(path: Path, expected_frames: int) -> list[str]:
    errors = []
    if not path.is_file():
        return [f"missing:{path}"]
    info = sf.info(path)
    if int(info.samplerate) != RATE:
        errors.append(f"rate:{path}:{info.samplerate}")
    if int(info.frames) != expected_frames:
        errors.append(f"length:{path}:{info.frames}!={expected_frames}")
    values = load_mono(path)
    if not np.isfinite(values).all():
        errors.append(f"nonfinite:{path}")
    return errors


def audit() -> int:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "status": "IN_PROGRESS",
        "sample_rate_hz": RATE,
        "natural": {},
        "controlled": {},
        "duplicates": 0,
        "missing": 0,
        "non_finite": 0,
        "test_model_results_used": False,
    }
    for split in ("dev", "test"):
        natural_rows = read_jsonl(ANALYSIS / "data" / f"natural_{split}_mixtures.jsonl")
        natural_trials = read_jsonl(natural_manifest_path(split))
        ids = [row["mixture_id"] for row in natural_rows]
        if len(natural_rows) != 3000 or len(set(ids)) != 3000:
            errors.append(f"natural_{split}_mixture_coverage")
        trial_counts = Counter(row["trial_id"].split(":")[1] for row in natural_trials)
        if len(natural_trials) != 6000 or any(value != 2 for value in trial_counts.values()):
            errors.append(f"natural_{split}_target_direction_coverage")
        for row in natural_rows:
            errors.extend(audit_wave(Path(row["mix_both_path"]), int(row["num_samples"])))
            errors.extend(audit_wave(Path(row["noise_path"]), int(row["num_samples"])))
        for row in natural_trials:
            if str(row["target_speaker"]) != row["trial_id"].split(":")[-1]:
                errors.append(f"natural_{split}_target_orientation:{row['trial_id']}")
            enrollment_speaker = str(row["enrollment_utterance"]).split("-")[0]
            if enrollment_speaker != str(row["target_speaker"]):
                errors.append(f"natural_{split}_enrollment_speaker:{row['trial_id']}")
            if row.get("gender_cohort") not in {"same", "different"}:
                errors.append(f"natural_{split}_gender:{row['trial_id']}")
        gender_counts = Counter(row.get("gender_cohort") for row in natural_trials)
        if sum(gender_counts[value] for value in ("same", "different")) != 6000:
            errors.append(f"natural_{split}_gender_coverage")
        summary["natural"][split] = {
            "mixtures": len(natural_rows),
            "target_trials": len(natural_trials),
            "unique_mixture_ids": len(set(ids)),
            "two_directions_per_mixture": all(value == 2 for value in trial_counts.values()),
            "gender_counts": dict(gender_counts),
            "mixture_manifest_sha256": sha256(ANALYSIS / "data" / f"natural_{split}_mixtures.jsonl"),
            "trial_manifest_sha256": sha256(natural_manifest_path(split)),
        }

        controlled_rows = read_jsonl(ANALYSIS / "data" / f"controlled_{split}_mixtures.jsonl")
        controlled_trials = read_jsonl(MANIFESTS / f"controlled_{split}.jsonl")
        keys = [(row["mixture_id"], row["snr_db"]) for row in controlled_rows]
        per_snr = Counter(row["snr_db"] for row in controlled_rows)
        trial_snr = Counter(row["construction_snr_db_evaluation_only"] for row in controlled_trials)
        if len(controlled_rows) != 1200 or len(set(keys)) != 1200 or any(per_snr[value] != 240 for value in SNRS):
            errors.append(f"controlled_{split}_mixture_coverage")
        if len(controlled_trials) != 2400 or len({row['trial_id'] for row in controlled_trials}) != 2400:
            errors.append(f"controlled_{split}_trial_coverage")
        if any(trial_snr[value] != 480 for value in SNRS):
            errors.append(f"controlled_{split}_snr_trial_coverage")
        lufs_errors = []
        for row in controlled_rows:
            for name, value in row["waveform_paths"].items():
                errors.extend(audit_wave(Path(value), int(row["num_samples"])))
                if sha256(Path(value)) != row["waveform_sha256"][name]:
                    errors.append(f"controlled_{split}_hash:{row['mixture_id']}:{row['snr_db']}:{name}")
            lufs_errors.append(abs(float(row["achieved_lufs_snr_db"]) - float(row["snr_db"])))
        summary["controlled"][split] = {
            "selected_mixtures": 240,
            "mixture_snr_waveforms": len(controlled_rows),
            "target_trials": len(controlled_trials),
            "mixtures_per_snr": dict(per_snr),
            "trials_per_snr": dict(trial_snr),
            "maximum_absolute_lufs_snr_error_db": max(lufs_errors),
            "mixture_manifest_sha256": sha256(ANALYSIS / "data" / f"controlled_{split}_mixtures.jsonl"),
            "trial_manifest_sha256": sha256(MANIFESTS / f"controlled_{split}.jsonl"),
        }

    summary["errors"] = errors[:200]
    summary["error_count"] = len(errors)
    summary["duplicates"] = sum("coverage" in value and "unique" in value for value in errors)
    summary["missing"] = sum(value.startswith("missing:") for value in errors)
    summary["non_finite"] = sum("nonfinite:" in value for value in errors)
    summary["status"] = "PASS" if not errors else "FAIL"
    atomic_json(ANALYSIS / "data_audit.json", summary)

    def natural_line(split: str) -> str:
        row = summary["natural"][split]
        return f"| {split.upper()} | {row['mixtures']} | {row['target_trials']} | {row['unique_mixture_ids']} | {row['two_directions_per_mixture']} |"

    def controlled_line(split: str) -> str:
        row = summary["controlled"][split]
        return f"| {split.upper()} | {row['selected_mixtures']} | {row['mixture_snr_waveforms']} | {row['target_trials']} | {row['maximum_absolute_lufs_snr_error_db']:.6f} |"

    report = f"""# Noisy WHAM data audit

## Verdict

**{summary['status']}**. This audit covers waveform construction and manifest integrity only. No WeSep, Qwen-TSE, CosyVoice synthesis, ASR, or noisy TEST model result was loaded.

The official Libri2Mix `mix_both` waveforms were generated from the existing frozen official per-mixture source/noise/gain metadata. Existing clean `mix_clean`, `s1`, and `s2` files were not overwritten. Controlled-SNR waveforms use the preregistered 240-mixture split-specific samples and the same official WHAM recording/segment at all five SNRs.

## Natural official `mix_both`

| Split | Mixtures | Target trials | Unique IDs | Two directions each |
|---|---:|---:|---:|---:|
{natural_line('dev')}
{natural_line('test')}

## Controlled-SNR suite

| Split | Selected mixtures | Mixture×SNR waveforms | Target trials | Max |LUFS error| dB |
|---|---:|---:|---:|---:|
{controlled_line('dev')}
{controlled_line('test')}

Each SNR bin contains 240 mixture waveforms and 480 target directions. The active-RMS audit uses non-overlapping 20 ms speech frames within 40 dB of the maximum frame power; it is diagnostic only and does not set the noise gain.

## Integrity checks

- Sample rate: 16 kHz.
- Natural min-mode lengths equal the frozen clean mixture lengths.
- Controlled mixture, both sources, and noise share one length and one anti-clipping gain.
- Target direction and enrollment speaker identity are checked against frozen trial IDs.
- Missing files: **{summary['missing']}**.
- Duplicate/coverage errors: **{summary['duplicates']}**.
- Non-finite waveforms: **{summary['non_finite']}**.
- Total audit errors: **{summary['error_count']}**.

Machine-readable hashes and per-mixture construction values are under `analysis/noisy_wham/data/`; target-trial manifests are under `manifests/noisy_wham/`.
"""
    atomic_text(ROOT / "docs/NOISY_WHAM_DATA_AUDIT.md", report)
    update_status("data_audit", 1, 1, summary["status"])
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not errors else 2


def main() -> int:
    args = parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    if args.command == "natural":
        return build_natural(args.split, args.status_every)
    if args.command == "controlled":
        return build_controlled(args.split, args.status_every)
    if args.command == "gender-metadata":
        return repair_gender_metadata()
    return audit()


if __name__ == "__main__":
    raise SystemExit(main())
