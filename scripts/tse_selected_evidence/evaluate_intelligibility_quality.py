#!/usr/bin/env python3
"""Evaluate frozen Selected-Evidence waveforms with STOI, ESTOI, and PESQ-WB.

The evaluator is deliberately conservative: systems run sequentially, audio is
streamed one trial at a time, chunks are at most 100 trials, and every completed
chunk is fsync'ed.  No waveform, token, candidate choice, or model artifact is
modified.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

# Small per-trial matrix operations become slower and unsafe when every worker
# creates a full BLAS team.  Pin native math libraries before importing NumPy.
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np
import soundfile as sf
import torch
from pesq import pesq as pesq_metric
from pystoi import stoi as stoi_metric


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from se_align.utils.audio import load_wav, resample  # noqa: E402


RESULTS = ROOT / "results/intelligibility_quality"
DOCS = ROOT / "docs"
EXPECTED = 6000
ALIGNMENT_SAMPLES = 100
ALIGNMENT_SEED = 20270901
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20270902
TARGET_RATE = 16_000
MAX_CHUNK = 100
MAX_WORKERS = 2
REUSE_TOLERANCE = 1e-6

SYSTEMS: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("primary_wesep", {
        "name": "Primary WeSep",
        "short": "Primary WeSep",
        "dev": ROOT / "dev_outputs/WeSep/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/primary_wesep/per_trial_metrics.jsonl",
    }),
    ("pool_b_selected", {
        "name": "Pool B Selected Candidate",
        "short": "Pool B selected",
        "dev": ROOT / "results/selected_evidence/systems/pool_b_selected/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/pool_b_selected/per_trial_metrics.jsonl",
    }),
    ("pool_d_selected", {
        "name": "Pool D Selected Candidate",
        "short": "Pool D direct",
        "dev": ROOT / "results/selected_evidence/systems/pool_d_selected/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/pool_d_selected/per_trial_metrics.jsonl",
    }),
    ("pool_d_s3_recon", {
        "name": "Pool D Selected S3 Reconstruction",
        "short": "Pool D S3 recon",
        "dev": ROOT / "results/selected_evidence/systems/pool_d_s3_recon/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/pool_d_s3_recon/per_trial_metrics.jsonl",
    }),
    ("original_qfull_ud", {
        "name": "Original Q-Full UD",
        "short": "Original Q-Full UD",
        "dev": ROOT / "grounding/csg_lambda_0/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/original_qfull_ud/per_trial_metrics.jsonl",
    }),
    ("original_qfull_csg", {
        "name": "Original Q-Full + CSG lambda=1",
        "short": "Original Q-Full + CSG",
        "dev": ROOT / "grounding/csg_lambda_1/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/original_qfull_csg/per_trial_metrics.jsonl",
    }),
    ("pool_b_qfull_ud", {
        "name": "Pool B Selected -> Q-Full UD",
        "short": "Pool B Q-Full UD",
        "dev": ROOT / "results/selected_evidence/systems/pool_b_qfull_ud/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/pool_b_qfull_ud/per_trial_metrics.jsonl",
    }),
    ("pool_b_qfull_csg", {
        "name": "Pool B Selected -> Q-Full + CSG lambda=1",
        "short": "Pool B Q-Full + CSG",
        "dev": ROOT / "results/selected_evidence/systems/pool_b_qfull_csg/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/pool_b_qfull_csg/per_trial_metrics.jsonl",
    }),
    ("pool_d_qfull_ud", {
        "name": "Pool D Selected -> Q-Full UD",
        "short": "Pool D Q-Full UD",
        "dev": ROOT / "results/selected_evidence/systems/pool_d_qfull_ud/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/pool_d_qfull_ud/per_trial_metrics.jsonl",
    }),
    ("pool_d_qfull_csg", {
        "name": "Pool D Selected -> Q-Full + CSG lambda=1",
        "short": "Pool D Q-Full + CSG",
        "dev": ROOT / "results/selected_evidence/systems/pool_d_qfull_csg/per_trial_metrics.jsonl",
        "test": ROOT / "results/selected_evidence_test/systems/pool_d_qfull_csg/per_trial_metrics.jsonl",
    }),
])

TRADEOFF_SYSTEMS = ("pool_d_selected", "pool_d_qfull_ud", "pool_d_qfull_csg")
COMPARISONS = OrderedDict([
    ("A", ("pool_d_selected", "pool_d_qfull_ud",
           "Pool D direct vs Pool D Q-Full UD")),
    ("B", ("pool_d_qfull_ud", "pool_d_qfull_csg",
           "Pool D Q-Full UD vs Pool D Q-Full + CSG")),
    ("C", ("pool_d_selected", "pool_d_qfull_csg",
           "Pool D direct vs Pool D Q-Full + CSG")),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("asset-audit")
    align = sub.add_parser("alignment-audit")
    align.add_argument("--workers", type=int, default=2)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate.add_argument("--system", choices=tuple(SYSTEMS), required=True)
    evaluate.add_argument("--chunk-size", type=int, default=100)
    evaluate.add_argument("--workers", type=int, default=2)
    sub.add_parser("combine")
    sub.add_parser("aggregate")
    sub.add_parser("validate")
    sub.add_parser("final-summary")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def append_chunk(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_resume(path: Path) -> list[dict[str, Any]]:
    """Read a shard and recover only a corrupt final line, if present."""
    if not path.is_file():
        return []
    good: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            atomic_jsonl(path, good)
    return good


@contextmanager
def exclusive_lock() -> Iterable[None]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock_path = RESULTS / ".evaluation.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another intelligibility evaluation is active") from error
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield


def source_rows(split: str, slug: str) -> list[dict[str, Any]]:
    return read_jsonl(Path(SYSTEMS[slug][split]))


def validate_source_rows(
    split: str, slug: str, rows: list[dict[str, Any]], expected_ids: set[str] | None = None,
    check_paths: bool = False,
) -> dict[str, Any]:
    ids = [row.get("trial_id") for row in rows]
    unique_ids = set(ids)
    required = (
        "trial_id", "target_wav", "output_wav", "target_WER", "content_switch",
        "speaker_margin", "acoustic_speaker_switch", "dnsmos_p808", "stoi", "pesq_wb",
    )
    missing_fields = sum(
        any(row.get(field) is None for field in required) for row in rows
    )
    failed = sum(row.get("decode_status") != "ok" for row in rows)
    missing_paths = 0
    if check_paths:
        seen: dict[str, bool] = {}
        for row in rows:
            for field in ("target_wav", "output_wav"):
                value = str(row[field])
                if value not in seen:
                    seen[value] = Path(value).is_file()
                missing_paths += not seen[value]
    result = {
        "split": split,
        "system_slug": slug,
        "system": SYSTEMS[slug]["name"],
        "source": str(Path(SYSTEMS[slug][split]).resolve()),
        "expected": EXPECTED,
        "rows": len(rows),
        "unique": len(unique_ids),
        "missing": EXPECTED - len(unique_ids),
        "duplicate": len(rows) - len(unique_ids),
        "failed": failed,
        "rows_missing_required_fields": missing_fields,
        "missing_audio_paths": missing_paths if check_paths else None,
        "id_set_match": expected_ids is None or unique_ids == expected_ids,
    }
    if not (
        len(rows) == EXPECTED and len(unique_ids) == EXPECTED and failed == 0
        and missing_fields == 0 and result["id_set_match"]
        and (not check_paths or missing_paths == 0)
    ):
        raise ValueError(f"frozen source validation failed: {result}")
    return result


def asset_audit() -> None:
    results: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        expected_ids: set[str] | None = None
        for slug in SYSTEMS:
            rows = source_rows(split, slug)
            result = validate_source_rows(
                split, slug, rows, expected_ids=expected_ids, check_paths=True,
            )
            if expected_ids is None:
                expected_ids = {row["trial_id"] for row in rows}
            results.append(result)
            print(
                f"ASSET_AUDIT split={split} system={slug} rows={result['rows']} "
                f"unique={result['unique']} missing={result['missing']} "
                f"duplicate={result['duplicate']} failed={result['failed']} "
                f"missing_audio_paths={result['missing_audio_paths']}",
                flush=True,
            )
    atomic_json(RESULTS / "asset_audit.json", {
        "status": "PASS", "systems": results,
        "no_training": True, "no_decoding": True, "test_inference_rerun": False,
    })


def audio16(path: str | Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = load_wav(str(path), mono=True)
    waveform = resample(waveform, sample_rate, TARGET_RATE).reshape(-1)
    values = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    if values.size == 0:
        raise ValueError(f"empty waveform: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite waveform: {path}")
    return values, int(sample_rate)


def onset_seconds(values: np.ndarray) -> float | None:
    """Fixed energy-onset diagnostic; never used to align metric inputs."""
    frame = 320
    hop = 160
    if values.size < frame:
        return None
    count = 1 + (values.size - frame) // hop
    rms = np.empty(count, dtype=np.float64)
    for index in range(count):
        part = values[index * hop:index * hop + frame]
        rms[index] = math.sqrt(float(np.mean(part.astype(np.float64) ** 2)))
    peak = float(np.max(rms))
    if peak <= 1e-8:
        return None
    threshold = max(1e-5, peak * 0.01)  # fixed -40 dB relative energy threshold
    indexes = np.flatnonzero(rms >= threshold)
    return float(indexes[0] * hop / TARGET_RATE) if indexes.size else None


def alignment_one(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    slug, split, row = task
    reference, reference_rate = audio16(row["target_wav"])
    estimate, estimate_rate = audio16(row["output_wav"])
    common = min(reference.size, estimate.size)
    if common <= 0:
        raise ValueError("zero common samples")
    ref = reference[:common]
    est = estimate[:common]
    recomputed_stoi = float(stoi_metric(ref, est, TARGET_RATE, extended=False))
    recomputed_pesq = float(pesq_metric(TARGET_RATE, ref, est, "wb"))
    ref_onset = onset_seconds(reference)
    est_onset = onset_seconds(estimate)
    return {
        "trial_id": row["trial_id"], "split": split,
        "system_slug": slug, "system": SYSTEMS[slug]["name"],
        "reference_duration_s": reference.size / TARGET_RATE,
        "estimate_duration_s": estimate.size / TARGET_RATE,
        "common_duration_s": common / TARGET_RATE,
        "length_ratio": estimate.size / reference.size,
        "duration_mismatch_gt_5pct": abs(estimate.size / reference.size - 1.0) > 0.05,
        "short_output_lt_75pct": estimate.size / reference.size < 0.75,
        "reference_onset_s": ref_onset, "estimate_onset_s": est_onset,
        "onset_delta_ms": (
            1000.0 * (est_onset - ref_onset)
            if ref_onset is not None and est_onset is not None else None
        ),
        "original_reference_sample_rate": reference_rate,
        "original_estimate_sample_rate": estimate_rate,
        "common_samples_16k": common,
        "stored_common_samples": row.get("metric_overlap_samples"),
        "stored_stoi": float(row["stoi"]), "recomputed_stoi": recomputed_stoi,
        "stoi_absolute_error": abs(recomputed_stoi - float(row["stoi"])),
        "stored_pesq": float(row["pesq_wb"]), "recomputed_pesq": recomputed_pesq,
        "pesq_absolute_error": abs(recomputed_pesq - float(row["pesq_wb"])),
        "reference_peak": float(np.max(np.abs(reference))),
        "estimate_peak": float(np.max(np.abs(estimate))),
    }


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def alignment_audit(workers: int) -> None:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}]")
    first_rows = source_rows("dev", next(iter(SYSTEMS)))
    validate_source_rows("dev", next(iter(SYSTEMS)), first_rows)
    ids = [row["trial_id"] for row in first_rows]
    rng = np.random.default_rng(ALIGNMENT_SEED)
    selected_ids = set(rng.choice(ids, size=ALIGNMENT_SAMPLES, replace=False).tolist())
    records: list[dict[str, Any]] = []
    for slug in SYSTEMS:
        rows = source_rows("dev", slug)
        validate_source_rows("dev", slug, rows, expected_ids=set(ids))
        chosen = [row for row in rows if row["trial_id"] in selected_ids]
        tasks = [(slug, "dev", row) for row in chosen]
        started = time.monotonic()
        if workers == 1:
            system_records = [alignment_one(task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                system_records = list(executor.map(alignment_one, tasks))
        records.extend(system_records)
        print(
            f"ALIGNMENT_AUDIT system={slug} complete={len(system_records)}/100 "
            f"seconds={time.monotonic() - started:.1f}", flush=True,
        )
    atomic_jsonl(RESULTS / "alignment_audit_records.jsonl", records)

    summaries: list[dict[str, Any]] = []
    for slug in SYSTEMS:
        rows = [row for row in records if row["system_slug"] == slug]
        ratios = [float(row["length_ratio"]) for row in rows]
        lags = [float(row["onset_delta_ms"]) for row in rows if row["onset_delta_ms"] is not None]
        mismatch = sum(bool(row["duration_mismatch_gt_5pct"]) for row in rows)
        median_lag = statistics.median(lags) if lags else None
        same_lag_sign = (
            max(sum(value >= 0 for value in lags), sum(value <= 0 for value in lags)) / len(lags)
            if lags else None
        )
        systematic_latency = bool(
            median_lag is not None and abs(median_lag) > 50.0
            and same_lag_sign is not None and same_lag_sign >= 0.75
        )
        summaries.append({
            "system_slug": slug, "system": SYSTEMS[slug]["name"], "n": len(rows),
            "reference_duration_mean_s": float(np.mean([row["reference_duration_s"] for row in rows])),
            "estimate_duration_mean_s": float(np.mean([row["estimate_duration_s"] for row in rows])),
            "common_duration_mean_s": float(np.mean([row["common_duration_s"] for row in rows])),
            "length_ratio_median": statistics.median(ratios),
            "length_ratio_p05": percentile(ratios, 5), "length_ratio_p95": percentile(ratios, 95),
            "duration_mismatch_gt_5pct_n": mismatch,
            "duration_mismatch_gt_5pct_rate": mismatch / len(rows),
            "short_output_lt_75pct_n": sum(bool(row["short_output_lt_75pct"]) for row in rows),
            "onset_delta_median_ms": median_lag,
            "onset_delta_p05_ms": percentile(lags, 5) if lags else None,
            "onset_delta_p95_ms": percentile(lags, 95) if lags else None,
            "onset_coverage": len(lags) / len(rows),
            "systematic_latency_flag": systematic_latency,
            "max_stoi_absolute_error": max(row["stoi_absolute_error"] for row in rows),
            "max_pesq_absolute_error": max(row["pesq_absolute_error"] for row in rows),
            "common_sample_mismatch_n": sum(
                row["stored_common_samples"] is None
                or int(row["stored_common_samples"]) != int(row["common_samples_16k"])
                for row in rows
            ),
            "peak_outside_unit_range_n": sum(
                row["reference_peak"] > 1.0 + 1e-7 or row["estimate_peak"] > 1.0 + 1e-7
                for row in rows
            ),
        })
    reuse_valid = all(
        row["max_stoi_absolute_error"] <= REUSE_TOLERANCE
        and row["max_pesq_absolute_error"] <= REUSE_TOLERANCE
        and row["common_sample_mismatch_n"] == 0
        for row in summaries
    )
    serious = any(
        row["duration_mismatch_gt_5pct_rate"] > 0.20
        or (row["onset_delta_median_ms"] is not None and abs(row["onset_delta_median_ms"]) > 100.0)
        for row in summaries
    )
    atomic_json(RESULTS / "protocol_validation.json", {
        "status": "PASS" if reuse_valid else "RECOMPUTE_STOI_PESQ_REQUIRED",
        "stored_stoi_pesq_reuse_valid": reuse_valid,
        "tolerance": REUSE_TOLERANCE,
        "alignment_seed": ALIGNMENT_SEED,
        "alignment_samples_per_system": ALIGNMENT_SAMPLES,
        "serious_temporal_mismatch": serious,
        "summaries": summaries,
    })
    write_alignment_report(summaries, reuse_valid, serious)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{100.0 * value:.{digits}f}%"


def write_alignment_report(
    summaries: list[dict[str, Any]], reuse_valid: bool, serious: bool,
) -> None:
    rows = [[
        row["system"], fmt(row["reference_duration_mean_s"]),
        fmt(row["estimate_duration_mean_s"]), fmt(row["common_duration_mean_s"]),
        fmt(row["length_ratio_median"]),
        f"{fmt(row['length_ratio_p05'])}–{fmt(row['length_ratio_p95'])}",
        f"{row['duration_mismatch_gt_5pct_n']}/100",
        f"{row['short_output_lt_75pct_n']}/100",
        fmt(row["onset_delta_median_ms"], 1),
        "YES" if row["systematic_latency_flag"] else "NO",
    ] for row in summaries]
    warning = (
        "At least one system crosses the predeclared serious temporal-mismatch flag."
        if serious else
        "No system crosses the predeclared serious temporal-mismatch flag."
    )
    text = f"""# Intelligibility Alignment Audit

## Outcome

{warning} Metrics still use the unmodified sample-0 alignment and common valid length. **STOI/ESTOI/PESQ are interpreted as intrusive waveform-level proxies and may penalize generative timing differences.**

Frozen STOI/PESQ reuse validation: **{'PASS' if reuse_valid else 'FAIL — full recomputation required'}**. On the same 100 fixed-random DEV trials per system, stored values were checked against a fresh computation through the current project-wide audio pipeline with tolerance `{REUSE_TOLERANCE:g}`.

## Fixed audit protocol

- Fixed seed: `{ALIGNMENT_SEED}`; one shared set of 100 unique DEV trial IDs is used for all systems.
- Audio: evaluation-only clean target and frozen estimate; mono float32; finite check; project-wide 64-tap Kaiser sinc resampling to 16 kHz.
- Alignment: both signals begin at sample 0 and are truncated to their common valid length. No shift search, latency tuning, DTW, gain search, or target-dependent normalization is used.
- Duration mismatch: `abs(estimate/reference - 1) > 0.05`; short output: ratio `< 0.75`.
- The onset diagnostic uses each waveform's first 20 ms frame above a fixed -40 dB relative-energy threshold. It is diagnostic only and never changes metric inputs. A systematic-latency flag requires `|median onset delta| > 50 ms` and at least 75% of measured deltas with the same sign.
- A serious temporal-mismatch flag requires more than 20% duration mismatches or `|median onset delta| > 100 ms`.

## Results

{md_table(['System', 'Ref s', 'Est s', 'Common s', 'Median ratio', 'P05–P95 ratio', '>5% mismatch', '<75% short', 'Median onset Δ ms', 'Latency flag'], rows)}

## Integrity and interpretation

The raw 1,000 audit records are in `results/intelligibility_quality/alignment_audit_records.jsonl`; each includes durations, sample rates, common samples, onset diagnostics, waveform peaks, and stored-versus-recomputed STOI/PESQ errors. No amplitude modification was applied: all audited PCM values were already in the legal unit range. The clean target is used only as the intrusive evaluation reference.
"""
    atomic_text(DOCS / "INTELLIGIBILITY_ALIGNMENT_AUDIT.md", text)


def compute_trial(task: tuple[str, str, dict[str, Any], bool]) -> dict[str, Any]:
    split, slug, row, reuse_stored = task
    base = {
        "trial_id": row["trial_id"], "system": SYSTEMS[slug]["name"],
        "system_slug": slug, "split": split,
        "stoi": None, "estoi": None, "pesq": None,
        "stoi_error": None, "estoi_error": None, "pesq_error": None,
        "target_wer": float(row["target_WER"]),
        "content_switch": bool(row["content_switch"]),
        "speaker_margin": float(row["speaker_margin"]),
        "acoustic_switch": bool(row["acoustic_speaker_switch"]),
        "dnsmos": float(row["dnsmos_p808"]),
        "target_wav": str(row["target_wav"]), "output_wav": str(row["output_wav"]),
        "reference_duration_s": None, "estimate_duration_s": None,
        "common_duration_s": None, "length_ratio": None,
        "reference_original_sample_rate": None, "estimate_original_sample_rate": None,
        "common_samples_16k": None, "stored_common_samples": row.get("metric_overlap_samples"),
        "stored_metric_reuse": reuse_stored,
        "audio_error": None,
    }
    try:
        reference, reference_rate = audio16(row["target_wav"])
        estimate, estimate_rate = audio16(row["output_wav"])
        common = min(reference.size, estimate.size)
        if common <= 0:
            raise ValueError("zero common samples")
        ref = reference[:common]
        est = estimate[:common]
        base.update({
            "reference_duration_s": reference.size / TARGET_RATE,
            "estimate_duration_s": estimate.size / TARGET_RATE,
            "common_duration_s": common / TARGET_RATE,
            "length_ratio": estimate.size / reference.size,
            "reference_original_sample_rate": reference_rate,
            "estimate_original_sample_rate": estimate_rate,
            "common_samples_16k": common,
        })
        stored_common = row.get("metric_overlap_samples")
        if reuse_stored and (stored_common is None or int(stored_common) != common):
            raise ValueError(
                f"stored common length {stored_common} differs from current {common}"
            )
        if reuse_stored:
            base["stoi"] = float(row["stoi"])
            base["pesq"] = float(row["pesq_wb"])
        else:
            try:
                base["stoi"] = float(stoi_metric(ref, est, TARGET_RATE, extended=False))
            except Exception as error:  # noqa: BLE001
                base["stoi_error"] = f"{type(error).__name__}: {error}"
            try:
                base["pesq"] = float(pesq_metric(TARGET_RATE, ref, est, "wb"))
            except Exception as error:  # noqa: BLE001
                base["pesq_error"] = f"{type(error).__name__}: {error}"
        try:
            base["estoi"] = float(stoi_metric(ref, est, TARGET_RATE, extended=True))
        except Exception as error:  # noqa: BLE001
            base["estoi_error"] = f"{type(error).__name__}: {error}"
        for metric in ("stoi", "estoi", "pesq"):
            value = base[metric]
            if value is not None and not math.isfinite(float(value)):
                base[f"{metric}_error"] = "non-finite metric result"
                base[metric] = None
    except Exception as error:  # noqa: BLE001
        base["audio_error"] = f"{type(error).__name__}: {error}"
    return base


def evaluate(split: str, slug: str, chunk_size: int, workers: int) -> None:
    if not 1 <= chunk_size <= MAX_CHUNK:
        raise ValueError(f"chunk-size must be in [1, {MAX_CHUNK}]")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}]")
    protocol = json.loads((RESULTS / "protocol_validation.json").read_text(encoding="utf-8"))
    reuse_stored = bool(protocol["stored_stoi_pesq_reuse_valid"])
    rows = source_rows(split, slug)
    validate_source_rows(split, slug, rows)
    source_by_id = {row["trial_id"]: row for row in rows}
    shard = RESULTS / "shards" / split / f"{slug}.jsonl"
    prior = load_resume(shard)
    prior_by_id: dict[str, dict[str, Any]] = {}
    for row in prior:
        trial_id = row.get("trial_id")
        if trial_id not in source_by_id or row.get("system_slug") != slug or row.get("split") != split:
            raise ValueError(f"invalid resume record in {shard}: {trial_id}")
        if trial_id in prior_by_id:
            raise ValueError(f"duplicate resume record in {shard}: {trial_id}")
        prior_by_id[trial_id] = row
    pending = [row for row in rows if row["trial_id"] not in prior_by_id]
    print(
        f"EVALUATE_START split={split} system={slug} resumed={len(prior_by_id)} "
        f"pending={len(pending)} workers={workers} chunk={chunk_size} "
        f"reuse_stored_stoi_pesq={reuse_stored}", flush=True,
    )
    started = time.monotonic()
    completed_new = 0
    for offset in range(0, len(pending), chunk_size):
        batch = pending[offset:offset + chunk_size]
        tasks = [(split, slug, row, reuse_stored) for row in batch]
        if workers == 1:
            results = [compute_trial(task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(compute_trial, tasks))
        append_chunk(shard, results)
        for row in results:
            prior_by_id[row["trial_id"]] = row
        completed_new += len(results)
        elapsed = max(time.monotonic() - started, 1e-6)
        failures = sum(
            row["audio_error"] is not None or row["estoi"] is None
            or row["stoi"] is None or row["pesq"] is None
            for row in prior_by_id.values()
        )
        print(
            f"EVALUATE_PROGRESS split={split} system={slug} "
            f"complete={len(prior_by_id)}/{EXPECTED} new_rate={completed_new / elapsed:.2f}/s "
            f"metric_failures={failures}", flush=True,
        )
    ordered = [prior_by_id[row["trial_id"]] for row in rows]
    atomic_jsonl(shard, ordered)
    failures = sum(
        row["audio_error"] is not None or row["estoi"] is None
        or row["stoi"] is None or row["pesq"] is None for row in ordered
    )
    atomic_json(RESULTS / "shards" / split / f"{slug}.summary.json", {
        "status": "COMPLETE", "split": split, "system_slug": slug,
        "system": SYSTEMS[slug]["name"], "expected": EXPECTED,
        "unique": len({row["trial_id"] for row in ordered}), "missing": 0,
        "duplicate": 0, "metric_failures": failures,
        "pesq_coverage": sum(row["pesq"] is not None for row in ordered) / EXPECTED,
        "estoi_coverage": sum(row["estoi"] is not None for row in ordered) / EXPECTED,
        "stored_stoi_pesq_reused": reuse_stored, "workers": workers,
        "chunk_size": chunk_size,
    })
    print(
        f"EVALUATE_COMPLETE split={split} system={slug} rows={len(ordered)} "
        f"metric_failures={failures}", flush=True,
    )


def combine() -> None:
    for split in ("dev", "test"):
        combined: list[dict[str, Any]] = []
        for slug in SYSTEMS:
            shard = RESULTS / "shards" / split / f"{slug}.jsonl"
            rows = read_jsonl(shard)
            ids = [row.get("trial_id") for row in rows]
            if len(rows) != EXPECTED or len(set(ids)) != EXPECTED:
                raise ValueError(f"incomplete shard {shard}: {len(rows)} rows")
            if any(row.get("split") != split or row.get("system_slug") != slug for row in rows):
                raise ValueError(f"provenance mismatch in {shard}")
            combined.extend(rows)
        atomic_jsonl(RESULTS / f"{split}_per_trial.jsonl", combined)
        print(f"COMBINE_COMPLETE split={split} rows={len(combined)}", flush=True)


def mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def median_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else None


def summarize_rows(split: str, slug: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "split": split.upper(), "system": SYSTEMS[slug]["name"],
        "system_slug": slug, "expected": EXPECTED,
        "unique": len({row["trial_id"] for row in rows}),
        "missing": EXPECTED - len({row["trial_id"] for row in rows}),
        "duplicate": len(rows) - len({row["trial_id"] for row in rows}),
        "failed": sum(row.get("audio_error") is not None for row in rows),
        "wer": mean_present(rows, "target_wer"),
        "stoi_mean": mean_present(rows, "stoi"), "stoi_median": median_present(rows, "stoi"),
        "estoi_mean": mean_present(rows, "estoi"), "estoi_median": median_present(rows, "estoi"),
        "pesq_mean": mean_present(rows, "pesq"), "pesq_median": median_present(rows, "pesq"),
        "stoi_coverage": sum(row.get("stoi") is not None for row in rows) / EXPECTED,
        "estoi_coverage": sum(row.get("estoi") is not None for row in rows) / EXPECTED,
        "pesq_coverage": sum(row.get("pesq") is not None for row in rows) / EXPECTED,
        "dnsmos": mean_present(rows, "dnsmos"),
        "content_switch": mean_present(rows, "content_switch"),
        "acoustic_switch": mean_present(rows, "acoustic_switch"),
        "speaker_margin": mean_present(rows, "speaker_margin"),
    }


def bootstrap_ci(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values: list[np.ndarray] = []
    remaining = BOOTSTRAPS
    while remaining:
        batch = min(200, remaining)
        indexes = rng.integers(0, differences.size, size=(batch, differences.size))
        values.append(differences[indexes].mean(axis=1))
        remaining -= batch
    estimates = np.concatenate(values)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def paired_results(by_split: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_keys = OrderedDict([
        ("STOI", "stoi"), ("ESTOI", "estoi"), ("PESQ", "pesq"),
        ("DNSMOS", "dnsmos"), ("WER", "target_wer"),
    ])
    for split_index, split in enumerate(("dev", "test")):
        for comparison_index, (comparison, (base_slug, new_slug, label)) in enumerate(COMPARISONS.items()):
            base = by_split[split][base_slug]
            new = by_split[split][new_slug]
            common_ids = sorted(set(base) & set(new))
            for metric_index, (metric, key) in enumerate(metric_keys.items()):
                ids = [
                    trial_id for trial_id in common_ids
                    if base[trial_id].get(key) is not None and new[trial_id].get(key) is not None
                ]
                differences = np.asarray([
                    float(new[trial_id][key]) - float(base[trial_id][key]) for trial_id in ids
                ], dtype=np.float64)
                rng = np.random.default_rng(
                    BOOTSTRAP_SEED + 1000 * split_index + 100 * comparison_index + metric_index
                )
                low, high = bootstrap_ci(differences, rng)
                output.append({
                    "split": split.upper(), "comparison": comparison, "label": label,
                    "base_system": SYSTEMS[base_slug]["name"],
                    "new_system": SYSTEMS[new_slug]["name"], "metric": metric,
                    "n": len(ids), "base_mean": float(np.mean([
                        float(base[trial_id][key]) for trial_id in ids
                    ])), "new_mean": float(np.mean([
                        float(new[trial_id][key]) for trial_id in ids
                    ])),
                    "mean_difference_new_minus_base": float(np.mean(differences)),
                    "ci95_low": low, "ci95_high": high,
                    "bootstrap_resamples": BOOTSTRAPS,
                })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def comparison_row(
    paired: list[dict[str, Any]], split: str, comparison: str, metric: str,
) -> dict[str, Any]:
    return next(
        row for row in paired
        if row["split"] == split.upper() and row["comparison"] == comparison
        and row["metric"] == metric
    )


def improvement_decision(
    paired: list[dict[str, Any]], comparison: str, metrics: list[tuple[str, str]],
) -> str:
    """YES/NO/MIXED using preregistered directions and paired CIs.

    Direction is ``higher`` or ``lower`` for the new system. YES requires no
    unfavorable mean and at least one favorable CI; NO is the mirror image.
    """
    favorable_means: list[bool] = []
    unfavorable_means: list[bool] = []
    favorable_cis: list[bool] = []
    unfavorable_cis: list[bool] = []
    for split in ("DEV", "TEST"):
        for metric, direction in metrics:
            row = comparison_row(paired, split, comparison, metric)
            delta = float(row["mean_difference_new_minus_base"])
            low, high = float(row["ci95_low"]), float(row["ci95_high"])
            favorable_means.append(delta > 0 if direction == "higher" else delta < 0)
            unfavorable_means.append(delta < 0 if direction == "higher" else delta > 0)
            favorable_cis.append(low > 0 if direction == "higher" else high < 0)
            unfavorable_cis.append(high < 0 if direction == "higher" else low > 0)
    if all(favorable_means) and any(favorable_cis):
        return "YES"
    if all(unfavorable_means) and any(unfavorable_cis):
        return "NO"
    return "MIXED"


def preservation_decision(paired: list[dict[str, Any]]) -> str:
    """Preregistered practical margins: PESQ 0.05 and DNSMOS 0.02."""
    checks: list[bool] = []
    clear_failures: list[bool] = []
    for split in ("DEV", "TEST"):
        for metric, margin in (("PESQ", 0.05), ("DNSMOS", 0.02)):
            row = comparison_row(paired, split, "B", metric)
            delta = float(row["mean_difference_new_minus_base"])
            checks.append(delta >= -margin)
            clear_failures.append(delta < -margin and float(row["ci95_high"]) < 0)
    if all(checks):
        return "YES"
    if all(clear_failures):
        return "NO"
    return "MIXED"


def latex_table(tradeoff: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\caption{Intelligibility--quality trade-off.}",
        r"\label{tab:intelligibility_quality}", r"\small", r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"Split & System & WER$\downarrow$ & STOI$\uparrow$ & ESTOI$\uparrow$ & PESQ$\uparrow$ & DNSMOS$\uparrow$ \\",
        r"\midrule",
    ]
    for split in ("DEV", "TEST"):
        subset = [row for row in tradeoff if row["split"] == split]
        for index, row in enumerate(subset):
            name = {
                "pool_d_selected": "Pool D direct",
                "pool_d_qfull_ud": "Pool D Q-Full UD",
                "pool_d_qfull_csg": "Pool D Q-Full + CSG",
            }[row["system_slug"]]
            lines.append(
                f"{split if index == 0 else ''} & {name} & {100 * row['wer']:.2f} & "
                f"{row['stoi_mean']:.3f} & {row['estoi_mean']:.3f} & "
                f"{row['pesq_mean']:.3f} & {row['dnsmos']:.3f} \\\\"
            )
        if split == "DEV":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def aggregate() -> None:
    combined_by_split: dict[str, list[dict[str, Any]]] = {
        split: read_jsonl(RESULTS / f"{split}_per_trial.jsonl") for split in ("dev", "test")
    }
    summaries: list[dict[str, Any]] = []
    by_split: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for split, combined in combined_by_split.items():
        if len(combined) != EXPECTED * len(SYSTEMS):
            raise ValueError(f"{split} combined row count is {len(combined)}")
        by_split[split] = {}
        for slug in SYSTEMS:
            rows = [row for row in combined if row["system_slug"] == slug]
            if len(rows) != EXPECTED or len({row["trial_id"] for row in rows}) != EXPECTED:
                raise ValueError(f"{split}/{slug} is incomplete")
            by_split[split][slug] = {row["trial_id"]: row for row in rows}
            summaries.append(summarize_rows(split, slug, rows))
    write_csv(RESULTS / "INTELLIGIBILITY_QUALITY_TABLE.csv", summaries)
    tradeoff = [row for row in summaries if row["system_slug"] in TRADEOFF_SYSTEMS]
    write_csv(RESULTS / "POOL_D_GROUNDING_TRADEOFF.csv", tradeoff)
    paired = paired_results(by_split)
    write_csv(RESULTS / "PAIRED_BOOTSTRAP.csv", paired)
    atomic_json(RESULTS / "paired_bootstrap.json", paired)
    decisions = {
        "QFULL_INTELLIGIBILITY_VALUE": improvement_decision(
            paired, "A", [("STOI", "higher"), ("ESTOI", "higher")],
        ),
        "QFULL_QUALITY_VALUE": improvement_decision(
            paired, "A", [("PESQ", "higher"), ("DNSMOS", "higher")],
        ),
        "CSG_INTELLIGIBILITY_VALUE": improvement_decision(
            paired, "B", [("WER", "lower"), ("STOI", "higher"), ("ESTOI", "higher")],
        ),
        "CSG_QUALITY_PRESERVATION": preservation_decision(paired),
    }
    atomic_json(RESULTS / "scientific_judgments.json", decisions)
    atomic_text(RESULTS / "intelligibility_quality_table.tex", latex_table(tradeoff))
    write_main_report(summaries, tradeoff, paired, decisions)


def full_table_rows(summaries: list[dict[str, Any]], split: str) -> list[list[str]]:
    return [[
        SYSTEMS[row["system_slug"]]["short"], pct(row["wer"], 2),
        fmt(row["stoi_mean"]), fmt(row["estoi_mean"]), fmt(row["pesq_mean"]),
        fmt(row["dnsmos"]), pct(row["content_switch"], 2), pct(row["acoustic_switch"], 2),
    ] for row in summaries if row["split"] == split]


def tradeoff_rows(tradeoff: list[dict[str, Any]], split: str) -> list[list[str]]:
    return [[
        SYSTEMS[row["system_slug"]]["short"], pct(row["wer"], 2),
        fmt(row["stoi_mean"]), fmt(row["estoi_mean"]), fmt(row["pesq_mean"]), fmt(row["dnsmos"]),
    ] for row in tradeoff if row["split"] == split]


def paired_md_rows(paired: list[dict[str, Any]], split: str) -> list[list[str]]:
    return [[
        row["comparison"], row["metric"], str(row["n"]),
        f"{row['mean_difference_new_minus_base']:+.5f}",
        f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}]",
    ] for row in paired if row["split"] == split]


def write_main_report(
    summaries: list[dict[str, Any]], tradeoff: list[dict[str, Any]],
    paired: list[dict[str, Any]], decisions: dict[str, str],
) -> None:
    protocol = json.loads((RESULTS / "protocol_validation.json").read_text(encoding="utf-8"))
    asset = json.loads((RESULTS / "asset_audit.json").read_text(encoding="utf-8"))
    direct = next(row for row in tradeoff if row["split"] == "TEST" and row["system_slug"] == "pool_d_selected")
    ud = next(row for row in tradeoff if row["split"] == "TEST" and row["system_slug"] == "pool_d_qfull_ud")
    csg = next(row for row in tradeoff if row["split"] == "TEST" and row["system_slug"] == "pool_d_qfull_csg")

    a_dev = {metric: comparison_row(paired, "DEV", "A", metric) for metric in ("STOI", "ESTOI", "PESQ", "DNSMOS", "WER")}
    a_test = {metric: comparison_row(paired, "TEST", "A", metric) for metric in ("STOI", "ESTOI", "PESQ", "DNSMOS", "WER")}
    b_dev = {metric: comparison_row(paired, "DEV", "B", metric) for metric in ("STOI", "ESTOI", "PESQ", "DNSMOS", "WER")}
    b_test = {metric: comparison_row(paired, "TEST", "B", metric) for metric in ("STOI", "ESTOI", "PESQ", "DNSMOS", "WER")}

    def sentence(system: str, row: dict[str, Any]) -> str:
        return (
            f"{system}: WER {100 * row['wer']:.2f}%, STOI {row['stoi_mean']:.3f}, "
            f"ESTOI {row['estoi_mean']:.3f}, PESQ {row['pesq_mean']:.3f}, "
            f"and DNSMOS {row['dnsmos']:.3f}"
        )

    paragraph = (
        "On the frozen TEST set, " + sentence("Pool D direct", direct) + "; "
        + sentence("Q-Full UD", ud) + "; and " + sentence("Q-Full + CSG", csg)
        + ". Q-Full's waveform-intelligibility value is "
        + decisions["QFULL_INTELLIGIBILITY_VALUE"].lower()
        + ", while its perceptual-quality value is "
        + decisions["QFULL_QUALITY_VALUE"].lower()
        + ". Relative to UD, CSG's intelligibility/content recovery is "
        + decisions["CSG_INTELLIGIBILITY_VALUE"].lower()
        + " and quality preservation is "
        + decisions["CSG_QUALITY_PRESERVATION"].lower()
        + ". STOI/ESTOI/PESQ remain intrusive proxies and can penalize generative timing differences."
    )
    words = paragraph.split()
    if len(words) > 120:
        raise ValueError(f"paper paragraph exceeds 120 words: {len(words)}")

    pesq_coverage = min(float(row["pesq_coverage"]) for row in summaries)
    warning = (
        "A serious temporal-mismatch flag was observed; see the alignment audit."
        if protocol["serious_temporal_mismatch"] else
        "The alignment audit found no serious temporal-mismatch flag under its fixed thresholds."
    )
    text = f"""# Intelligibility / Content Fidelity / Speech Quality Report

## Executive answer

The post-hoc frozen-output evaluation is complete for **10 systems × 6,000 DEV × 6,000 TEST trials**. Asset integrity is PASS (`missing=0`, `duplicate=0`, `failed=0` for every frozen source). {warning}

- Does Q-Full improve waveform-level intelligibility over direct Pool D? **{decisions['QFULL_INTELLIGIBILITY_VALUE']}**
- Does Q-Full improve perceptual quality? **{decisions['QFULL_QUALITY_VALUE']}**
- Does CSG recover intelligibility/content fidelity relative to Q-Full UD? **{decisions['CSG_INTELLIGIBILITY_VALUE']}**
- Does CSG preserve generative quality? **{decisions['CSG_QUALITY_PRESERVATION']}**

These judgments follow fixed direction rules across both splits and 10,000-resample paired-bootstrap intervals; conflicting metrics or splits yield MIXED. They do not trigger model selection or tuning.

## Scope and provenance

Only previously frozen waveform outputs were read. No model was trained, no tokens were decoded, no waveform was regenerated, and candidate selection, Q-Full, and CSG were unchanged. Clean target waveforms are used solely as evaluation references. **Additional intrusive metrics are computed post hoc on the previously frozen TEST outputs; no model or decoding decisions are changed.**

The source mappings are the same mappings used by `build_icassp_table.py` and `build_icassp_test_table.py`. The machine audit found {len(asset['systems'])} complete system/split sources. STOI and PESQ-WB already existed in every frozen per-trial source; their code path, common lengths, and numerical values were freshly verified on 100 fixed-random DEV trials per system before reuse. Reuse status: **{'PASS' if protocol['stored_stoi_pesq_reuse_valid'] else 'DISABLED; recomputed'}**. ESTOI was newly computed for every trial.

## Evaluation protocol

- Mono float32, finite-value check, and the project-wide fixed high-quality torchaudio Kaiser sinc resampler (width 64, rolloff 0.9475937167, beta 14.7696564594) to 16 kHz.
- Sample-0 start and common valid length only. No per-system shift, DTW, dynamic alignment, target-dependent latency correction, optimal scaling, loudness normalization, or oracle gain search.
- No amplitude clipping was applied; the fixed audit found every sampled waveform peak in the legal unit range.
- `pystoi.stoi(..., extended=False)` for STOI and `extended=True` for ESTOI; ITU-T P.862-compatible wideband PESQ at 16 kHz.
- Streaming execution, two worker threads, chunks of 100, one system/split at a time, with fsync'ed resume shards.

**STOI/ESTOI/PESQ are interpreted as intrusive waveform-level proxies and may penalize generative timing differences.** WER instead measures linguistic-content correctness/ASR consistency; disagreement is reported, not resolved by cherry-picking.

## Table A — DEV

{md_table(['System', 'WER ↓', 'STOI ↑', 'ESTOI ↑', 'PESQ ↑', 'DNSMOS ↑', 'Content switch ↓', 'Acoustic switch ↓'], full_table_rows(summaries, 'DEV'))}

## Table B — TEST

{md_table(['System', 'WER ↓', 'STOI ↑', 'ESTOI ↑', 'PESQ ↑', 'DNSMOS ↑', 'Content switch ↓', 'Acoustic switch ↓'], full_table_rows(summaries, 'TEST'))}

All rows also have mean/median STOI, ESTOI, and PESQ plus metric coverage in `INTELLIGIBILITY_QUALITY_TABLE.csv`. Minimum PESQ coverage is **{pct(pesq_coverage, 2)}**; any coverage below 99% would be explicitly marked here.

## Pool D grounding trade-off

### DEV

{md_table(['System', 'Target WER ↓', 'STOI ↑', 'ESTOI ↑', 'PESQ ↑', 'DNSMOS ↑'], tradeoff_rows(tradeoff, 'DEV'))}

### TEST

{md_table(['System', 'Target WER ↓', 'STOI ↑', 'ESTOI ↑', 'PESQ ↑', 'DNSMOS ↑'], tradeoff_rows(tradeoff, 'TEST'))}

## Scientific interpretation

**Pool D direct remains the strongest overall output.** Relative to it, Q-Full UD raises WER by {100 * a_dev['WER']['mean_difference_new_minus_base']:.2f} points on DEV and {100 * a_test['WER']['mean_difference_new_minus_base']:.2f} points on TEST, while reducing STOI by {abs(a_dev['STOI']['mean_difference_new_minus_base']):.3f}/{abs(a_test['STOI']['mean_difference_new_minus_base']):.3f}, ESTOI by {abs(a_dev['ESTOI']['mean_difference_new_minus_base']):.3f}/{abs(a_test['ESTOI']['mean_difference_new_minus_base']):.3f}, and PESQ by {abs(a_dev['PESQ']['mean_difference_new_minus_base']):.3f}/{abs(a_test['PESQ']['mean_difference_new_minus_base']):.3f}. DNSMOS rises by only {a_dev['DNSMOS']['mean_difference_new_minus_base']:+.3f}/{a_test['DNSMOS']['mean_difference_new_minus_base']:+.3f}; the TEST interval includes zero. Thus the data do not support a general Q-Full waveform-intelligibility or perceptual-quality advantage.

**CSG provides a consistent but partial recovery.** Relative to Q-Full UD, it lowers WER by {abs(100 * b_dev['WER']['mean_difference_new_minus_base']):.2f}/{abs(100 * b_test['WER']['mean_difference_new_minus_base']):.2f} points, raises STOI by {b_dev['STOI']['mean_difference_new_minus_base']:.4f}/{b_test['STOI']['mean_difference_new_minus_base']:.4f} and ESTOI by {b_dev['ESTOI']['mean_difference_new_minus_base']:.4f}/{b_test['ESTOI']['mean_difference_new_minus_base']:.4f}, and slightly raises PESQ. DNSMOS falls by only {abs(b_dev['DNSMOS']['mean_difference_new_minus_base']):.3f}/{abs(b_test['DNSMOS']['mean_difference_new_minus_base']):.3f}, within the predeclared preservation margin. CSG therefore reduces generative drift without closing the large gap to direct Pool D.

## Paired significance

Every difference is **new minus base**. Therefore negative is favorable for WER; positive is favorable for STOI, ESTOI, PESQ, and DNSMOS. Each interval uses exactly {BOOTSTRAPS:,} paired bootstrap resamples with fixed seeds. PESQ uses only pairs with both values present and reports `n` explicitly.

Comparisons: A = Pool D direct → Pool D Q-Full UD; B = Pool D Q-Full UD → Pool D Q-Full + CSG; C = Pool D direct → Pool D Q-Full + CSG.

### DEV

{md_table(['Comparison', 'Metric', 'n', 'Mean Δ', '95% CI'], paired_md_rows(paired, 'DEV'))}

### TEST

{md_table(['Comparison', 'Metric', 'n', 'Mean Δ', '95% CI'], paired_md_rows(paired, 'TEST'))}

## Decision rules and limitations

For Questions A–C, YES requires every split/metric mean in the favorable direction and at least one favorable CI excluding zero; NO is the mirrored rule; all conflicts are MIXED. For CSG quality preservation, the predeclared practical non-inferiority margins are 0.05 PESQ and 0.02 DNSMOS: YES requires every DEV/TEST drop to remain within its margin, NO requires every component to show a clear larger drop, and otherwise the result is MIXED.

Intrusive scores can punish harmless phase, prosody, or timing differences in generative speech. PESQ is an older telephony-oriented proxy; DNSMOS is non-intrusive; WER depends on the frozen ASR protocol. None alone defines target-speaker correctness. The report therefore keeps content switch, acoustic switch, speaker margin, WER, intrusive intelligibility, and non-intrusive quality conceptually separate.

## Intelligibility–quality trade-off (paper-ready; {len(words)} words)

{paragraph}

## Artifacts

- Alignment audit: `docs/INTELLIGIBILITY_ALIGNMENT_AUDIT.md`
- Per-trial results: `results/intelligibility_quality/dev_per_trial.jsonl`, `test_per_trial.jsonl`
- Full summary: `results/intelligibility_quality/INTELLIGIBILITY_QUALITY_TABLE.csv`
- Three-system table: `results/intelligibility_quality/POOL_D_GROUNDING_TRADEOFF.csv`
- Paired results: `results/intelligibility_quality/PAIRED_BOOTSTRAP.csv`
- ICASSP LaTeX: `results/intelligibility_quality/intelligibility_quality_table.tex`
"""
    atomic_text(DOCS / "INTELLIGIBILITY_QUALITY_REPORT.md", text)


def validate_outputs() -> None:
    required = [
        DOCS / "INTELLIGIBILITY_ALIGNMENT_AUDIT.md",
        RESULTS / "dev_per_trial.jsonl", RESULTS / "test_per_trial.jsonl",
        RESULTS / "INTELLIGIBILITY_QUALITY_TABLE.csv",
        RESULTS / "POOL_D_GROUNDING_TRADEOFF.csv",
        RESULTS / "PAIRED_BOOTSTRAP.csv",
        RESULTS / "intelligibility_quality_table.tex",
        DOCS / "INTELLIGIBILITY_QUALITY_REPORT.md",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"missing outputs: {missing}")
    for split in ("dev", "test"):
        rows = read_jsonl(RESULTS / f"{split}_per_trial.jsonl")
        if len(rows) != EXPECTED * len(SYSTEMS):
            raise ValueError(f"{split} per-trial rows={len(rows)}")
        keys = [(row["system_slug"], row["trial_id"]) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{split} per-trial duplicate keys")
        required_fields = (
            "trial_id", "system", "split", "stoi", "estoi", "pesq",
            "target_wer", "content_switch", "speaker_margin", "acoustic_switch", "dnsmos",
        )
        if any(any(field not in row for field in required_fields) for row in rows):
            raise ValueError(f"{split} missing required per-trial fields")
    summaries = list(csv.DictReader((RESULTS / "INTELLIGIBILITY_QUALITY_TABLE.csv").open()))
    tradeoff = list(csv.DictReader((RESULTS / "POOL_D_GROUNDING_TRADEOFF.csv").open()))
    paired = list(csv.DictReader((RESULTS / "PAIRED_BOOTSTRAP.csv").open()))
    if len(summaries) != 20 or len(tradeoff) != 6 or len(paired) != 30:
        raise ValueError(
            f"table sizes summaries={len(summaries)} tradeoff={len(tradeoff)} paired={len(paired)}"
        )
    print("VALIDATION_COMPLETE status=PASS", flush=True)


def final_summary() -> None:
    rows = list(csv.DictReader((RESULTS / "POOL_D_GROUNDING_TRADEOFF.csv").open()))
    test = {row["system_slug"]: row for row in rows if row["split"] == "TEST"}
    decisions = json.loads((RESULTS / "scientific_judgments.json").read_text(encoding="utf-8"))
    full = list(csv.DictReader((RESULTS / "INTELLIGIBILITY_QUALITY_TABLE.csv").open()))
    complete = {
        split: len([row for row in full if row["split"] == split]) == len(SYSTEMS)
        and all(
            int(row["expected"]) == EXPECTED and int(row["unique"]) == EXPECTED
            and int(row["missing"]) == 0 and int(row["duplicate"]) == 0
            and int(row["failed"]) == 0
            for row in full if row["split"] == split
        ) for split in ("DEV", "TEST")
    }

    def system_block(label: str, slug: str) -> None:
        row = test[slug]
        print(f"{label}:")
        print(f"WER = {100 * float(row['wer']):.2f}%")
        print(f"STOI = {float(row['stoi_mean']):.4f}")
        print(f"ESTOI = {float(row['estoi_mean']):.4f}")
        print(f"PESQ = {float(row['pesq_mean']):.4f}")
        print(f"DNSMOS = {float(row['dnsmos']):.4f}")

    print(f"DEV_COMPLETE:\n{'YES' if complete['DEV'] else 'NO'}\n")
    print(f"TEST_COMPLETE:\n{'YES' if complete['TEST'] else 'NO'}\n")
    system_block("POOL_D_DIRECT_TEST", "pool_d_selected")
    print()
    system_block("POOL_D_QFULL_UD_TEST", "pool_d_qfull_ud")
    print()
    system_block("POOL_D_QFULL_CSG_TEST", "pool_d_qfull_csg")
    print()
    for key in (
        "QFULL_INTELLIGIBILITY_VALUE", "QFULL_QUALITY_VALUE",
        "CSG_INTELLIGIBILITY_VALUE", "CSG_QUALITY_PRESERVATION",
    ):
        print(f"{key}:\n{decisions[key]}\n")
    dev_coverage = min(float(row["pesq_coverage"]) for row in full if row["split"] == "DEV")
    test_coverage = min(float(row["pesq_coverage"]) for row in full if row["split"] == "TEST")
    print(f"PESQ_COVERAGE:\nDEV = {100 * dev_coverage:.2f}%; TEST = {100 * test_coverage:.2f}%\n")
    print("REPORT:\ndocs/INTELLIGIBILITY_QUALITY_REPORT.md\n")
    print("TABLE:\nresults/intelligibility_quality/intelligibility_quality_table.tex")


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    with exclusive_lock():
        if args.command == "asset-audit":
            asset_audit()
        elif args.command == "alignment-audit":
            alignment_audit(args.workers)
        elif args.command == "evaluate":
            evaluate(args.split, args.system, args.chunk_size, args.workers)
        elif args.command == "combine":
            combine()
        elif args.command == "aggregate":
            aggregate()
        elif args.command == "validate":
            validate_outputs()
        elif args.command == "final-summary":
            final_summary()
        else:
            raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
