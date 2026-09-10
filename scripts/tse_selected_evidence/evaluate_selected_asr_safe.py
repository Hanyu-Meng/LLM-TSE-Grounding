#!/usr/bin/env python3
"""Resume-safe Whisper and DNSMOS evaluation for selected-evidence audio."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resource_guard import ResourceGuardStop, check, check_or_raise  # noqa: E402
from se_align.eval.metrics import _DNSMOS  # noqa: E402
from se_align.utils.audio import load_wav, resample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument(
        "--asr-model", type=Path, default=ROOT / "pretrained/whisper-small.en"
    )
    parser.add_argument("--transcript-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--expected", type=int, default=6000)
    parser.add_argument("--status-every", type=int, default=50)
    parser.add_argument("--resource-seconds", type=float, default=20.0)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--guard-log", type=Path,
        default=ROOT / "analysis/selected_evidence/resource_guard.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def normalize(text: str) -> str:
    lowered = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", lowered).strip()


def audio16(path: str) -> np.ndarray:
    waveform, sample_rate = load_wav(path)
    return resample(waveform, sample_rate, 16000).reshape(-1).numpy()


def load_keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get(key):
            result[str(row[key])] = row
    return result


def build_asr(args: argparse.Namespace):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.asr_model)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0 if use_cuda else -1,
    )


def word_score(reference: str, hypothesis: str) -> tuple[float | None, int | None]:
    if not reference:
        return None, None
    score = jiwer.process_words(reference, hypothesis)
    return float(score.wer), int(
        score.substitutions + score.deletions + score.insertions
    )


def leakage(target: str, interferer: str, output: str) -> float | None:
    target_words = set(target.split())
    interferer_only = set(interferer.split()) - target_words
    if not interferer_only:
        return None
    return len(interferer_only.intersection(output.split())) / len(interferer_only)


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key]) for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row for row in rows
        if row.get("decode_status") == "ok" and row.get("target_WER") is not None
    ]
    summary: dict[str, Any] = {"count": len(successful)}
    for key in (
        "target_WER", "sim_target", "sim_interferer", "speaker_margin",
        "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovl",
        "si_sdr_db", "si_sdri_db", "stoi",
        "pesq_wb", "interferer_leakage",
    ):
        values = finite_values(successful, key)
        summary[key] = float(np.mean(values)) if values else None
    wers = finite_values(successful, "target_WER")
    summary.update({
        "target_WER_median": float(np.median(wers)) if wers else None,
        "target_WER_p95": float(np.percentile(wers, 95)) if wers else None,
        "target_WER_p99": float(np.percentile(wers, 99)) if wers else None,
        "target_WER_gt_0p5": float(np.mean(np.asarray(wers) > 0.5)) if wers else None,
        "content_switch_rate": float(np.mean([
            row["content_switch"] for row in successful
        ])) if successful else None,
        "acoustic_speaker_switch_rate": float(np.mean([
            row["acoustic_speaker_switch"] for row in successful
        ])) if successful else None,
        "empty_output_rate": float(np.mean([
            not row["output_text"] for row in successful
        ])) if successful else None,
        "short_output_rate": float(np.mean([
            row["unrelated_short_output"] for row in successful
        ])) if successful else None,
    })
    return summary


def main() -> int:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    manifest = {row["trial_id"]: row for row in manifest_rows}
    audio_rows = read_jsonl(args.audio_metrics)
    ids = [row["trial_id"] for row in audio_rows]
    if (
        len(manifest) != args.expected or len(manifest_rows) != args.expected
        or len(audio_rows) != args.expected or len(set(ids)) != args.expected
        or set(ids) != set(manifest)
    ):
        raise ValueError("manifest/audio metrics are not exact 6,000-trial peers")
    if any(row.get("split") != args.split for row in manifest_rows + audio_rows):
        raise ValueError(f"inputs are not exclusively split={args.split}")
    if any(row.get("decode_status") != "ok" for row in audio_rows):
        raise ValueError("audio metrics contain failed rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    guard_log = args.guard_log
    guard_log.parent.mkdir(parents=True, exist_ok=True)
    phase = f"selected_asr_{args.system_name}"
    preflight = check_or_raise(
        phase=phase, disk_path=ROOT, log_path=guard_log, starting_new_stage=True
    )

    transcript_rows = load_keyed(args.transcript_cache, "audio_path")
    transcript_cache = {
        path: str(row.get("text") or "") for path, row in transcript_rows.items()
    }
    all_paths: list[str] = []
    for row in audio_rows:
        all_paths.extend((row["target_wav"], row["interferer_wav"], row["output_wav"]))
    missing_transcripts = list(dict.fromkeys(
        path for path in all_paths if path not in transcript_cache
    ))

    guard_stop: str | None = None
    started = time.monotonic()
    last_guard = time.monotonic()
    if missing_transcripts:
        asr = build_asr(args)
        loaded = check(
            phase=f"{phase}_whisper_loaded", disk_path=ROOT,
            log_path=guard_log, starting_new_stage=False,
        )
        if loaded["evaluation"]["decision"] == "GRACEFUL_STOP":
            raise ResourceGuardStop("; ".join(loaded["evaluation"]["stop_reasons"]))
        try:
            for start in range(0, len(missing_transcripts), args.batch_size):
                paths = missing_transcripts[start:start + args.batch_size]
                inputs = [
                    {"raw": audio16(path), "sampling_rate": 16000}
                    for path in paths
                ]
                outputs = asr(
                    inputs,
                    batch_size=len(inputs),
                    generate_kwargs={"num_beams": 1, "do_sample": False},
                )
                for path, output in zip(paths, outputs):
                    text = str(output.get("text") or "").strip()
                    transcript_cache[path] = text
                    append_jsonl(args.transcript_cache, {
                        "audio_path": path, "text": text,
                    })
                completed = min(start + len(paths), len(missing_transcripts))
                now = time.monotonic()
                if now - last_guard >= args.resource_seconds:
                    resource = check(
                        phase=phase, disk_path=ROOT, log_path=guard_log,
                        starting_new_stage=False,
                    )
                    last_guard = now
                    if resource["evaluation"]["decision"] == "GRACEFUL_STOP":
                        guard_stop = "; ".join(resource["evaluation"]["stop_reasons"])
                        raise ResourceGuardStop(guard_stop)
                if completed % args.status_every == 0 or completed == len(missing_transcripts):
                    print(
                        f"asr_cached={completed}/{len(missing_transcripts)}",
                        flush=True,
                    )
        except ResourceGuardStop as error:
            guard_stop = str(error)
        finally:
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            check(
                phase=f"{phase}_whisper_unloaded", disk_path=ROOT,
                log_path=guard_log, starting_new_stage=False,
            )
    if guard_stop:
        print(f"RESOURCE_GUARD_STOP: {guard_stop}", flush=True)
        return 3

    dnsmos_path = args.output_dir / "dnsmos_cache.jsonl"
    dnsmos_rows = load_keyed(dnsmos_path, "audio_path")
    missing_mos = list(dict.fromkeys(
        row["output_wav"] for row in audio_rows
        if row["output_wav"] not in dnsmos_rows
    ))
    dnsmos_model = None
    if missing_mos:
        os.environ["DNSMOS_THREADS"] = "1"
        dnsmos_model = _DNSMOS(use_gpu=False)
        try:
            for index, path in enumerate(missing_mos, 1):
                scores = dnsmos_model(audio16(path))
                row = {"audio_path": path, **scores}
                append_jsonl(dnsmos_path, row)
                dnsmos_rows[path] = row
                now = time.monotonic()
                if now - last_guard >= args.resource_seconds:
                    resource = check(
                        phase=f"{phase}_dnsmos", disk_path=ROOT,
                        log_path=guard_log, starting_new_stage=False,
                    )
                    last_guard = now
                    if resource["evaluation"]["decision"] == "GRACEFUL_STOP":
                        guard_stop = "; ".join(resource["evaluation"]["stop_reasons"])
                        raise ResourceGuardStop(guard_stop)
                if index % args.status_every == 0 or index == len(missing_mos):
                    print(f"dnsmos={index}/{len(missing_mos)}", flush=True)
        except ResourceGuardStop as error:
            guard_stop = str(error)
        finally:
            del dnsmos_model
            gc.collect()
    if guard_stop:
        print(f"RESOURCE_GUARD_STOP: {guard_stop}", flush=True)
        return 3

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, audio_row in enumerate(audio_rows, 1):
        record = dict(audio_row)
        try:
            target_text = normalize(transcript_cache[record["target_wav"]])
            interferer_text = normalize(transcript_cache[record["interferer_wav"]])
            output_text = normalize(transcript_cache[record["output_wav"]])
            target_wer, target_distance = word_score(target_text, output_text)
            interferer_wer, interferer_distance = word_score(
                interferer_text, output_text
            )
            if target_wer is None:
                raise ValueError("empty normalized target reference")
            short_threshold = max(3, int(np.ceil(0.25 * len(target_text.split()))))
            unrelated_short = len(output_text.split()) < short_threshold
            mos = dnsmos_rows[record["output_wav"]]
            record.update({
                "target_text": target_text,
                "interferer_text": interferer_text,
                "output_text": output_text,
                "target_WER": target_wer,
                "target_word_distance": target_distance,
                "interferer_WER": interferer_wer,
                "interferer_word_distance": interferer_distance,
                "content_switch": (
                    interferer_wer is not None and interferer_wer < target_wer
                ),
                "interferer_leakage": leakage(
                    target_text, interferer_text, output_text
                ),
                "target_utterance": manifest[record["trial_id"]].get(
                    "target_utterance"
                ),
                "unrelated_short_output": unrelated_short,
                "short_output_word_threshold": short_threshold,
                "dnsmos_p808": float(mos["dnsmos_p808"]),
                "dnsmos_sig": float(mos["dnsmos_sig"]),
                "dnsmos_bak": float(mos["dnsmos_bak"]),
                "dnsmos_ovl": float(mos["dnsmos_ovl"]),
                "test_used": args.split == "test",
            })
        except Exception as error:  # noqa: BLE001
            record["decode_status"] = "failed"
            record["failure_reason"] = f"{type(error).__name__}: {error}"
            failures.append({
                "trial_id": record["trial_id"],
                "error_type": type(error).__name__,
                "error": str(error),
            })
        records.append(record)
        if index % 500 == 0 or index == len(audio_rows):
            print(f"metrics={index}/{len(audio_rows)}", flush=True)

    per_trial_path = args.output_dir / "per_trial_metrics.jsonl"
    atomic_jsonl(per_trial_path, records)
    atomic_jsonl(args.output_dir / "metric_failures.jsonl", failures)
    transcript_output = args.output_dir / "asr_transcripts/transcripts.jsonl"
    transcript_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(transcript_output, [{
        key: row.get(key)
        for key in ("trial_id", "target_text", "interferer_text", "output_text")
    } for row in records])
    decoded = sum(row.get("decode_status") == "ok" for row in records)
    by_cohort = {
        cohort: summarize([row for row in records if row.get("cohort") == cohort])
        for cohort in (
            "natural_primary_swap", "primary_correct_control",
            "ambiguous_primary_wrong",
        )
    }
    summary = {
        "status": "COMPLETE" if decoded == args.expected and not failures else "FAILED",
        "system": args.system_name,
        "expected": args.expected,
        "unique": len({row["trial_id"] for row in records}),
        "attempted": len(records),
        "decoded": decoded,
        "missing": args.expected - len(records),
        "duplicate": len(records) - len({row["trial_id"] for row in records}),
        "failed": len(failures),
        "asr_model": str(args.asr_model.resolve()),
        "asr_decoding": "English-only greedy, num_beams=1, do_sample=false",
        "text_normalization": "lowercase, ASCII punctuation removal, whitespace collapse",
        "wer_reference": "same-ASR clean target waveform (ASR-consistency WER)",
        "quality_metric": "DNSMOS P.808 (CPU ONNX, one serial worker)",
        f"full_{args.split}": summarize(records),
        "cohorts": by_cohort,
        "preflight": preflight["evaluation"]["decision"],
        "test_used": args.split == "test",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
