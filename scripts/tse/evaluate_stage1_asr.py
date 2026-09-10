#!/usr/bin/env python3
"""Run the fixed Whisper/DNSMOS Stage-1 evaluator and finalize trial metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import re
import string
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from se_align.eval.metrics import _DNSMOS  # noqa: E402
from se_align.utils.audio import load_wav, resample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--transcript-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dnsmos-workers", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def audio16(path: str) -> np.ndarray:
    waveform, sample_rate = load_wav(path)
    return resample(waveform, sample_rate, 16000).reshape(-1).numpy()


def load_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {row["audio_path"]: row["text"] for row in read_jsonl(path)}


_DNSMOS_MODEL = None


def init_dnsmos_worker() -> None:
    global _DNSMOS_MODEL
    os.environ["DNSMOS_THREADS"] = "1"
    _DNSMOS_MODEL = _DNSMOS(use_gpu=False)


def dnsmos_worker(path: str) -> tuple[str, dict[str, float]]:
    if _DNSMOS_MODEL is None:
        raise RuntimeError("DNSMOS worker was not initialized")
    return path, _DNSMOS_MODEL(audio16(path))


def score_dnsmos(paths: list[str], workers: int) -> dict[str, dict[str, float]]:
    unique = list(dict.fromkeys(paths))
    context = multiprocessing.get_context("spawn")
    scores = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=init_dnsmos_worker,
    ) as executor:
        for index, (path, score) in enumerate(
            executor.map(dnsmos_worker, unique, chunksize=8), 1
        ):
            scores[path] = score
            if index == 1 or index % 200 == 0 or index == len(unique):
                print(f"dnsmos={index}/{len(unique)}", flush=True)
    return scores


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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.asr_model)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0 if use_cuda else -1,
    )


def transcribe_missing(asr, paths: list[str], cache: dict[str, str], cache_path: Path,
                       batch_size: int) -> None:
    missing = list(dict.fromkeys(path for path in paths if path not in cache))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing), batch_size):
        batch_paths = missing[start:start + batch_size]
        inputs = [{"raw": audio16(path), "sampling_rate": 16000} for path in batch_paths]
        outputs = asr(
            inputs,
            batch_size=len(inputs),
            generate_kwargs={"num_beams": 1, "do_sample": False},
        )
        with cache_path.open("a", encoding="utf-8") as handle:
            for path, output in zip(batch_paths, outputs):
                text = (output.get("text") or "").strip()
                cache[path] = text
                handle.write(json.dumps({"audio_path": path, "text": text}) + "\n")
        completed = min(start + len(batch_paths), len(missing))
        if completed == len(missing) or completed % 200 < len(batch_paths):
            print(f"asr_cached={completed}/{len(missing)}", flush=True)


def word_score(reference: str, hypothesis: str) -> tuple[float | None, int | None]:
    if not reference:
        return None, None
    score = jiwer.process_words(reference, hypothesis)
    return float(score.wer), int(score.substitutions + score.deletions + score.insertions)


def leakage(target: str, interferer: str, output: str) -> float | None:
    target_words = set(target.split())
    interferer_only = set(interferer.split()) - target_words
    if not interferer_only:
        return None
    return len(interferer_only.intersection(output.split())) / len(interferer_only)


def finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["decode_status"] == "ok" and row.get("target_WER") is not None]
    summary = {"count": len(successful)}
    for key in (
        "target_WER", "sim_target", "sim_interferer", "speaker_margin",
        "dnsmos_p808", "dnsmos_ovl", "si_sdri_db", "stoi", "pesq_wb",
        "interferer_leakage",
    ):
        values = finite_values(successful, key)
        summary[key] = float(np.mean(values)) if values else None
    wers = finite_values(successful, "target_WER")
    summary.update(
        {
            "target_WER_median": float(np.median(wers)) if wers else None,
            "target_WER_p95": float(np.percentile(wers, 95)) if wers else None,
            "target_WER_p99": float(np.percentile(wers, 99)) if wers else None,
            "target_WER_gt_0p5": float(np.mean(np.asarray(wers) > 0.5)) if wers else None,
            "content_switch_rate": float(np.mean([row["content_switch"] for row in successful])) if successful else None,
            "acoustic_speaker_switch_rate": float(np.mean([row["acoustic_speaker_switch"] for row in successful])) if successful else None,
        }
    )
    return summary


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = args.output_dir / "asr_transcripts"
    transcript_dir.mkdir(exist_ok=True)
    manifest = {row["trial_id"]: row for row in read_jsonl(args.manifest)}
    audio_rows = read_jsonl(args.audio_metrics)
    cache = load_cache(args.transcript_cache)
    paths = []
    for row in audio_rows:
        if row["decode_status"] == "ok":
            paths.extend((row["target_wav"], row["interferer_wav"], row["output_wav"]))
    asr = build_asr(args)
    transcribe_missing(asr, paths, cache, args.transcript_cache, args.batch_size)
    del asr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    quality_scores = score_dnsmos(
        [row["output_wav"] for row in audio_rows if row["decode_status"] == "ok"],
        args.dnsmos_workers,
    )

    records = []
    for index, audio_row in enumerate(audio_rows, 1):
        record = dict(audio_row)
        if record["decode_status"] != "ok":
            records.append(record)
            continue
        try:
            target_text = normalize(cache[record["target_wav"]])
            interferer_text = normalize(cache[record["interferer_wav"]])
            output_text = normalize(cache[record["output_wav"]])
            target_wer, target_distance = word_score(target_text, output_text)
            interferer_wer, interferer_distance = word_score(interferer_text, output_text)
            content_switch = (
                interferer_wer is not None
                and target_wer is not None
                and interferer_wer < target_wer
            )
            mos = quality_scores[record["output_wav"]]
            source_row = manifest[record["trial_id"]]
            record.update(
                {
                    "target_text": target_text,
                    "interferer_text": interferer_text,
                    "output_text": output_text,
                    "target_WER": target_wer,
                    "target_word_distance": target_distance,
                    "interferer_WER": interferer_wer,
                    "interferer_word_distance": interferer_distance,
                    "content_switch": content_switch,
                    "interferer_leakage": leakage(target_text, interferer_text, output_text),
                    "target_utterance": source_row.get("target_utterance"),
                    **mos,
                }
            )
        except Exception as error:  # noqa: BLE001
            record["decode_status"] = "failed"
            record["failure_reason"] = f"evaluator {type(error).__name__}: {error}"
        records.append(record)
        if index == 1 or index % 200 == 0 or index == len(audio_rows):
            print(f"metrics={index}/{len(audio_rows)}", flush=True)

    per_trial_path = args.output_dir / "per_trial_metrics.jsonl"
    with per_trial_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with (transcript_dir / "transcripts.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({
                key: record.get(key)
                for key in ("trial_id", "target_text", "interferer_text", "output_text")
            }) + "\n")
    decoded = sum(row["decode_status"] == "ok" for row in records)
    summary = {
        "expected": len(manifest),
        "attempted": len(records),
        "decoded": decoded,
        "failed": len(records) - decoded,
        "missing": len(manifest) - len(records),
        "asr_model": str(args.asr_model.resolve()),
        "asr_language": "English-only model",
        "asr_decoding": "English-only greedy, num_beams=1, do_sample=false",
        "text_normalization": "lowercase, ASCII punctuation removal, whitespace collapse",
        "wer_reference": "same-ASR clean target waveform (ASR-consistency WER)",
        "quality_metric": "DNSMOS P.808 (CPU ONNX)",
        "full_dev": summarize(records),
        "qc_valid_pass": summarize([row for row in records if row["qc_status"] == "PASS"]),
        "qc_counts": {
            status: sum(row.get("qc_status") == status for row in records)
            for status in ("PASS", "RETRY", "FAIL", "MISSING")
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if decoded == len(manifest) else 2


if __name__ == "__main__":
    raise SystemExit(main())
