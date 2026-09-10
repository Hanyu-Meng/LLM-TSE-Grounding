#!/usr/bin/env python3
"""Score candidate content with the frozen natural-DEV Whisper protocol."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic-metrics", type=Path, required=True)
    parser.add_argument(
        "--asr-model", type=Path, default=PROJECT_ROOT / "pretrained/whisper-small.en"
    )
    parser.add_argument(
        "--base-cache", type=Path, default=PROJECT_ROOT / "dev_metrics/asr_transcript_cache.jsonl"
    )
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-trials", type=int, default=5991)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def load_mono16(path: str) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if int(sample_rate) != 16000:
        raise ValueError(f"expected 16 kHz audio: {path}")
    return values.mean(axis=1, dtype=np.float64).astype(np.float32)


def load_cache(*paths: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            audio_path = row["audio_path"]
            cache[audio_path] = row["text"]
            # The frozen WeSep paths in the old cache are symlinks while the
            # candidate index stores their canonical targets. Treat identical
            # files as one cached waveform; no transcript metric changes.
            cache[str(Path(audio_path).resolve())] = row["text"]
    return cache


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


def transcribe_missing(
    asr, paths: list[str], cache: dict[str, str], cache_path: Path, batch_size: int
) -> int:
    missing = list(dict.fromkeys(path for path in paths if path not in cache))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for start in range(0, len(missing), batch_size):
        batch_paths = missing[start : start + batch_size]
        inputs = [{"raw": load_mono16(path), "sampling_rate": 16000} for path in batch_paths]
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
        if completed == len(missing) or completed // 256 != start // 256:
            print(
                f"asr={completed}/{len(missing)} rate={completed / max(time.monotonic() - started, 1e-6):.2f}/s",
                flush=True,
            )
    return len(missing)


def word_score(reference: str, hypothesis: str) -> tuple[float | None, int | None]:
    if not reference:
        return None, None
    score = jiwer.process_words(reference, hypothesis)
    return float(score.wer), int(score.substitutions + score.deletions + score.insertions)


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.acoustic_metrics)
    if len(rows) != args.expected_trials or len({row["trial_id"] for row in rows}) != args.expected_trials:
        raise ValueError(
            f"acoustic metrics must contain {args.expected_trials:,} unique DEV trials"
        )
    if any(row.get("split") != args.split for row in rows):
        raise ValueError(f"input is not exclusively split={args.split}")
    paths: list[str] = []
    for row in rows:
        paths.extend((row["target_wav"], row["interferer_wav"]))
        paths.extend(metrics["output_wav"] for metrics in row["candidates"].values())
    cache = load_cache(args.base_cache, args.candidate_cache)
    existing_count = sum(path in cache for path in dict.fromkeys(paths))
    asr = build_asr(args)
    started = time.monotonic()
    missing_count = transcribe_missing(asr, paths, cache, args.candidate_cache, args.batch_size)

    records = []
    for row in rows:
        target_text = normalize(cache[row["target_wav"]])
        interferer_text = normalize(cache[row["interferer_wav"]])
        for candidate, metrics in row["candidates"].items():
            output_text = normalize(cache[metrics["output_wav"]])
            target_wer, target_distance = word_score(target_text, output_text)
            interferer_wer, interferer_distance = word_score(interferer_text, output_text)
            records.append(
                {
                    "trial_id": row["trial_id"],
                    "split": args.split,
                    "cohort": row["cohort"],
                    "candidate": candidate,
                    "output_wav": metrics["output_wav"],
                    "target_text": target_text,
                    "interferer_text": interferer_text,
                    "output_text": output_text,
                    "target_WER": target_wer,
                    "target_word_distance": target_distance,
                    "interferer_WER": interferer_wer,
                    "interferer_word_distance": interferer_distance,
                    "content_switch": (
                        target_wer is not None
                        and interferer_wer is not None
                        and interferer_wer < target_wer
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    summary = {
        "status": "COMPLETE",
        "split": args.split,
        "trials": len(rows),
        "candidate_outputs": len(records),
        "unique_paths_preexisting_in_cache": existing_count,
        "unique_paths_transcribed": missing_count,
        "asr_model": str(args.asr_model.resolve()),
        "asr_language": "English-only model",
        "asr_decoding": "English-only greedy, num_beams=1, do_sample=false",
        "text_normalization": "lowercase, ASCII punctuation removal, whitespace collapse",
        "wer_reference": "same-ASR clean target waveform (ASR-consistency WER)",
        "elapsed_seconds": time.monotonic() - started,
        "test_used": args.split == "test",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
