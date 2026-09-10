#!/usr/bin/env python3
"""Synthesize and score Stage-1 TSE waveforms with one fixed audio protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from se_align.codec.cosyvoice3_codec import CosyVoice3Codec  # noqa: E402
from se_align.eval.metrics import pesq_wb, secs, si_sdr, stoi_score  # noqa: E402
from se_align.utils.audio import load_wav, resample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--system", choices=("wesep", "wesep_s3", "tokens"), required=True)
    parser.add_argument("--token-records", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qc-metrics", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path, required=True)
    parser.add_argument("--cosyvoice-model", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in trial_id)
    return f"{safe[:100]}-{digest}"


def absolute(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def audio16(path: Path) -> np.ndarray:
    waveform, sample_rate = load_wav(str(path))
    return resample(waveform, sample_rate, 16000).reshape(-1).numpy()


def cached_embedding(codec: CosyVoice3Codec, path: Path, cache_path: Path) -> np.ndarray:
    if cache_path.is_file():
        return np.load(cache_path, allow_pickle=False)
    waveform, sample_rate = load_wav(str(path))
    embedding = codec.extract_spk_emb(waveform, sample_rate).numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    return embedding


def aligned_metrics(target: np.ndarray, mixture: np.ndarray, output: np.ndarray) -> dict:
    length = min(target.size, mixture.size, output.size)
    target = target[:length]
    mixture = mixture[:length]
    output = output[:length]
    output_sdr = si_sdr(target, output)
    mixture_sdr = si_sdr(target, mixture)
    return {
        "si_sdr_db": output_sdr,
        "si_sdri_db": output_sdr - mixture_sdr,
        "stoi": stoi_score(target, output, 16000),
        "pesq_wb": pesq_wb(target, output, 16000),
        "metric_overlap_samples": length,
    }


def main() -> int:
    args = parse_args()
    if args.system == "tokens" and args.token_records is None:
        raise ValueError("--token-records is required for --system tokens")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.output_dir / "wav"
    embedding_dir = args.output_dir / "speaker_embeddings"
    wav_dir.mkdir(exist_ok=True)
    embedding_dir.mkdir(exist_ok=True)

    all_rows = read_jsonl(args.manifest)
    rows = all_rows[args.shard_index::args.num_shards]
    qc = {row["trial_id"]: row for row in read_jsonl(args.qc_metrics)}
    token_records = (
        {row["trial_id"]: row for row in read_jsonl(args.token_records)}
        if args.token_records else {}
    )
    codec = CosyVoice3Codec(
        model_dir=str(args.cosyvoice_model),
        cv3_root=str(args.cosyvoice_root),
        device=args.device,
        flow_steps=args.flow_steps,
        seed=args.seed,
    )
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    records: list[dict] = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        trial_id = row["trial_id"]
        name = safe_name(trial_id)
        output_path = wav_dir / f"{name}.wav"
        record = {
            "trial_id": trial_id,
            "target_spk": row["target_speaker"],
            "interferer_spk": row["interferer_speakers"][0],
            "qc_status": qc.get(trial_id, {}).get("waveform_qc_status", "MISSING"),
            "decode_status": "failed",
            "failure_reason": None,
            "output_wav": str(output_path.resolve()),
        }
        try:
            if not output_path.is_file():
                if args.system == "wesep":
                    source = absolute(row["evidence_wav"])
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    os.symlink(source.resolve(), output_path)
                else:
                    token_path = (
                        absolute(row["evidence_token_path"])
                        if args.system == "wesep_s3"
                        else absolute(token_records[trial_id]["token_path"])
                    )
                    tokens = np.load(token_path, allow_pickle=False).reshape(-1)
                    enrollment_path = str(absolute(row["enrollment_wav"]).resolve())
                    prepared_prompt = prompt_cache.get(enrollment_path)
                    if prepared_prompt is None:
                        prompt, prompt_sr = load_wav(enrollment_path)
                        prepared_prompt = codec.prepare_prompt(
                            prompt, prompt_sr, prompt_strategy="self"
                        )
                        prompt_cache[enrollment_path] = prepared_prompt
                    waveform, sample_rate = codec.decode(
                        tokens,
                        prepared_prompt=prepared_prompt,
                    )
                    sf.write(
                        output_path,
                        waveform.squeeze(0).numpy(),
                        sample_rate,
                        subtype="PCM_16",
                    )

            target_path = absolute(row["target_wav"])
            mixture_path = absolute(row["mixture_wav"])
            interferer_path = absolute(row["interferer_wavs"][0])
            target_key = hashlib.sha1(str(target_path).encode()).hexdigest()
            interferer_key = hashlib.sha1(str(interferer_path).encode()).hexdigest()
            target_emb = cached_embedding(
                codec, target_path, args.reference_cache_dir / f"{target_key}.npy"
            )
            interferer_emb = cached_embedding(
                codec, interferer_path, args.reference_cache_dir / f"{interferer_key}.npy"
            )
            output_waveform, output_sr = load_wav(str(output_path))
            output_emb = codec.extract_spk_emb(output_waveform, output_sr).numpy()
            np.save(embedding_dir / f"{name}.npy", output_emb)
            sim_target = secs(target_emb, output_emb)
            sim_interferer = secs(interferer_emb, output_emb)
            record.update(
                {
                    "target_wav": str(target_path),
                    "interferer_wav": str(interferer_path),
                    "mixture_wav": str(mixture_path),
                    "sim_target": sim_target,
                    "sim_interferer": sim_interferer,
                    "speaker_margin": sim_target - sim_interferer,
                    "acoustic_speaker_switch": sim_target < sim_interferer,
                    **aligned_metrics(
                        audio16(target_path), audio16(mixture_path), audio16(output_path)
                    ),
                    "decode_status": "ok",
                }
            )
        except Exception as error:  # noqa: BLE001
            record["failure_reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
        if index == 1 or index % 20 == 0 or index == len(rows):
            elapsed = max(time.monotonic() - started, 1e-6)
            ok = sum(item["decode_status"] == "ok" for item in records)
            print(f"attempted={index}/{len(rows)} ok={ok} rate={index / elapsed:.3f}/s", flush=True)

    shard_suffix = (
        f".shard-{args.shard_index}-of-{args.num_shards}"
        if args.num_shards > 1 else ""
    )
    metrics_path = args.output_dir / f"audio_metrics{shard_suffix}.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    decoded = sum(row["decode_status"] == "ok" for row in records)
    summary = {
        "system": args.system,
        "manifest": str(args.manifest.resolve()),
        "global_expected": len(all_rows),
        "expected": len(rows),
        "attempted": len(records),
        "decoded": decoded,
        "failed": len(records) - decoded,
        "missing": len(rows) - len(records),
        "qc_counts": {
            status: sum(row["qc_status"] == status for row in records)
            for status in ("PASS", "RETRY", "FAIL", "MISSING")
        },
        "flow_steps": args.flow_steps if args.system != "wesep" else None,
        "seed": args.seed,
        "unique_prompts_prepared": len(prompt_cache),
        "elapsed_seconds": time.monotonic() - started,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (args.output_dir / f"audio_summary{shard_suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if decoded == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
