#!/usr/bin/env python3
"""Synthesize selected-evidence tokens without loading clean references."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from se_align.codec.cosyvoice3_codec import CosyVoice3Codec  # noqa: E402
from se_align.data.store import read_manifest  # noqa: E402
from se_align.utils.audio import load_wav  # noqa: E402


FORBIDDEN_SUBSTRINGS = (
    "target", "interferer", "transcript", "sisdr", "si_sdr", "qc", "reference", "label"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--token-records", type=Path)
    source.add_argument(
        "--manifest-token-field",
        choices=("cdcs2_evidence_token_path", "cdcs5_evidence_token_path"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cosyvoice-model", type=Path, default=ROOT / "pretrained/Fun-CosyVoice3-0.5B"
    )
    parser.add_argument("--cosyvoice-root", type=Path, default=ROOT / "external/CosyVoice")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1986)
    return parser.parse_args()


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in trial_id)
    return f"{safe[:100]}-{digest}"


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("empty deployment manifest")
    for index, row in enumerate(rows):
        leaked = sorted(
            key for key in row if any(value in key.lower() for value in FORBIDDEN_SUBSTRINGS)
        )
        if leaked or row.get("split") != "dev":
            raise ValueError(f"fail-closed deployment row={index} leaked={leaked}")
    ids = [row["trial_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate deployment trial IDs")
    token_records: dict[str, dict[str, Any]] = {}
    if args.token_records is not None:
        with args.token_records.open(encoding="utf-8") as handle:
            token_records = {
                row["trial_id"]: row for row in (json.loads(line) for line in handle if line.strip())
            }
        if set(token_records) != set(ids):
            raise ValueError("token records do not exactly match deployment manifest")

    codec = CosyVoice3Codec(
        model_dir=str(args.cosyvoice_model), cv3_root=str(args.cosyvoice_root),
        device=args.device, flow_steps=args.flow_steps, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.output_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    prompt_cache: dict[str, Any] = {}
    records = []
    failures = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        trial_id = row["trial_id"]
        output_path = wav_dir / f"{safe_name(trial_id)}.wav"
        try:
            token_path = (
                token_records[trial_id]["token_path"]
                if args.token_records is not None else row[args.manifest_token_field]
            )
            tokens = np.load(token_path, allow_pickle=False).reshape(-1)
            if tokens.size == 0 or tokens.min() < 0 or tokens.max() >= 6561:
                raise ValueError("invalid raw S3 token input")
            if not output_path.is_file():
                enrollment = row["enrollment_wav"]
                prepared_prompt = prompt_cache.get(enrollment)
                if prepared_prompt is None:
                    prompt, prompt_sr = load_wav(enrollment)
                    prepared_prompt = codec.prepare_prompt(prompt, prompt_sr, prompt_strategy="self")
                    prompt_cache[enrollment] = prepared_prompt
                waveform, sample_rate = codec.decode(tokens, prepared_prompt=prepared_prompt)
                sf.write(
                    output_path, waveform.squeeze(0).numpy(), sample_rate, subtype="PCM_16"
                )
            info = sf.info(output_path)
            records.append({
                "trial_id": trial_id,
                "split": "dev",
                "token_path": str(token_path),
                "token_count": int(tokens.size),
                "output_wav": str(output_path),
                "output_sample_rate": int(info.samplerate),
                "output_num_samples": int(info.frames),
                "synthesis_status": "ok",
            })
        except Exception as error:  # noqa: BLE001
            failures.append({
                "trial_id": trial_id,
                "error_type": type(error).__name__,
                "error": str(error),
            })
        if index == 1 or index % 20 == 0 or index == len(rows):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(f"synthesized={index}/{len(rows)} rate={index / elapsed:.3f}/s failures={len(failures)}", flush=True)

    with (args.output_dir / "synthesis_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "synthesis_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "status": "COMPLETE" if len(records) == len(rows) and not failures else "FAILED",
        "manifest": str(args.manifest),
        "manifest_fields": sorted(rows[0]),
        "token_source": str(args.token_records) if args.token_records else args.manifest_token_field,
        "expected": len(rows),
        "synthesized": len(records),
        "failed": len(failures),
        "flow_steps": args.flow_steps,
        "seed": args.seed,
        "unique_prompts_prepared": len(prompt_cache),
        "clean_reference_used": False,
        "test_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "synthesis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
