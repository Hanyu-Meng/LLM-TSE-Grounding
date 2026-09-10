#!/usr/bin/env python3
"""Run frozen WeSep for deterministic enrollment views on locked natural DEV."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resource_guard import ResourceGuardStop, check, check_or_raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "analysis/candidate_gate/candidate_input_manifest.jsonl",
    )
    parser.add_argument(
        "--wesep-repo", type=Path, default=ROOT / "external/wesep-real-tse"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "pretrained/wesep/spk_emb_100"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", action="append", required=True)
    parser.add_argument("--enrollment-view", action="append", required=True)
    parser.add_argument("--reuse-primary", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-workers", type=int, default=1)
    parser.add_argument("--model-batch-size", type=int, default=0,
                        help="0 batches all requested views; use 1 for OOM-safe noisy runs")
    parser.add_argument("--resource-seconds", type=float, default=20.0)
    parser.add_argument(
        "--guard-log", type=Path,
        default=ROOT / "analysis/candidate_gate/resource_guard.jsonl",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compatibility() -> None:
    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *_args, **_kwargs: None
    if "torchaudio.sox_effects" not in sys.modules:
        module = types.ModuleType("torchaudio.sox_effects")
        module.apply_effects_file = lambda *_args, **_kwargs: None
        module.apply_effects_tensor = lambda *_args, **_kwargs: None
        sys.modules["torchaudio.sox_effects"] = module


def load_mono16(path: str) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if int(sample_rate) != 16000:
        raise ValueError(f"input must be 16 kHz: {path}")
    waveform = values.mean(axis=1, dtype=np.float64).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid waveform: {path}")
    return waveform


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    readable = "".join(c if c.isalnum() or c in "-_" else "_" for c in trial_id)
    return f"{readable[:92]}-{digest}"


def write_wav(path: str, waveform: np.ndarray) -> None:
    sf.write(path, waveform, 16000, subtype="PCM_16")


def main() -> int:
    args = parse_args()
    if len(args.candidate_name) != len(args.enrollment_view):
        raise ValueError("candidate-name and enrollment-view counts must match")
    if len(set(args.candidate_name)) != len(args.candidate_name):
        raise ValueError("candidate names must be unique")
    if args.write_workers > 1 or args.model_batch_size < 0:
        raise ValueError("resource protocol requires write_workers<=1 and model_batch_size>=0")
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("empty candidate input manifest")
    if any(row.get("split") != args.split for row in rows):
        raise ValueError(f"candidate input is not exclusively split={args.split}")

    import torch

    preflight = check_or_raise(
        phase=f"frozen_wesep_candidates_{args.split}",
        disk_path=ROOT,
        log_path=args.guard_log,
        starting_new_stage=True,
    )

    compatibility()
    repo = str(args.wesep_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import wesep

    extractor = wesep.load_model_local(str(args.checkpoint.resolve()))
    extractor.set_resample_rate(16000)
    extractor.set_vad(False)
    extractor.set_device(args.device)
    model = extractor.model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.candidate_name:
        (args.output_dir / "wav" / name).mkdir(parents=True, exist_ok=True)

    records = []
    started = time.monotonic()
    cpu_started = time.process_time()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    inferred = reused = 0
    model_forward_calls = 0
    total_mixture_seconds = 0.0
    writer = ThreadPoolExecutor(max_workers=args.write_workers)
    writes: list[Future[None]] = []
    last_guard = time.monotonic()
    for index, row in enumerate(rows, 1):
        paths: dict[str, str] = {}
        pending_names = []
        pending_views = []
        for name, view in zip(args.candidate_name, args.enrollment_view):
            if view == "primary":
                if not args.reuse_primary:
                    raise ValueError("primary view requires --reuse-primary")
                source = Path(row["primary_output_wav"])
                if not source.is_file():
                    raise FileNotFoundError(source)
                paths[name] = str(source.resolve())
                reused += 1
                continue
            if view not in row["enrollment_views"]:
                raise KeyError(f"unknown enrollment view {view}")
            destination = args.output_dir / "wav" / name / f"{safe_name(row['trial_id'])}.wav"
            paths[name] = str(destination.resolve())
            if not destination.is_file():
                pending_names.append(name)
                pending_views.append(view)
        if pending_names:
            mixture = load_mono16(row["mixture_wav"])
            total_mixture_seconds += mixture.size / 16000.0
            enrollments = [
                load_mono16(row["enrollment_views"][view]) for view in pending_views
            ]
            lengths = {values.size for values in enrollments}
            if len(lengths) != 1 and args.model_batch_size != 1:
                raise ValueError(f"batched enrollment lengths differ: {row['trial_id']}")
            batch_limit = args.model_batch_size or len(enrollments)
            output_rows = []
            for start in range(0, len(enrollments), batch_limit):
                enrollment_chunk = enrollments[start:start + batch_limit]
                mixture_tensor = torch.from_numpy(mixture).reshape(1, 1, -1)
                mixture_tensor = mixture_tensor.repeat(len(enrollment_chunk), 1, 1).to(
                    extractor.device
                )
                enrollment_tensor = torch.stack(
                    [torch.from_numpy(values) for values in enrollment_chunk]
                ).unsqueeze(1).to(extractor.device)
                with torch.inference_mode():
                    outputs = model(mixture_tensor, enrollment_tensor)
                    model_forward_calls += 1
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    if outputs.ndim == 2:
                        outputs = outputs.unsqueeze(1)
                    output_rows.extend(outputs.detach().cpu().float()[:, 0])
                del mixture_tensor, enrollment_tensor, outputs
            for name, waveform in zip(pending_names, output_rows):
                peak = float(waveform.abs().max())
                if peak <= 0.0 or not torch.isfinite(waveform).all():
                    raise RuntimeError(f"invalid frozen output: {row['trial_id']} {name}")
                waveform = waveform / peak * 0.9
                writes.append(writer.submit(write_wav, paths[name], waveform.numpy().copy()))
                if len(writes) >= max(1, args.write_workers * 4):
                    writes.pop(0).result()
                inferred += 1
            del output_rows
        records.append(
            {
                "trial_id": row["trial_id"],
                "split": args.split,
                "cohort": row["cohort"],
                "checkpoint": str(args.checkpoint.resolve()),
                "candidate_paths": paths,
            }
        )
        if index == 1 or index % args.progress_interval == 0 or index == len(rows):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"trials={index}/{len(rows)} inferred={inferred} rate={index / elapsed:.2f}/s",
                flush=True,
            )
        now = time.monotonic()
        if now - last_guard >= args.resource_seconds:
            state = check(
                phase=f"frozen_wesep_candidates_{args.split}",
                disk_path=ROOT,
                log_path=args.guard_log,
                starting_new_stage=False,
            )
            last_guard = now
            if state["evaluation"]["decision"] == "GRACEFUL_STOP":
                raise ResourceGuardStop("; ".join(state["evaluation"]["stop_reasons"]))
    for future in writes:
        future.result()
    writer.shutdown(wait=True)
    index_path = args.output_dir / "candidate_paths.jsonl"
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    elapsed_seconds = time.monotonic() - started
    cpu_seconds = time.process_time() - cpu_started
    output_paths = {
        path for record in records for path in record["candidate_paths"].values()
    }
    summary = {
        "status": "COMPLETE",
        "trials": len(records),
        "candidate_names": args.candidate_name,
        "enrollment_views": args.enrollment_view,
        "checkpoint": str(args.checkpoint.resolve()),
        "waveforms_inferred": inferred,
        "primary_paths_reused": reused,
        "elapsed_seconds": elapsed_seconds,
        "latency_seconds_per_trial": elapsed_seconds / len(records),
        "total_mixture_audio_seconds": total_mixture_seconds,
        "candidate_generation_rtf": (
            elapsed_seconds / total_mixture_seconds if total_mixture_seconds else None
        ),
        "model_forward_calls": model_forward_calls,
        "model_batch_size": args.model_batch_size or len(args.candidate_name),
        "cpu_process_seconds": cpu_seconds,
        "average_process_cpu_percent": 100.0 * cpu_seconds / max(elapsed_seconds, 1e-9),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_torch_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
        ),
        "peak_torch_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if args.device.startswith("cuda") else 0
        ),
        "waveform_storage_bytes": sum(Path(path).stat().st_size for path in output_paths),
        "split": args.split,
        "test_used": args.split == "test",
        "training_used": False,
        "resource_preflight": preflight["evaluation"]["decision"],
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    del model, extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    check(
        phase=f"frozen_wesep_candidates_{args.split}_after_unload",
        disk_path=ROOT,
        log_path=args.guard_log,
        starting_new_stage=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
