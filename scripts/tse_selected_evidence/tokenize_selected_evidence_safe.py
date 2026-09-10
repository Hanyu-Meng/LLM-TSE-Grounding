#!/usr/bin/env python3
"""Resume-safe, single-pool selected-evidence alignment and S3 tokenization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resource_guard import ResourceGuardStop, check, check_or_raise  # noqa: E402
from se_align.codec.cosyvoice3_codec import CosyVoice3S3Tokenizer  # noqa: E402


ANALYSIS = ROOT / "analysis/selected_evidence"
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
POOLS = {
    "pool_full": ("full",),
    "pool_b": ("full", "tfmap_context_full"),
    "pool_d": ORDER,
}
FORBIDDEN_KEY_PARTS = (
    "target", "interferer", "transcript", "sisdr", "si_sdr", "qc", "reference", "label"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=tuple(POOLS), required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ANALYSIS / "full_dev_candidates.jsonl"
    )
    parser.add_argument(
        "--cosyvoice-model", type=Path, default=ROOT / "pretrained/Fun-CosyVoice3-0.5B"
    )
    parser.add_argument(
        "--provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
    )
    parser.add_argument("--expected", type=int, default=6000)
    parser.add_argument("--status-every", type=int, default=50)
    parser.add_argument("--resource-seconds", type=float, default=20.0)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    parser.add_argument(
        "--status-path", type=Path, default=ROOT / "docs/SELECTED_EVIDENCE_STATUS.md"
    )
    return parser.parse_args()


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    readable = "".join(char if char.isalnum() or char in "-_" else "_" for char in trial_id)
    return f"{readable[:100]}-{digest}"


def load_mono16(path: str) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if int(sample_rate) != 16000:
        raise ValueError(f"expected 16 kHz candidate: {path}")
    waveform = values.mean(axis=1, dtype=np.float64).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid candidate waveform: {path}")
    return waveform


def read_rows(path: Path, expected: int, split: str) -> list[dict[str, Any]]:
    rows = []
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = sorted(
                key for key in row if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS)
            )
            if leaked or row.get("split") != split:
                raise ValueError(
                    f"fail-closed selected-evidence row={index} leaked={leaked} split={row.get('split')}"
                )
            if row["trial_id"] in ids:
                raise ValueError(f"duplicate trial ID: {row['trial_id']}")
            ids.add(row["trial_id"])
            rows.append(row)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} unique {split.upper()} rows, found {len(rows)}")
    return rows


def validate_pair(wav_path: Path, token_path: Path, mixture_path: str) -> tuple[bool, dict[str, Any]]:
    if not wav_path.is_file() or not token_path.is_file():
        return False, {"reason": "missing", "wav_exists": wav_path.is_file(), "token_exists": token_path.is_file()}
    try:
        mixture = sf.info(mixture_path)
        aligned = sf.info(wav_path)
        tokens = np.load(token_path, allow_pickle=False).reshape(-1)
        expected = math.ceil(int(mixture.frames) / 640)
        valid = (
            int(mixture.samplerate) == int(aligned.samplerate) == 16000
            and int(aligned.frames) == int(mixture.frames)
            and tokens.size == expected
            and tokens.size > 0
            and np.issubdtype(tokens.dtype, np.integer)
            and int(tokens.min()) >= 0
            and int(tokens.max()) < 6561
        )
        return valid, {
            "reason": "valid" if valid else "shape_or_range",
            "mixture_samples": int(mixture.frames),
            "aligned_samples": int(aligned.frames),
            "expected_tokens": expected,
            "tokens": int(tokens.size),
        }
    except Exception as error:  # noqa: BLE001
        return False, {"reason": f"{type(error).__name__}: {error}"}


def atomic_wav(path: Path, waveform: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.wav")
    sf.write(temporary, waveform, 16000, subtype="PCM_16", format="WAV")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def status_markdown(
    *, pool: str, done: int, expected: int, failed: int, last_trial: str | None,
    next_trial: str | None, snapshot: dict[str, Any], log_path: Path, state: str,
    reason: str | None = None, test_used: bool = False,
) -> str:
    values = snapshot["snapshot"]
    gpu = values["gpu"]
    gpu_used = gpu.get("used_bytes", 0) / 1024 ** 3 if gpu.get("available") else None
    return f"""# Selected Evidence Integration Status

Updated: {values['timestamp']}

- SYSTEM: selected-evidence integration
- POOL: {pool}
- STAGE: deterministic alignment + CosyVoice3 S3 tokenization
- STATE: {state}
- DONE: {done}
- EXPECTED: {expected}
- FAILED: {failed}
- BATCH_SIZE: 1
- NUM_WORKERS: 0
- PERSISTENT_WORKERS: false
- RAM_USED: {values['host']['used_bytes'] / 1024 ** 3:.2f} GiB ({values['host']['used_percent']:.2f}%)
- SWAP_USED: {values['swap']['used_bytes'] / 1024 ** 3:.2f} GiB ({values['swap']['used_percent']:.2f}%)
- GPU_USED: {gpu_used if gpu_used is not None else 'unavailable'} GiB
- DISK_FREE: {values['disk']['free_bytes'] / 1024 ** 3:.2f} GiB
- LAST_TRIAL: {last_trial or 'NONE'}
- NEXT_TRIAL: {next_trial or 'NONE'}
- LOG: {log_path}
- RESOURCE_GUARD_STOP: {'YES' if state == 'RESOURCE_GUARD_STOP' else 'NO'}
- REASON: {reason or 'NONE'}
- TEST_USED: {'YES' if test_used else 'NO'}
"""


def write_status(destination: Path, **kwargs: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.md")
    temporary.write_text(status_markdown(**kwargs), encoding="utf-8")
    temporary.replace(destination)


def append_progress(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    analysis = args.analysis_dir
    phase = f"selected_{args.pool}_tokenize"
    guard_log = analysis / "resource_guard.jsonl"
    stage_log = analysis / f"evidence/{args.pool}/progress.jsonl"
    log_path = analysis / f"evidence/{args.pool}/tokenize_safe.log"
    wav_dir = analysis / f"evidence/{args.pool}/wav"
    token_dir = analysis / f"evidence/{args.pool}/tokens"
    wav_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    stage_log.parent.mkdir(parents=True, exist_ok=True)

    preflight = check_or_raise(
        phase=phase, disk_path=ROOT, log_path=guard_log, starting_new_stage=True
    )
    rows = read_rows(args.manifest, args.expected, args.split)
    complete = []
    for index, row in enumerate(rows):
        stem = safe_name(row["trial_id"])
        valid, _details = validate_pair(
            wav_dir / f"{stem}.wav", token_dir / f"{stem}.npy", row["mixture_wav"]
        )
        if valid:
            complete.append(index)
    done = len(complete)
    failed = 0
    last_trial = rows[max(complete)]["trial_id"] if complete else None
    next_row = next((row for index, row in enumerate(rows) if index not in set(complete)), None)
    write_status(args.status_path,
        pool=args.pool, done=done, expected=args.expected, failed=failed,
        last_trial=last_trial, next_trial=next_row["trial_id"] if next_row else None,
        snapshot=preflight, log_path=log_path, state="RUNNING",
        test_used=args.split == "test",
    )

    tokenizer = CosyVoice3S3Tokenizer(str(args.cosyvoice_model), provider=args.provider)
    last_guard = time.monotonic()
    started = time.monotonic()
    complete_set = set(complete)
    latest_snapshot = preflight
    guard_reason = None
    try:
        for index, row in enumerate(rows):
            if index in complete_set:
                continue
            trial_id = row["trial_id"]
            stem = safe_name(trial_id)
            wav_path = wav_dir / f"{stem}.wav"
            token_path = token_dir / f"{stem}.npy"
            mixture = sf.info(row["mixture_wav"])
            if int(mixture.samplerate) != 16000:
                raise ValueError(f"mixture is not 16 kHz: {trial_id}")
            mixture_samples = int(mixture.frames)
            selected_name = row["selected"][args.pool]
            if selected_name not in POOLS[args.pool]:
                raise ValueError(f"selected candidate outside {args.pool}: {trial_id}")

            wav_valid = False
            if wav_path.is_file():
                try:
                    info = sf.info(wav_path)
                    wav_valid = int(info.samplerate) == 16000 and int(info.frames) == mixture_samples
                except Exception:  # noqa: BLE001
                    wav_valid = False
            if wav_valid:
                aligned = load_mono16(str(wav_path))
                waveform_action = "resume_existing_aligned_wav"
                source_samples = None
            else:
                source = row["candidates"][selected_name]["waveform_path"]
                waveform = load_mono16(source)
                source_samples = int(waveform.size)
                if waveform.size >= mixture_samples:
                    aligned = waveform[:mixture_samples]
                    waveform_action = "none" if waveform.size == mixture_samples else "tail_truncate"
                else:
                    aligned = np.pad(waveform, (0, mixture_samples - waveform.size))
                    waveform_action = "tail_zero_pad"
                atomic_wav(wav_path, aligned)
                del waveform

            encoded_tensor = tokenizer.encode(torch.from_numpy(aligned).unsqueeze(0), 16000)
            encoded = encoded_tensor.cpu().numpy().reshape(-1)
            expected_tokens = math.ceil(mixture_samples / 640)
            raw_length = int(encoded.size)
            difference = raw_length - expected_tokens
            if difference == 1:
                encoded = encoded[:expected_tokens]
                token_action = "tail_truncate_one"
            elif difference == -1:
                encoded = np.concatenate([encoded, encoded[-1:]])
                token_action = "repeat_final_token_once"
            elif difference == 0:
                token_action = "none"
            else:
                raise ValueError(
                    f"token length difference {difference} outside preregistered policy: {trial_id}"
                )
            if (
                encoded.size != expected_tokens or encoded.size == 0
                or int(encoded.min()) < 0 or int(encoded.max()) >= 6561
            ):
                raise ValueError(f"invalid S3 result: {trial_id}")
            atomic_npy(token_path, encoded.astype(np.int32, copy=False))
            valid, details = validate_pair(wav_path, token_path, row["mixture_wav"])
            if not valid:
                raise ValueError(f"post-write validation failed: {trial_id}: {details}")
            done += 1
            last_trial = trial_id
            next_trial = rows[index + 1]["trial_id"] if index + 1 < len(rows) else None
            append_progress(stage_log, {
                "trial_index": index,
                "trial_id": trial_id,
                "pool": args.pool,
                "selected_candidate": selected_name,
                "source_num_samples": source_samples,
                "mixture_num_samples": mixture_samples,
                "waveform_action": waveform_action,
                "raw_evidence_tokens": raw_length,
                "expected_tokens": expected_tokens,
                "token_action": token_action,
                "token_path": str(token_path),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "target_length_used": False,
                "test_used": args.split == "test",
            })
            del aligned, encoded_tensor, encoded
            gc.collect()

            now = time.monotonic()
            if now - last_guard >= args.resource_seconds:
                latest_snapshot = check(
                    phase=phase, disk_path=ROOT, log_path=guard_log,
                    starting_new_stage=False,
                )
                last_guard = now
                if latest_snapshot["evaluation"]["decision"] == "GRACEFUL_STOP":
                    guard_reason = "; ".join(latest_snapshot["evaluation"]["stop_reasons"])
                    raise ResourceGuardStop(guard_reason)
            if done % args.status_every == 0 or next_trial is None:
                write_status(args.status_path,
                    pool=args.pool, done=done, expected=args.expected, failed=failed,
                    last_trial=last_trial, next_trial=next_trial,
                    snapshot=latest_snapshot, log_path=log_path, state="RUNNING",
                    test_used=args.split == "test",
                )
                print(
                    f"pool={args.pool} done={done}/{args.expected} failed={failed} "
                    f"rate_new={(done - len(complete)) / max(now - started, 1e-6):.3f}/s",
                    flush=True,
                )
    except ResourceGuardStop as error:
        guard_reason = str(error)
        latest_snapshot = check(
            phase=phase, disk_path=ROOT, log_path=guard_log, starting_new_stage=False
        )
        next_trial = next(
            (row["trial_id"] for index, row in enumerate(rows) if index >= done), None
        )
        write_status(args.status_path,
            pool=args.pool, done=done, expected=args.expected, failed=failed,
            last_trial=last_trial, next_trial=next_trial,
            snapshot=latest_snapshot, log_path=log_path,
            state="RESOURCE_GUARD_STOP", reason=guard_reason,
            test_used=args.split == "test",
        )
        print("RESOURCE_GUARD_STOP = YES", flush=True)
        print(f"REASON: {guard_reason}", flush=True)
        print(f"LAST_COMPLETED_TRIAL: {last_trial}", flush=True)
        print(f"RESUME_FROM: {next_trial}", flush=True)
        return 3
    except Exception as error:  # noqa: BLE001
        failed += 1
        latest_snapshot = check(
            phase=phase, disk_path=ROOT, log_path=guard_log, starting_new_stage=False
        )
        write_status(args.status_path,
            pool=args.pool, done=done, expected=args.expected, failed=failed,
            last_trial=last_trial, next_trial=None,
            snapshot=latest_snapshot, log_path=log_path,
            state="FAILED", reason=f"{type(error).__name__}: {error}",
            test_used=args.split == "test",
        )
        raise
    finally:
        before_unload = check(
            phase=f"{phase}_before_unload", disk_path=ROOT, log_path=guard_log,
            starting_new_stage=False,
        )
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        after_unload = check(
            phase=f"{phase}_after_unload", disk_path=ROOT, log_path=guard_log,
            starting_new_stage=False,
        )
        unload_path = analysis / f"evidence/{args.pool}/unload_snapshot.json"
        unload_path.write_text(
            json.dumps({"before": before_unload, "after": after_unload}, indent=2) + "\n"
        )

    final = check(
        phase=f"{phase}_complete", disk_path=ROOT, log_path=guard_log,
        starting_new_stage=False,
    )
    write_status(args.status_path,
        pool=args.pool, done=done, expected=args.expected, failed=failed,
        last_trial=last_trial, next_trial=None, snapshot=final,
        log_path=log_path, state="COMPLETE", test_used=args.split == "test",
    )
    summary = {
        "status": "COMPLETE",
        "pool": args.pool,
        "expected": args.expected,
        "valid_completed": done,
        "failed": failed,
        "resumed_valid_outputs": len(complete),
        "new_outputs": done - len(complete),
        "provider": args.provider,
        "batch_size": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "target_length_used": False,
        "test_used": args.split == "test",
        "elapsed_seconds": time.monotonic() - started,
    }
    (analysis / f"evidence/{args.pool}/tokenization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
