#!/usr/bin/env python3
"""Resume-safe synthesis and acoustic evaluation for selected-evidence systems."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resource_guard import ResourceGuardStop, check, check_or_raise  # noqa: E402
from se_align.codec.cosyvoice3_codec import CosyVoice3Codec  # noqa: E402
from se_align.eval.metrics import pesq_wb, secs, si_sdr, stoi_score  # noqa: E402
from se_align.utils.audio import load_wav, resample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--selected-waveform-field",
        choices=(
            "evidence_wav", "mixture_wav", "tfmap_context_waveform",
            "primary_selected_waveform",
            "cdcs2_direct_waveform", "cdcs5_direct_waveform",
        ),
    )
    source.add_argument("--token-records", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--qc-metrics", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--cosyvoice-model", type=Path,
        default=ROOT / "pretrained/Fun-CosyVoice3-0.5B",
    )
    parser.add_argument(
        "--cosyvoice-root", type=Path, default=ROOT / "external/CosyVoice"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1986)
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


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in trial_id
    )
    return f"{safe[:100]}-{digest}"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_rss_bytes() -> int:
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def audio16(path: Path) -> np.ndarray:
    waveform, sample_rate = load_wav(str(path))
    return resample(waveform, sample_rate, 16000).reshape(-1).numpy()


def atomic_embedding(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def cached_embedding(codec: CosyVoice3Codec, path: Path, cache: Path) -> np.ndarray:
    if cache.is_file():
        values = np.load(cache, allow_pickle=False)
        if values.size > 0 and np.isfinite(values).all():
            return values
    waveform, sample_rate = load_wav(str(path))
    values = codec.extract_spk_emb(waveform, sample_rate).numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_embedding(cache, values)
    return values


def aligned_metrics(
    target: np.ndarray, mixture: np.ndarray, output: np.ndarray,
) -> dict[str, float | int]:
    length = min(target.size, mixture.size, output.size)
    if length <= 0:
        raise ValueError("empty acoustic overlap")
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


def valid_record(row: dict[str, Any], trial_id: str) -> bool:
    try:
        numeric = (
            "sim_target", "sim_interferer", "speaker_margin", "si_sdr_db",
            "si_sdri_db", "stoi", "pesq_wb",
        )
        return (
            row.get("trial_id") == trial_id
            and row.get("decode_status") == "ok"
            and Path(row["output_wav"]).is_file()
            and all(np.isfinite(float(row[key])) for key in numeric)
        )
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    ids = [row["trial_id"] for row in rows]
    if len(rows) != args.expected or len(set(ids)) != args.expected:
        raise ValueError(
            f"expected {args.expected} unique evaluation rows, found {len(set(ids))}"
        )
    if any(row.get("split") != args.split for row in rows):
        raise ValueError(f"evaluation manifest is not exclusively split={args.split}")
    qc_rows = read_jsonl(args.qc_metrics)
    qc_ids = [row["trial_id"] for row in qc_rows]
    if len(qc_ids) != len(set(qc_ids)):
        raise ValueError("QC metrics contain duplicate trial IDs")
    qc = {row["trial_id"]: row for row in qc_rows}
    missing_qc = set(ids) - set(qc)
    if missing_qc:
        raise ValueError(
            f"QC metrics are missing {len(missing_qc)} selected {args.split.upper()} trial IDs"
        )
    token_records = None
    if args.token_records is not None:
        token_rows = read_jsonl(args.token_records)
        token_records = {row["trial_id"]: row for row in token_rows}
        if len(token_records) != args.expected or set(token_records) != set(ids):
            raise ValueError(f"token records do not exactly cover selected {args.split.upper()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.output_dir / "wav"
    embedding_dir = args.output_dir / "speaker_embeddings"
    wav_dir.mkdir(exist_ok=True)
    embedding_dir.mkdir(exist_ok=True)
    progress_path = args.output_dir / "audio_progress.jsonl"
    output_path = args.output_dir / "audio_metrics.jsonl"
    failures_path = args.output_dir / "audio_failures.jsonl"
    guard_log = args.guard_log
    guard_log.parent.mkdir(parents=True, exist_ok=True)
    phase = f"selected_audio_{args.system_name}"

    prior_rows: list[dict[str, Any]] = []
    for source in (progress_path, output_path):
        if source.is_file():
            prior_rows.extend(read_jsonl(source))
    records_by_id = {
        row["trial_id"]: row for row in prior_rows
        if row.get("trial_id") in set(ids) and valid_record(row, row["trial_id"])
    }
    pending = [row for row in rows if row["trial_id"] not in records_by_id]
    preflight = check_or_raise(
        phase=phase, disk_path=ROOT, log_path=guard_log,
        starting_new_stage=bool(pending),
    )

    codec = None
    if pending:
        codec = CosyVoice3Codec(
            model_dir=str(args.cosyvoice_model),
            cv3_root=str(args.cosyvoice_root),
            device=args.device,
            flow_steps=args.flow_steps,
            seed=args.seed,
        )
        loaded = check(
            phase=f"{phase}_model_loaded", disk_path=ROOT,
            log_path=guard_log, starting_new_stage=False,
        )
        if loaded["evaluation"]["decision"] == "GRACEFUL_STOP":
            raise ResourceGuardStop("; ".join(loaded["evaluation"]["stop_reasons"]))

    # Controlled SNR views and many natural trials reuse enrollment files.
    # Cache only the frozen prompt representation, with a bounded LRU to keep
    # memory stable. CosyVoice3Codec.decode resets the registered seed on every
    # call, so reuse cannot advance or share stochastic state between trials.
    prompt_cache: OrderedDict[str, Any] = OrderedDict()
    # The codec rejects prompts over 30 s, bounding each cached prompt below
    # roughly 1 MiB of feature/token/embedding tensors. A 2,048-entry LRU is
    # therefore host-memory safe here (<2 GiB worst case, normally far less)
    # and captures nearly all repeated enrollment use in the frozen ordering.
    prompt_cache_limit = 2048
    prompt_cache_hits = 0
    prompts_prepared = 0
    prompt_cache_live_checks = 0

    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    last_guard = time.monotonic()
    guard_stop: str | None = None
    try:
        for index, row in enumerate(pending, 1):
            trial_id = row["trial_id"]
            name = safe_name(trial_id)
            generated_path = wav_dir / f"{name}.wav"
            try:
                if codec is None:
                    raise RuntimeError("codec unavailable for pending evaluation")
                if args.selected_waveform_field is not None:
                    system_wav = Path(row[args.selected_waveform_field])
                    if not system_wav.is_file():
                        raise FileNotFoundError(system_wav)
                    token_path = None
                    synthesis_seconds = 0.0
                else:
                    if token_records is None:
                        raise RuntimeError("token records unavailable")
                    token_path = Path(token_records[trial_id]["token_path"])
                    tokens = np.load(token_path, allow_pickle=False).reshape(-1)
                    if (
                        tokens.size == 0 or not np.issubdtype(tokens.dtype, np.integer)
                        or int(tokens.min()) < 0 or int(tokens.max()) >= 6561
                    ):
                        raise ValueError("invalid raw S3 tokens")
                    if not generated_path.is_file():
                        synthesis_started = time.monotonic()
                        enrollment_key = str(Path(row["enrollment_wav"]).resolve())
                        prepared_prompt = prompt_cache.pop(enrollment_key, None)
                        cache_hit = prepared_prompt is not None
                        if cache_hit:
                            prompt_cache_hits += 1
                        else:
                            prompt, prompt_rate = load_wav(enrollment_key)
                            prepared_prompt = codec.prepare_prompt(
                                prompt, prompt_rate, prompt_strategy="self"
                            )
                            prompts_prepared += 1
                            del prompt
                        prompt_cache[enrollment_key] = prepared_prompt
                        if len(prompt_cache) > prompt_cache_limit:
                            prompt_cache.popitem(last=False)
                        waveform, sample_rate = codec.decode(
                            tokens, prepared_prompt=prepared_prompt
                        )
                        # Fail closed on the first two live hits: a freshly
                        # prepared prompt must produce the exact same tensor.
                        if cache_hit and prompt_cache_live_checks < 2:
                            prompt_fresh, prompt_rate_fresh = load_wav(enrollment_key)
                            prepared_fresh = codec.prepare_prompt(
                                prompt_fresh, prompt_rate_fresh,
                                prompt_strategy="self",
                            )
                            waveform_fresh, sample_rate_fresh = codec.decode(
                                tokens, prepared_prompt=prepared_fresh
                            )
                            if (
                                sample_rate_fresh != sample_rate
                                or waveform_fresh.shape != waveform.shape
                                or not torch.equal(waveform_fresh, waveform)
                            ):
                                raise ValueError(
                                    "cached enrollment prompt changed frozen codec output"
                                )
                            prompt_cache_live_checks += 1
                            del prompt_fresh, prepared_fresh, waveform_fresh
                        temporary = generated_path.with_suffix(".tmp.wav")
                        sf.write(
                            temporary, waveform.squeeze(0).numpy(), sample_rate,
                            subtype="PCM_16", format="WAV",
                        )
                        with temporary.open("rb") as handle:
                            os.fsync(handle.fileno())
                        temporary.replace(generated_path)
                        synthesis_seconds = time.monotonic() - synthesis_started
                        del prepared_prompt, waveform
                    else:
                        synthesis_seconds = 0.0
                    system_wav = generated_path
                info = sf.info(system_wav)
                if info.frames <= 0:
                    raise ValueError("empty output waveform")

                target_path = Path(row["target_wav"])
                mixture_path = Path(row["mixture_wav"])
                interferer_path = Path(
                    row["interferer_wav"]
                    if row.get("interferer_wav")
                    else row["interferer_wavs"][0]
                )
                target_key = hashlib.sha1(str(target_path).encode()).hexdigest()
                interferer_key = hashlib.sha1(str(interferer_path).encode()).hexdigest()
                target_embedding = cached_embedding(
                    codec, target_path,
                    args.reference_cache_dir / f"{target_key}.npy",
                )
                interferer_embedding = cached_embedding(
                    codec, interferer_path,
                    args.reference_cache_dir / f"{interferer_key}.npy",
                )
                output_waveform, output_rate = load_wav(str(system_wav))
                output_embedding = codec.extract_spk_emb(
                    output_waveform, output_rate
                ).numpy()
                atomic_embedding(embedding_dir / f"{name}.npy", output_embedding)
                sim_target = secs(target_embedding, output_embedding)
                sim_interferer = secs(interferer_embedding, output_embedding)
                record = {
                    "trial_id": trial_id,
                    "split": args.split,
                    "cohort": row["cohort"],
                    "system": args.system_name,
                    "selected_candidate": (
                        "full" if args.selected_waveform_field == "evidence_wav"
                        else "mixture" if args.selected_waveform_field == "mixture_wav"
                        else "tfmap_context_full" if args.selected_waveform_field == "tfmap_context_waveform"
                        else (
                            row.get(args.selected_waveform_field.replace("waveform", "candidate"))
                            or row.get("selected", {}).get(
                                args.selected_waveform_field.split("_selected_waveform")[0]
                            )
                        ) if args.selected_waveform_field else None
                    ),
                    "token_path": str(token_path.resolve()) if token_path else None,
                    "target_spk": row["target_speaker"],
                    "interferer_spk": (
                        row["interferer_speaker"]
                        if row.get("interferer_speaker") is not None
                        else row["interferer_speakers"][0]
                    ),
                    "qc_status": qc[trial_id].get("waveform_qc_status", "MISSING"),
                    "decode_status": "ok",
                    "failure_reason": None,
                    "output_wav": str(system_wav.resolve()),
                    "output_sample_rate": int(info.samplerate),
                    "output_num_samples": int(info.frames),
                    "synthesis_seconds": synthesis_seconds,
                    "target_wav": str(target_path.resolve()),
                    "interferer_wav": str(interferer_path.resolve()),
                    "mixture_wav": str(mixture_path.resolve()),
                    "sim_target": sim_target,
                    "sim_interferer": sim_interferer,
                    "speaker_margin": sim_target - sim_interferer,
                    "acoustic_speaker_switch": sim_target < sim_interferer,
                    **aligned_metrics(
                        audio16(target_path), audio16(mixture_path), audio16(system_wav)
                    ),
                    "test_used": args.split == "test",
                    "process_rss_bytes": process_rss_bytes(),
                    "cuda_allocated_bytes": (
                        int(torch.cuda.memory_allocated())
                        if torch.cuda.is_available() else 0
                    ),
                    "cuda_reserved_bytes": (
                        int(torch.cuda.memory_reserved())
                        if torch.cuda.is_available() else 0
                    ),
                }
                if not valid_record(record, trial_id):
                    raise RuntimeError("post-write audio record validation failed")
                append_jsonl(progress_path, record)
                records_by_id[trial_id] = record
                del output_waveform, output_embedding
            except Exception as error:  # noqa: BLE001
                failures.append({
                    "trial_id": trial_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
            if index % args.status_every == 0:
                gc.collect()
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
            done = len(records_by_id)
            if index == 1 or done % args.status_every == 0 or index == len(pending):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"audio={done}/{len(rows)} new_rate={index / elapsed:.3f}/s "
                    f"failures={len(failures)}",
                    flush=True,
                )
    except ResourceGuardStop as error:
        guard_stop = str(error)
        print(f"RESOURCE_GUARD_STOP: {guard_stop}", flush=True)
    finally:
        before = check(
            phase=f"{phase}_before_unload", disk_path=ROOT,
            log_path=guard_log, starting_new_stage=False,
        )
        if codec is not None:
            del codec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        after = check(
            phase=f"{phase}_after_unload", disk_path=ROOT,
            log_path=guard_log, starting_new_stage=False,
        )
        (args.output_dir / "audio_unload_snapshot.json").write_text(
            json.dumps({"before": before, "after": after}, indent=2) + "\n",
            encoding="utf-8",
        )

    ordered = [
        records_by_id[trial_id] for trial_id in ids if trial_id in records_by_id
    ]
    atomic_jsonl(output_path, ordered)
    atomic_jsonl(failures_path, failures)
    summary = {
        "status": (
            "COMPLETE" if len(ordered) == args.expected and not failures
            else "RESOURCE_GUARD_STOP" if guard_stop else "PARTIAL"
        ),
        "system": args.system_name,
        "manifest": str(args.manifest.resolve()),
        "source": args.selected_waveform_field or str(args.token_records.resolve()),
        "expected": args.expected,
        "unique": len({row["trial_id"] for row in ordered}),
        "decoded": len(ordered),
        "missing": args.expected - len(ordered),
        "duplicate": len(ordered) - len({row["trial_id"] for row in ordered}),
        "failed": len(failures),
        "resumed_valid": len(ordered) - min(len(pending), len(ordered)),
        "flow_steps": args.flow_steps if token_records is not None else None,
        "seed": args.seed,
        "batch_size": 1,
        "num_workers": 0,
        "prompt_cache_limit": prompt_cache_limit,
        "unique_prompt_preparations": prompts_prepared,
        "prompt_cache_hits": prompt_cache_hits,
        "prompt_cache_live_exact_checks": prompt_cache_live_checks,
        "resource_guard_stop": guard_stop,
        "preflight": preflight["evaluation"]["decision"],
        "test_used": args.split == "test",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "audio_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] == "COMPLETE":
        return 0
    return 3 if guard_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
