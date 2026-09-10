#!/usr/bin/env python3
"""Resume-safe adaptive CSG and fixed-anchor GNR decoding for noisy TSE."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.resource_guard import ResourceGuardStop, check, check_or_raise  # noqa: E402
from scripts.tse_selected_evidence.decode_selected_evidence import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    SelectedEvidenceCollator,
    SelectedEvidenceDataset,
    atomic_jsonl,
    atomic_npy,
    build_model,
    hamming_digits,
    process_rss_bytes,
    sha256,
    token_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", choices=("pool_full", "pool_b", "pool_d"), default="pool_d")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "experiments/qfull_sme_rawqwen_seed0_20260810_0858/checkpoints/best.pt")
    parser.add_argument("--qwen", type=Path, default=ROOT / "pretrained/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--wavlm", type=Path, default=ROOT / "pretrained/wavlm-base-plus")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("adaptive_csg", "gnr"), required=True)
    parser.add_argument("--lambda-sidecar", type=Path)
    parser.add_argument("--anchor-records", type=Path)
    parser.add_argument("--temporal-tolerance", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gnr-k", type=int, default=50)
    parser.add_argument("--gnr-r", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--status-every", type=int, default=100)
    parser.add_argument("--resource-seconds", type=float, default=20.0)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--guard-log", type=Path, default=ROOT / "analysis/noisy_wham/resource_guard.jsonl")
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


def load_token_records(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {row["trial_id"]: row for row in rows}
    if len(rows) != len(result) or set(result) != expected_ids:
        raise ValueError(f"token record coverage mismatch: {path}")
    return result


def load_tokens(path: str, expected_length: int) -> torch.LongTensor:
    values = np.load(path, allow_pickle=False).reshape(-1)
    if (
        values.size != expected_length
        or not np.issubdtype(values.dtype, np.integer)
        or int(values.min()) < 0
        or int(values.max()) >= 6561
    ):
        raise ValueError(f"invalid token sequence: {path}")
    return torch.from_numpy(values.astype(np.int64, copy=False).copy())


def valid_record(row: dict[str, Any], trial_id: str, token_dir: Path) -> bool:
    try:
        path = Path(row["token_path"])
        values = np.load(path, allow_pickle=False).reshape(-1)
        return (
            row["trial_id"] == trial_id
            and path.parent.resolve() == token_dir.resolve()
            and path.name == token_name(trial_id)
            and values.size == int(row["expected_tokens"])
            and values.size > 0
            and np.issubdtype(values.dtype, np.integer)
            and int(values.min()) >= 0
            and int(values.max()) < 6561
        )
    except Exception:  # noqa: BLE001
        return False


def fsq_digits(ids: torch.Tensor) -> torch.Tensor:
    values = ids.long()
    digits = []
    for _ in range(8):
        digits.append(values % 3)
        values = values // 3
    return torch.stack(digits, dim=-1)


@torch.no_grad()
def generate_adaptive(
    model: Any,
    batch: dict[str, torch.Tensor],
    lambdas: torch.Tensor,
    temporal_tolerance: int,
) -> list[torch.LongTensor]:
    """Greedy Q-Full generation with a fixed per-trial adaptive CSG lambda."""
    model.eval()
    mixture_features, mixture_lengths = model._encode_mixture(
        batch["mixture_values"], batch["mixture_attention_mask"]
    )
    prefixes = model._build_generation_prefixes(
        mixture_features,
        mixture_lengths,
        batch["speaker_embeddings"],
        batch["evidence_tokens"],
        batch["evidence_lengths"],
    )
    lengths = batch["output_lengths"].long()
    evidence_lengths = batch["evidence_lengths"].long()
    if bool((lengths <= 0).any()) or bool((lengths > evidence_lengths).any()):
        raise ValueError("adaptive CSG output/evidence length mismatch")
    batch_size = len(prefixes)
    max_prefix = max(prefix.shape[0] for prefix in prefixes)
    hidden_size = prefixes[0].shape[-1]
    prefix_batch = torch.zeros(
        batch_size, max_prefix, hidden_size,
        dtype=prefixes[0].dtype, device=prefixes[0].device,
    )
    attention_mask = torch.zeros(
        batch_size, max_prefix, dtype=torch.long, device=prefixes[0].device
    )
    prefix_lengths = torch.tensor(
        [prefix.shape[0] for prefix in prefixes], dtype=torch.long, device=prefixes[0].device
    )
    for index, prefix in enumerate(prefixes):
        prefix_batch[index, max_prefix - prefix.shape[0]:] = prefix
        attention_mask[index, max_prefix - prefix.shape[0]:] = 1
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    output_weight = model.llm.get_output_embeddings().weight
    audio_weight = output_weight[
        model.vocab.audio_shift:model.vocab.audio_shift + model.vocab.audio_vocabsize
    ]
    output = model.llm.model(
        inputs_embeds=prefix_batch,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past = output.past_key_values
    hidden = output.last_hidden_state[:, -1]
    candidate_digits = fsq_digits(torch.arange(6561, device=prefix_batch.device))
    generated = torch.zeros(
        batch_size, int(lengths.max()), dtype=torch.long, device=prefix_batch.device
    )
    batch_indices = torch.arange(batch_size, device=prefix_batch.device)
    lambdas = lambdas.to(prefix_batch.device, dtype=torch.float32).reshape(-1)
    if lambdas.numel() != batch_size or bool((lambdas < 0).any()):
        raise ValueError("invalid adaptive lambda vector")

    for step in range(int(lengths.max())):
        logits = hidden @ audio_weight.T
        distances = []
        for delta in range(-temporal_tolerance, temporal_tolerance + 1):
            positions = torch.full_like(evidence_lengths, step + delta)
            positions = torch.maximum(torch.zeros_like(positions), positions)
            positions = torch.minimum(positions, evidence_lengths - 1)
            evidence_ids = batch["evidence_tokens"][batch_indices, positions].long()
            reference = fsq_digits(evidence_ids)
            distances.append(
                (candidate_digits.unsqueeze(0) != reference.unsqueeze(1)).sum(dim=2)
            )
        distance = torch.stack(distances, dim=0).amin(dim=0)
        logits = logits - lambdas[:, None].to(logits.dtype) * distance.to(logits.dtype)
        next_raw = logits.argmax(dim=-1)
        generated[:, step] = next_raw
        if step + 1 == int(lengths.max()):
            break
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, 1, dtype=torch.long, device=attention_mask.device),
        ], dim=1)
        output = model.llm.model(
            inputs_embeds=model._audio_embeddings(next_raw).unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=(prefix_lengths + step).unsqueeze(1),
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        hidden = output.last_hidden_state[:, -1]
    return [generated[index, :int(lengths[index])] for index in range(batch_size)]


@torch.no_grad()
def refine_gnr(
    model: Any,
    batch: dict[str, torch.Tensor],
    anchors: list[torch.LongTensor],
    k: int,
    radius: int,
) -> tuple[list[torch.LongTensor], list[dict[str, Any]]]:
    """Parallel causal teacher-forcing over the immutable anchor prefix."""
    if not (1 <= k <= 6561 and 0 <= radius <= 8):
        raise ValueError("invalid GNR K/R")
    device = batch["mixture_values"].device
    lengths = batch["output_lengths"].long()
    padded = torch.full(
        (len(anchors), int(lengths.max())), -1, dtype=torch.long, device=device
    )
    for index, anchor in enumerate(anchors):
        if anchor.numel() != int(lengths[index]):
            raise ValueError("GNR anchor length mismatch")
        padded[index, :anchor.numel()] = anchor.to(device)
    mixture_features, mixture_lengths = model._encode_mixture(
        batch["mixture_values"], batch["mixture_attention_mask"]
    )
    inputs, attention_mask, labels, _ = model._build_embeddings(
        mixture_features,
        mixture_lengths,
        batch["speaker_embeddings"],
        batch["evidence_tokens"],
        batch["evidence_lengths"],
        padded,
        lengths,
    )
    hidden = model.llm.model(inputs_embeds=inputs, attention_mask=attention_mask).last_hidden_state
    output_weight = model.llm.get_output_embeddings().weight
    audio_weight = output_weight[
        model.vocab.audio_shift:model.vocab.audio_shift + model.vocab.audio_vocabsize
    ]
    results = []
    details = []
    for index, anchor in enumerate(anchors):
        positions = labels[index] != -100
        logits = hidden[index, positions] @ audio_weight.T
        if logits.shape[0] != anchor.numel() or not torch.isfinite(logits).all():
            raise RuntimeError("GNR teacher-forced position/logit audit failed")
        anchor_device = anchor.to(device)
        top_scores, top_ids = torch.topk(logits, k=k, dim=-1)
        distance = (
            fsq_digits(top_ids) != fsq_digits(anchor_device)[:, None, :]
        ).sum(dim=-1)
        allowed = distance <= radius
        refined = anchor_device.clone()
        candidate_sizes = []
        accepted_margins = []
        for step in range(anchor.numel()):
            anchor_id = int(anchor_device[step])
            anchor_score = logits[step, anchor_id]
            ids = top_ids[step, allowed[step]]
            scores = top_scores[step, allowed[step]]
            candidate_sizes.append(int(ids.numel()) + int(not bool((ids == anchor_id).any())))
            if ids.numel():
                best_offset = int(scores.argmax())
                best_id = int(ids[best_offset])
                best_score = scores[best_offset]
                if best_id != anchor_id and bool(best_score > anchor_score):
                    refined[step] = best_id
                    accepted_margins.append(float((best_score - anchor_score).float()))
        distances = hamming_digits(refined, anchor_device)
        results.append(refined)
        details.append({
            "gnr_k": k,
            "gnr_r": radius,
            "anchor_retained_in_every_candidate_set": True,
            "candidate_set_size_min": min(candidate_sizes),
            "candidate_set_size_mean": float(np.mean(candidate_sizes)),
            "candidate_set_size_max": max(candidate_sizes),
            "gnr_edit_rate": float((refined != anchor_device).float().mean()),
            "gnr_edit_count": int((refined != anchor_device).sum()),
            "mean_hamming_edit_distance": float(distances.float().mean()),
            "mean_accepted_logit_margin": (
                float(np.mean(accepted_margins)) if accepted_margins else 0.0
            ),
            "median_accepted_logit_margin": (
                float(np.median(accepted_margins)) if accepted_margins else 0.0
            ),
            "accepted_logit_margins": accepted_margins,
            "fraction_positions_unchanged": float(
                (refined == anchor_device).float().mean()
            ),
            "refined_tokens_fed_back": False,
            "teacher_forced_history": "immutable_anchor_prefix",
        })
    return results, details


def main() -> int:
    args = parse_args()
    if args.batch_size != 1 or args.num_workers > 1:
        raise ValueError("resource protocol requires batch_size=1 and num_workers<=1")
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen Q-Full checkpoint hash mismatch")
    if args.mode == "adaptive_csg" and args.lambda_sidecar is None:
        raise ValueError("adaptive CSG requires --lambda-sidecar")
    if args.mode == "gnr" and args.anchor_records is None:
        raise ValueError("GNR requires --anchor-records")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = SelectedEvidenceDataset(args.manifest, args.pool, args.split)
    if len(dataset) != args.expected:
        raise ValueError(f"expected {args.expected} rows, found {len(dataset)}")
    ids = [row["trial_id"] for row in dataset.rows]
    id_set = set(ids)
    sidecar = None
    if args.lambda_sidecar is not None:
        sidecar_rows = read_jsonl(args.lambda_sidecar)
        sidecar_all = {row["trial_id"]: row for row in sidecar_rows}
        if len(sidecar_rows) != len(sidecar_all) or not id_set.issubset(sidecar_all):
            raise ValueError("lambda sidecar coverage mismatch")
        sidecar = {trial_id: sidecar_all[trial_id] for trial_id in id_set}
    anchors = None
    if args.anchor_records is not None:
        anchors = load_token_records(args.anchor_records, id_set)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_dir = args.output_dir / "tokens"
    token_dir.mkdir(exist_ok=True)
    progress_path = args.output_dir / "decode_progress.jsonl"
    records_path = args.output_dir / "per_trial_tokens.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    prior = {}
    for path in (progress_path, records_path):
        if path.is_file():
            for row in read_jsonl(path):
                if row.get("trial_id") in id_set and valid_record(row, row["trial_id"], token_dir):
                    prior[row["trial_id"]] = row
    pending = [index for index, trial_id in enumerate(ids) if trial_id not in prior]
    phase = f"noisy_{args.mode}_{args.pool}"
    preflight = check_or_raise(
        phase=phase,
        disk_path=ROOT,
        log_path=args.guard_log,
        starting_new_stage=bool(pending),
    )
    loader = DataLoader(
        Subset(dataset, pending),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=SelectedEvidenceCollator(),
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=False,
    )
    model = build_model(args) [0] if pending else None
    records = dict(prior)
    failures = []
    started = time.monotonic()
    last_guard = time.monotonic()
    guard_stop = None
    try:
        for step, raw_batch in enumerate(loader, 1):
            trial_ids = raw_batch.pop("trial_ids")
            evidence_paths = raw_batch.pop("evidence_token_paths")
            batch = {key: value.to(args.device, non_blocking=True) for key, value in raw_batch.items()}
            trial_id = trial_ids[0]
            try:
                if model is None:
                    raise RuntimeError("model unavailable")
                started_trial = time.monotonic()
                if args.mode == "adaptive_csg":
                    value = sidecar[trial_id]
                    generated = generate_adaptive(
                        model,
                        batch,
                        torch.tensor([float(value["selected_lambda"])]),
                        args.temporal_tolerance,
                    )
                    extra = {
                        "difficulty_policy": value.get("policy"),
                        "difficulty_value": value.get("difficulty_value"),
                        "selected_lambda": float(value["selected_lambda"]),
                        "temporal_tolerance": args.temporal_tolerance,
                    }
                else:
                    anchor = load_tokens(
                        anchors[trial_id]["token_path"], int(batch["output_lengths"][0])
                    )
                    generated, detail = refine_gnr(
                        model, batch, [anchor], args.gnr_k, args.gnr_r
                    )
                    extra = detail[0] | {
                        "anchor_token_path": anchors[trial_id]["token_path"],
                    }
                generated_row = generated[0].detach().cpu()
                length = int(batch["output_lengths"][0])
                evidence = batch["evidence_tokens"][0, :length].detach().cpu()
                if generated_row.numel() != length:
                    raise RuntimeError("generated length mismatch")
                destination = token_dir / token_name(trial_id)
                atomic_npy(destination, generated_row.numpy().astype(np.int32, copy=False))
                distances = hamming_digits(generated_row, evidence)
                record = {
                    "trial_id": trial_id,
                    "split": args.split,
                    "mode": args.mode,
                    "candidate_pool": args.pool,
                    "token_path": str(destination.resolve()),
                    "evidence_token_path": evidence_paths[0],
                    "expected_tokens": length,
                    "generated_tokens": int(generated_row.numel()),
                    "token_id_min": int(generated_row.min()),
                    "token_id_max": int(generated_row.max()),
                    "token_flip_rate_vs_evidence": float((generated_row != evidence).float().mean()),
                    "mean_fsq_hamming_vs_evidence": float(distances.float().mean()),
                    "decode_seconds": time.monotonic() - started_trial,
                    "clean_reference_used": False,
                    "target_length_used": False,
                    "test_used": args.split == "test",
                    "process_rss_bytes": process_rss_bytes(),
                    "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
                    "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
                    **extra,
                }
                if not valid_record(record, trial_id, token_dir):
                    raise RuntimeError("post-write token validation failed")
                append_jsonl(progress_path, record)
                records[trial_id] = record
            except Exception as error:  # noqa: BLE001
                failures.append({
                    "trial_id": trial_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
            del batch
            now = time.monotonic()
            if now - last_guard >= args.resource_seconds:
                resource = check(
                    phase=phase,
                    disk_path=ROOT,
                    log_path=args.guard_log,
                    starting_new_stage=False,
                )
                last_guard = now
                if resource["evaluation"]["decision"] == "GRACEFUL_STOP":
                    guard_stop = "; ".join(resource["evaluation"]["stop_reasons"])
                    raise ResourceGuardStop(guard_stop)
            if len(records) % args.status_every == 0 or step == len(pending):
                print(
                    f"{args.mode}={len(records)}/{len(dataset)} failures={len(failures)} "
                    f"rate={step / max(time.monotonic() - started, 1e-6):.3f}/s",
                    flush=True,
                )
    except ResourceGuardStop as error:
        guard_stop = str(error)
        print(f"RESOURCE_GUARD_STOP: {guard_stop}", flush=True)
    finally:
        if model is not None:
            del model
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        check(
            phase=f"{phase}_after_unload",
            disk_path=ROOT,
            log_path=args.guard_log,
            starting_new_stage=False,
        )

    ordered = [records[trial_id] for trial_id in ids if trial_id in records]
    atomic_jsonl(records_path, ordered)
    atomic_jsonl(failures_path, failures)
    summary = {
        "status": "COMPLETE" if len(ordered) == len(dataset) and not failures else "RESOURCE_GUARD_STOP" if guard_stop else "PARTIAL",
        "mode": args.mode,
        "candidate_pool": args.pool,
        "expected": len(dataset),
        "decoded": len(ordered),
        "failures": len(failures),
        "resumed": len(prior),
        "checkpoint_sha256": sha256(args.checkpoint),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "temporal_tolerance": args.temporal_tolerance if args.mode == "adaptive_csg" else None,
        "gnr_k": args.gnr_k if args.mode == "gnr" else None,
        "gnr_r": args.gnr_r if args.mode == "gnr" else None,
        "resource_guard_stop": guard_stop,
        "preflight": preflight["evaluation"]["decision"],
        "elapsed_seconds": time.monotonic() - started,
        "test_used": args.split == "test",
    }
    (args.output_dir / "token_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETE" else 3 if guard_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
