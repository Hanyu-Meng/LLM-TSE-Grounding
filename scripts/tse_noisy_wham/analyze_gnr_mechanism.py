#!/usr/bin/env python3
"""DEV-only evaluation analysis of adaptive-anchor geometry and GNR candidates."""

from __future__ import annotations

import argparse
import gc
import json
import os
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
from scripts.tse_noisy_wham.decode_noisy_grounding import fsq_digits  # noqa: E402
from scripts.tse_selected_evidence.decode_selected_evidence import (  # noqa: E402
    SelectedEvidenceCollator,
    SelectedEvidenceDataset,
    build_model,
    hamming_digits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "manifests/noisy_wham/dev_full_inference.jsonl",
    )
    parser.add_argument(
        "--evaluation", type=Path,
        default=ROOT / "manifests/noisy_wham/dev_full_evaluation.jsonl",
    )
    parser.add_argument(
        "--anchor-records", type=Path,
        default=ROOT / "results/noisy_wham/dev/systems/pool_d_residual_csg/per_trial_tokens.jsonl",
    )
    parser.add_argument(
        "--gnr-records", type=Path,
        default=ROOT / "results/noisy_wham/dev/systems/pool_d_residual_csg_gnr_k50_r3/per_trial_tokens.jsonl",
    )
    parser.add_argument(
        "--prepared-dev-manifest", type=Path,
        default=ROOT / "manifests/tse_dev_prepared.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis/noisy_wham/dev/full/gnr_mechanism",
    )
    parser.add_argument("--expected", type=int, default=8400)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "experiments/qfull_sme_rawqwen_seed0_20260810_0858/checkpoints/best.pt",
    )
    parser.add_argument(
        "--qwen", type=Path,
        default=ROOT / "pretrained/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--wavlm", type=Path,
        default=ROOT / "pretrained/wavlm-base-plus",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--status-every", type=int, default=100)
    parser.add_argument("--resource-seconds", type=float, default=20.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n"); handle.flush(); os.fsync(handle.fileno())


def atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def keyed(path: Path, expected: int) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path); result = {row["trial_id"]: row for row in rows}
    if len(rows) != expected or len(result) != expected:
        raise ValueError(f"coverage failure: {path}")
    return result


def load_tokens(path: Path, expected: int) -> torch.LongTensor:
    values = np.load(path, allow_pickle=False).reshape(-1)
    if (
        len(values) != expected or not np.issubdtype(values.dtype, np.integer)
        or int(values.min()) < 0 or int(values.max()) >= 6561
    ):
        raise ValueError(f"invalid evaluation token sequence: {path}")
    return torch.from_numpy(values.astype(np.int64, copy=False).copy())


def load_clean_tokens(path: Path, generated_length: int) -> torch.LongTensor:
    """Load canonical clean tokens under the frozen tail-boundary policy."""
    values = np.load(path, allow_pickle=False).reshape(-1)
    shortfall = generated_length - len(values)
    if (
        shortfall not in (0, 1)
        or len(values) <= 0
        or not np.issubdtype(values.dtype, np.integer)
        or int(values.min()) < 0
        or int(values.max()) >= 6561
    ):
        raise ValueError(
            f"invalid clean evaluation token sequence: {path}: "
            f"clean={len(values)} generated={generated_length}"
        )
    return torch.from_numpy(values.astype(np.int64, copy=False).copy())


def valid(row: dict[str, Any], trial_id: str) -> bool:
    required = (
        "evidence_to_clean_mean_hamming", "anchor_to_clean_mean_hamming",
        "anchor_to_evidence_mean_hamming", "clean_candidate_coverage",
        "anchor_already_clean_rate", "oracle_correction_potential",
        "accepted_edit_rate", "average_hamming_edit_distance",
    )
    return (
        row.get("trial_id") == trial_id
        and all(np.isfinite(float(row[key])) for key in required)
        and int(row.get("generated_positions", -1))
        - int(row.get("positions", -1)) in (0, 1)
        and row.get("clean_token_alignment_policy")
        == "left-aligned shared codec frames; exclude at most one final boundary frame"
        and row.get("clean_reference_used") is True
        and not row.get("test_used")
    )


@torch.no_grad()
def analyze_one(model: Any, batch: dict[str, torch.Tensor], anchor: torch.LongTensor,
                refined: torch.LongTensor, clean: torch.LongTensor,
                k: int, radius: int) -> dict[str, Any]:
    device = batch["mixture_values"].device
    length = int(batch["output_lengths"][0])
    anchor_full = anchor.to(device)
    padded = anchor_full.unsqueeze(0)
    mixture_features, mixture_lengths = model._encode_mixture(
        batch["mixture_values"], batch["mixture_attention_mask"]
    )
    inputs, attention_mask, labels, _ = model._build_embeddings(
        mixture_features, mixture_lengths, batch["speaker_embeddings"],
        batch["evidence_tokens"], batch["evidence_lengths"], padded,
        batch["output_lengths"],
    )
    hidden = model.llm.model(
        inputs_embeds=inputs, attention_mask=attention_mask
    ).last_hidden_state
    positions = labels[0] != -100
    output_weight = model.llm.get_output_embeddings().weight
    audio_weight = output_weight[
        model.vocab.audio_shift:model.vocab.audio_shift + model.vocab.audio_vocabsize
    ]
    logits = hidden[0, positions] @ audio_weight.T
    if logits.shape != (length, 6561) or not torch.isfinite(logits).all():
        raise ValueError("invalid evaluation-only teacher-forced logits")
    comparison_length = clean.numel()
    if length - comparison_length not in (0, 1):
        raise ValueError("clean/GNR token framing differs by more than one tail frame")
    logits = logits[:comparison_length]
    clean_device = clean.to(device)
    anchor_device = anchor_full[:comparison_length]
    evidence = batch["evidence_tokens"][0, :comparison_length].long()
    clean_score = logits.gather(1, clean_device[:, None]).squeeze(1)
    ranks = (logits > clean_score[:, None]).sum(dim=1) + 1
    clean_anchor_hamming = hamming_digits(clean_device, anchor_device)
    covered = (clean_device == anchor_device) | (
        (ranks <= k) & (clean_anchor_hamming <= radius)
    )
    wrong = anchor_device != clean_device
    refined_device = refined.to(device)[:comparison_length]
    corrected = wrong & (refined_device == clean_device)
    worsened = (~wrong) & (refined_device != clean_device)
    bins = {
        "rank_1": int((ranks == 1).sum()),
        "rank_2_5": int(((ranks >= 2) & (ranks <= 5)).sum()),
        "rank_6_10": int(((ranks >= 6) & (ranks <= 10)).sum()),
        "rank_11_20": int(((ranks >= 11) & (ranks <= 20)).sum()),
        "rank_21_50": int(((ranks >= 21) & (ranks <= 50)).sum()),
        "rank_gt_50": int((ranks > 50).sum()),
    }
    return {
        "positions": comparison_length,
        "generated_positions": length,
        "boundary_positions_excluded": length - comparison_length,
        "clean_token_alignment_policy": (
            "left-aligned shared codec frames; exclude at most one final boundary frame"
        ),
        "evidence_to_clean_mean_hamming": float(
            hamming_digits(evidence, clean_device).float().mean()
        ),
        "anchor_to_clean_mean_hamming": float(clean_anchor_hamming.float().mean()),
        "anchor_to_evidence_mean_hamming": float(
            hamming_digits(anchor_device, evidence).float().mean()
        ),
        "clean_candidate_coverage": float(covered.float().mean()),
        "anchor_already_clean_rate": float((~wrong).float().mean()),
        "oracle_correction_potential": (
            float((covered & wrong).sum() / wrong.sum()) if bool(wrong.any()) else 0.0
        ),
        "accepted_clean_correction_rate": (
            float(corrected.sum() / wrong.sum()) if bool(wrong.any()) else 0.0
        ),
        "accepted_edit_rate": float((refined_device != anchor_device).float().mean()),
        "clean_position_worsened_rate": float(worsened.float().mean()),
        "average_hamming_edit_distance": float(
            hamming_digits(refined_device, anchor_device).float().mean()
        ),
        "clean_token_llm_rank_bins": bins,
        "clean_token_llm_mean_rank": float(ranks.float().mean()),
        "candidate_rule": f"Top-{k} intersect Hamming-ball-{radius}, union anchor",
        "teacher_forced_history": "immutable_anchor_prefix",
        "refined_tokens_fed_back": False,
        "clean_reference_used": True,
        "test_used": False,
    }


def main() -> int:
    args = parse_args()
    dataset = SelectedEvidenceDataset(args.manifest, "pool_d", "dev")
    if len(dataset) != args.expected:
        raise ValueError("DEV mechanism dataset coverage failure")
    evaluation = keyed(args.evaluation, args.expected)
    base_ids = {row["base_trial_id"] for row in evaluation.values()}
    prepared_dev = keyed(args.prepared_dev_manifest, len(base_ids))
    if set(prepared_dev) != base_ids:
        raise ValueError("prepared DEV target-token manifest coverage failure")
    for base_trial_id, row in prepared_dev.items():
        path = Path(row.get("target_token_path", ""))
        if row.get("split") != "dev" or not path.is_file():
            raise ValueError(
                f"invalid prepared DEV target-token row: {base_trial_id}: {path}"
            )
    anchors = keyed(args.anchor_records, args.expected)
    gnr = keyed(args.gnr_records, args.expected)
    ids = [row["trial_id"] for row in dataset.rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir / "progress.jsonl"
    records_path = args.output_dir / "per_trial.jsonl"
    prior = {}
    for path in (progress, records_path):
        if path.is_file():
            for row in read_jsonl(path):
                if row.get("trial_id") in set(ids) and valid(row, row["trial_id"]):
                    prior[row["trial_id"]] = row
    pending = [index for index, trial_id in enumerate(ids) if trial_id not in prior]
    check_or_raise(
        phase="noisy_dev_gnr_mechanism", disk_path=ROOT,
        log_path=ROOT / "analysis/noisy_wham/resource_guard.jsonl",
        starting_new_stage=bool(pending),
    )
    loader = DataLoader(
        Subset(dataset, pending), batch_size=1, shuffle=False, num_workers=0,
        collate_fn=SelectedEvidenceCollator(), pin_memory=True,
    )
    model = build_model(args)[0] if pending else None
    records = dict(prior); failures = []; started = time.monotonic(); last_guard = started
    guard_stop = None
    try:
        for step, raw in enumerate(loader, 1):
            trial_id = raw.pop("trial_ids")[0]
            raw.pop("evidence_token_paths")
            batch = {key: value.to(args.device, non_blocking=True) for key, value in raw.items()}
            try:
                length = int(batch["output_lengths"][0])
                anchor = load_tokens(Path(anchors[trial_id]["token_path"]), length)
                refined = load_tokens(Path(gnr[trial_id]["token_path"]), length)
                base_trial_id = evaluation[trial_id]["base_trial_id"]
                clean_path = Path(prepared_dev[base_trial_id]["target_token_path"])
                clean = load_clean_tokens(clean_path, length)
                values = analyze_one(
                    model, batch, anchor, refined, clean, args.k, args.radius
                )
                record = {
                    "trial_id": trial_id, "base_trial_id": base_trial_id,
                    "benchmark": evaluation[trial_id]["benchmark"],
                    "snr_db": evaluation[trial_id].get("snr_db"),
                    "gender_cohort": evaluation[trial_id]["gender_cohort"],
                    "clean_target_token_path": str(clean_path),
                    "clean_target_token_path_source": str(args.prepared_dev_manifest),
                    **values,
                }
                if not valid(record, trial_id):
                    raise ValueError("post-analysis validation failed")
                append(progress, record); records[trial_id] = record
            except Exception as error:  # noqa: BLE001
                failures.append({
                    "trial_id": trial_id, "error_type": type(error).__name__,
                    "error": str(error),
                })
            del batch
            now = time.monotonic()
            if now - last_guard >= args.resource_seconds:
                resource = check(
                    phase="noisy_dev_gnr_mechanism", disk_path=ROOT,
                    log_path=ROOT / "analysis/noisy_wham/resource_guard.jsonl",
                    starting_new_stage=False,
                )
                last_guard = now
                if resource["evaluation"]["decision"] == "GRACEFUL_STOP":
                    guard_stop = "; ".join(resource["evaluation"]["stop_reasons"])
                    raise ResourceGuardStop(guard_stop)
            if len(records) % args.status_every == 0 or step == len(pending):
                print(
                    f"gnr_mechanism={len(records)}/{len(dataset)} failures={len(failures)} "
                    f"rate={step/max(now-started,1e-6):.3f}/s", flush=True,
                )
    except ResourceGuardStop as error:
        guard_stop = str(error)
    finally:
        if model is not None:
            del model
        del loader; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        check(
            phase="noisy_dev_gnr_mechanism_after_unload", disk_path=ROOT,
            log_path=ROOT / "analysis/noisy_wham/resource_guard.jsonl",
            starting_new_stage=False,
        )
    ordered = [records[trial_id] for trial_id in ids if trial_id in records]
    atomic(records_path, "".join(json.dumps(row) + "\n" for row in ordered))
    atomic(args.output_dir / "failures.jsonl", "".join(json.dumps(row) + "\n" for row in failures))
    keys = (
        "evidence_to_clean_mean_hamming", "anchor_to_clean_mean_hamming",
        "anchor_to_evidence_mean_hamming", "clean_candidate_coverage",
        "anchor_already_clean_rate", "oracle_correction_potential",
        "accepted_clean_correction_rate", "accepted_edit_rate",
        "clean_position_worsened_rate", "average_hamming_edit_distance",
        "clean_token_llm_mean_rank",
    )
    rank_totals = {
        key: sum(row["clean_token_llm_rank_bins"][key] for row in ordered)
        for key in ("rank_1", "rank_2_5", "rank_6_10", "rank_11_20", "rank_21_50", "rank_gt_50")
    }
    positions = sum(row["positions"] for row in ordered)
    boundary_exclusions = [row["boundary_positions_excluded"] for row in ordered]
    accepted_margins = [
        float(value)
        for trial_id in ids if trial_id in gnr
        for value in gnr[trial_id].get("accepted_logit_margins", [])
    ]
    edit_by_snr = {}
    for snr in (-5, 0, 5, 10, 15):
        subset = [
            gnr[trial_id] for trial_id in ids
            if evaluation[trial_id].get("benchmark") == "controlled"
            and int(evaluation[trial_id]["snr_db"]) == snr
        ]
        edit_by_snr[str(snr)] = {
            "trials": len(subset),
            "mean_edit_rate": float(np.mean([
                row["gnr_edit_rate"] for row in subset
            ])) if subset else None,
            "mean_fraction_positions_unchanged": float(np.mean([
                row["fraction_positions_unchanged"] for row in subset
            ])) if subset else None,
        }
    summary = {
        "status": "COMPLETE" if len(ordered) == args.expected and not failures else "RESOURCE_GUARD_STOP" if guard_stop else "PARTIAL",
        "expected": args.expected, "completed": len(ordered), "failures": len(failures),
        "mean": {key: float(np.mean([row[key] for row in ordered])) if ordered else None for key in keys},
        "rank_bin_position_rates": {key: value/positions for key, value in rank_totals.items()} if positions else {},
        "gnr_k": args.k, "gnr_r": args.radius,
        "median_accepted_llm_score_margin": (
            float(np.median(accepted_margins)) if accepted_margins else 0.0
        ),
        "accepted_edit_positions": len(accepted_margins),
        "mean_fraction_positions_unchanged": float(np.mean([
            gnr[trial_id]["fraction_positions_unchanged"] for trial_id in ids
        ])) if ids else None,
        "controlled_edit_fraction_by_snr": edit_by_snr,
        "target_token_source_manifest": str(args.prepared_dev_manifest),
        "target_token_source_trials": len(prepared_dev),
        "token_alignment_policy": (
            "left-aligned shared codec frames; exclude at most one final boundary frame"
        ),
        "trials_with_one_boundary_position_excluded": int(sum(boundary_exclusions)),
        "total_boundary_positions_excluded": int(sum(boundary_exclusions)),
        "maximum_boundary_positions_excluded_per_trial": (
            max(boundary_exclusions) if boundary_exclusions else None
        ),
        "clean_reference_used_for_inference": False,
        "clean_reference_used_for_evaluation_analysis": True,
        "test_used": False, "resource_guard_stop": guard_stop,
    }
    atomic(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "COMPLETE" else 3 if guard_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
