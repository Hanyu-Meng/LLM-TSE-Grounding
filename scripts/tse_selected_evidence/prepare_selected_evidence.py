#!/usr/bin/env python3
"""Prepare, select, align, and tokenize frozen natural-DEV candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/selected_evidence"
CANDIDATE_GATE = ROOT / "analysis/candidate_gate"
PREPARED = ROOT / "manifests/tse_dev_prepared.jsonl"
NATURAL_AUDIT = ROOT / "analysis/wesep_speaker_selection/dev_all_trials.jsonl"
PRIMARY_METRICS = ROOT / "dev_outputs/WeSep/per_trial_metrics.jsonl"
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
POOLS = {
    "primary_only": ("full",),
    "cdcs2": ("full", "tfmap_context_full"),
    "wesep_multiview": ("full", "first", "middle", "final"),
    "cdcs5": ORDER,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="freeze all-DEV inputs and 100-trial smoke IDs")
    assemble = sub.add_parser("assemble", help="join paths, score the 9 ambiguous trials, and select")
    assemble.add_argument(
        "--ambiguous-multiview-index",
        type=Path,
        default=ANALYSIS / "ambiguous_multiview/candidate_paths.jsonl",
    )
    assemble.add_argument(
        "--ambiguous-tfmap-index",
        type=Path,
        default=ANALYSIS / "ambiguous_tfmap/candidate_paths.jsonl",
    )
    assemble.add_argument("--device", default="cuda")
    tokenize = sub.add_parser("tokenize", help="align selected waveforms and create leak-free S3 inputs")
    tokenize.add_argument(
        "--cosyvoice-model", type=Path, default=ROOT / "pretrained/Fun-CosyVoice3-0.5B"
    )
    tokenize.add_argument(
        "--provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False) + "\n")


def safe_name(trial_id: str) -> str:
    digest = hashlib.sha1(trial_id.encode()).hexdigest()[:12]
    readable = "".join(char if char.isalnum() or char in "-_" else "_" for char in trial_id)
    return f"{readable[:100]}-{digest}"


def cohort(audit: dict[str, Any]) -> str:
    if bool(audit["high_confidence_wrong"]):
        return "natural_primary_swap"
    if bool(audit["wrong_margin_0"]):
        return "ambiguous_primary_wrong"
    return "primary_correct_control"


def prepare() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    prepared_rows = read_jsonl(PREPARED)
    audit_rows = read_jsonl(NATURAL_AUDIT)
    primary_rows = read_jsonl(PRIMARY_METRICS)
    old_rows = read_jsonl(CANDIDATE_GATE / "candidate_input_manifest.jsonl")
    if not all(len(rows) == expected for rows, expected in (
        (prepared_rows, 6000), (audit_rows, 6000), (primary_rows, 6000), (old_rows, 5991)
    )):
        raise ValueError("unexpected frozen natural-DEV source counts")
    prepared = {row["trial_id"]: row for row in prepared_rows}
    audit = {row["trial_id"]: row for row in audit_rows}
    primary = {row["trial_id"]: row for row in primary_rows}
    old = {row["trial_id"]: row for row in old_rows}
    if not (set(prepared) == set(audit) == set(primary)):
        raise ValueError("6,000-trial source ID mismatch")

    candidate_script = ROOT / "scripts/tse_candidate_gate"
    sys.path.insert(0, str(candidate_script))
    from build_candidate_gate_inputs import materialize_views  # noqa: PLC0415

    output = []
    ambiguous = []
    ambiguous_evaluation = []
    by_mixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_row in prepared_rows:
        by_mixture[Path(source_row["mixture_wav"]).stem].append(source_row)
    for row in prepared_rows:
        trial_id = row["trial_id"]
        label = cohort(audit[trial_id])
        if trial_id in old:
            views = old[trial_id]["enrollment_views"]
        else:
            details = materialize_views(
                row["enrollment_wav"],
                CANDIDATE_GATE / "enrollment_views",
                segment_seconds=2.0,
                grid_seconds=0.01,
            )
            views = details["view_paths"]
        item = {
            "trial_id": trial_id,
            "split": "dev",
            "cohort": label,
            # Frozen manifests already contain absolute paths. Avoid thousands of
            # unnecessary realpath calls on the shared filesystem.
            "mixture_wav": str(row["mixture_wav"]),
            "enrollment_wav": str(row["enrollment_wav"]),
            "speaker_embedding_path": str(row["speaker_embedding_path"]),
            "enrollment_views": {name: str(path) for name, path in views.items()},
            "primary_output_wav": str(primary[trial_id]["output_wav"]),
        }
        output.append(item)
        if label == "ambiguous_primary_wrong":
            ambiguous.append(item)
            mixture_id = Path(row["mixture_wav"]).stem
            counterpart = next(
                value for value in by_mixture[mixture_id] if value["trial_id"] != trial_id
            )
            ambiguous_evaluation.append({
                "trial_id": trial_id,
                "split": "dev",
                "cohort": label,
                "target_speaker": row["target_speaker"],
                "interferer_speaker": row["interferer_speakers"][0],
                "target_wav": row["target_wav"],
                "interferer_wav": row["interferer_wavs"][0],
                "target_enrollment_embedding_path": row["speaker_embedding_path"],
                "interferer_enrollment_embedding_path": counterpart["speaker_embedding_path"],
            })
    counts = Counter(row["cohort"] for row in output)
    expected = Counter({
        "natural_primary_swap": 405,
        "primary_correct_control": 5586,
        "ambiguous_primary_wrong": 9,
    })
    if counts != expected or len({row["trial_id"] for row in output}) != 6000:
        raise ValueError(f"frozen cohort mismatch: {counts}")
    write_jsonl(ANALYSIS / "all_dev_candidate_inputs.jsonl", output)
    write_jsonl(ANALYSIS / "ambiguous_candidate_inputs.jsonl", ambiguous)
    write_jsonl(ANALYSIS / "ambiguous_candidate_evaluation.jsonl", ambiguous_evaluation)

    rng = random.Random(1986)
    by_cohort = defaultdict(list)
    for row in output:
        by_cohort[row["cohort"]].append(row["trial_id"])
    swaps = rng.sample(sorted(by_cohort["natural_primary_swap"]), 25)
    controls = rng.sample(sorted(by_cohort["primary_correct_control"]), 25)
    ambiguous_ids = sorted(by_cohort["ambiguous_primary_wrong"])
    already = set(swaps + controls + ambiguous_ids)
    remaining = sorted(set(prepared) - already)
    random_ids = rng.sample(remaining, 41)
    smoke_ids = swaps + controls + ambiguous_ids + random_ids
    smoke = {
        "registered_at": "2026-08-12T19:05:00+10:00",
        "status": "LOCKED_BEFORE_INTEGRATION_RESULTS",
        "seed": 1986,
        "selection_basis": "frozen cohort labels and trial IDs only; model results not read",
        "components": {
            "natural_primary_swap": swaps,
            "primary_correct_control": controls,
            "ambiguous_primary_wrong": ambiguous_ids,
            "deterministic_random_from_remaining": random_ids,
        },
        "ordered_trial_ids": smoke_ids,
        "count": len(smoke_ids),
        "test_used": False,
    }
    (ANALYSIS / "preregistered_smoke_subset.json").write_text(
        json.dumps(smoke, indent=2) + "\n", encoding="utf-8"
    )
    output_by_id = {row["trial_id"]: row for row in output}
    write_jsonl(
        ANALYSIS / "smoke_candidate_inputs.jsonl",
        [output_by_id[trial_id] for trial_id in smoke_ids],
    )
    print(json.dumps({"status": "COMPLETE", "counts": counts, "ambiguous": 9, "smoke": 100}, default=dict, indent=2))
    return 0


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
        raise ValueError(f"expected 16 kHz audio: {path}")
    waveform = values.mean(axis=1, dtype=np.float64).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid audio: {path}")
    return waveform


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        raise ValueError("zero-norm embedding")
    return float(np.dot(a, b) / denominator)


def score_ambiguous(rows: list[dict[str, Any]], device: str) -> dict[tuple[str, str], float]:
    import torch
    compatibility()
    repo = str((ROOT / "external/wesep-real-tse").resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import wesep  # noqa: PLC0415

    extractor = wesep.load_model_local(str((ROOT / "pretrained/wesep/spk_emb_100").resolve()))
    extractor.set_resample_rate(16000)
    extractor.set_vad(False)
    extractor.set_device(device)
    model = extractor.model.spk_ft.spkemb.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    for row in rows:
        for name, path in row["candidate_paths"].items():
            groups[int(sf.info(path).frames)].append((row["trial_id"], name, path))
    cache_dir = ANALYSIS / "ambiguous_candidate_embeddings"
    cache_dir.mkdir(exist_ok=True)
    embeddings: dict[tuple[str, str], np.ndarray] = {}
    for frame_count, tasks in sorted(groups.items()):
        for start in range(0, len(tasks), 64):
            batch = tasks[start:start + 64]
            tensor = torch.from_numpy(np.stack([load_mono16(path) for _, _, path in batch])).to(extractor.device)
            with torch.inference_mode():
                values = model.compute(tensor).detach().cpu().float().numpy().reshape(len(batch), -1)
            for (trial_id, name, path), embedding in zip(batch, values):
                if embedding.size != 192 or not np.isfinite(embedding).all():
                    raise ValueError(f"invalid ECAPA embedding: {trial_id} {name}")
                embeddings[(trial_id, name)] = embedding
                np.save(cache_dir / f"{safe_name(trial_id)}-{name}.npy", embedding)
    input_by_id = {row["trial_id"]: row for row in read_jsonl(ANALYSIS / "all_dev_candidate_inputs.jsonl")}
    scores = {}
    for row in rows:
        enrollment = np.load(input_by_id[row["trial_id"]]["speaker_embedding_path"], allow_pickle=False).reshape(-1)
        for name in ORDER:
            scores[(row["trial_id"], name)] = cosine(embeddings[(row["trial_id"], name)], enrollment)
    return scores


def assemble(args: argparse.Namespace) -> int:
    inputs = read_jsonl(ANALYSIS / "all_dev_candidate_inputs.jsonl")
    old = {row["trial_id"]: row for row in read_jsonl(CANDIDATE_GATE / "per_trial_candidates.jsonl")}
    multi = {row["trial_id"]: row for row in read_jsonl(args.ambiguous_multiview_index)}
    tfmap = {row["trial_id"]: row for row in read_jsonl(args.ambiguous_tfmap_index)}
    ambiguous_ids = {row["trial_id"] for row in inputs if row["cohort"] == "ambiguous_primary_wrong"}
    if set(multi) != ambiguous_ids or set(tfmap) != ambiguous_ids:
        raise ValueError("ambiguous candidate indexes do not exactly cover the frozen 9 trials")
    ambiguous_rows = []
    for trial_id in sorted(ambiguous_ids):
        paths = dict(multi[trial_id]["candidate_paths"])
        paths.update(tfmap[trial_id]["candidate_paths"])
        if tuple(paths) != ORDER or any(not Path(paths[name]).is_file() for name in ORDER):
            raise ValueError(f"invalid ambiguous candidate paths: {trial_id}")
        ambiguous_rows.append({"trial_id": trial_id, "candidate_paths": paths})
    ambiguous_scores = score_ambiguous(ambiguous_rows, args.device)
    ambiguous_paths = {row["trial_id"]: row["candidate_paths"] for row in ambiguous_rows}

    output = []
    for row in inputs:
        trial_id = row["trial_id"]
        candidates = {}
        if trial_id in old:
            for name in ORDER:
                candidate = old[trial_id]["candidates"][name]
                candidates[name] = {
                    "waveform_path": str(candidate["output_wav"]),
                    "enrollment_cosine": float(candidate["speaker_similarity_to_enrollment"]),
                }
        else:
            for name in ORDER:
                candidates[name] = {
                    "waveform_path": str(ambiguous_paths[trial_id][name]),
                    "enrollment_cosine": float(ambiguous_scores[(trial_id, name)]),
                }
        selected = {}
        for pool, names in POOLS.items():
            selected[pool] = max(
                names,
                key=lambda name: (candidates[name]["enrollment_cosine"], -ORDER.index(name)),
            )
        output.append({
            key: row[key]
            for key in ("trial_id", "split", "cohort", "mixture_wav", "enrollment_wav", "speaker_embedding_path")
        } | {"candidates": candidates, "selected": selected})
    if len(output) != 6000 or len({row["trial_id"] for row in output}) != 6000:
        raise ValueError("assembled selection is not exactly 6,000 unique trials")
    write_jsonl(ANALYSIS / "full_dev_candidates.jsonl", output)
    smoke_ids = json.loads((ANALYSIS / "preregistered_smoke_subset.json").read_text())["ordered_trial_ids"]
    by_id = {row["trial_id"]: row for row in output}
    write_jsonl(ANALYSIS / "smoke_candidates.jsonl", [by_id[trial_id] for trial_id in smoke_ids])
    summary = {
        "status": "COMPLETE",
        "trials": 6000,
        "candidate_order": ORDER,
        "selection_counts": {
            pool: dict(Counter(row["selected"][pool] for row in output)) for pool in POOLS
        },
        "selector_inputs": ["candidate waveform", "same complete target enrollment"],
        "clean_reference_used": False,
        "test_used": False,
    }
    (ANALYSIS / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def tokenize(args: argparse.Namespace) -> int:
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from se_align.codec.cosyvoice3_codec import CosyVoice3S3Tokenizer  # noqa: PLC0415

    rows = read_jsonl(ANALYSIS / "full_dev_candidates.jsonl")
    prepared = {row["trial_id"]: row for row in read_jsonl(PREPARED)}
    if len(rows) != 6000 or set(prepared) != {row["trial_id"] for row in rows}:
        raise ValueError("selection/prepared coverage mismatch")
    tokenizer = CosyVoice3S3Tokenizer(str(args.cosyvoice_model), provider=args.provider)
    records = []
    inference = []
    evaluation = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        paths = {}
        candidate_names = {}
        for pool in ("cdcs2", "cdcs5"):
            name = row["selected"][pool]
            source = row["candidates"][name]["waveform_path"]
            mixture_samples = int(sf.info(row["mixture_wav"]).frames)
            waveform = load_mono16(source)
            source_samples = int(waveform.size)
            if waveform.size >= mixture_samples:
                aligned = waveform[:mixture_samples]
                waveform_action = "none" if waveform.size == mixture_samples else "tail_truncate"
            else:
                aligned = np.pad(waveform, (0, mixture_samples - waveform.size))
                waveform_action = "tail_zero_pad"
            stem = safe_name(row["trial_id"])
            wav_path = ANALYSIS / "evidence" / pool / "wav" / f"{stem}.wav"
            token_path = ANALYSIS / "evidence" / pool / "tokens" / f"{stem}.npy"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav_path, aligned, 16000, subtype="PCM_16")
            encoded = tokenizer.encode(torch.from_numpy(aligned).unsqueeze(0), 16000).numpy().reshape(-1)
            expected = math.ceil(mixture_samples / 640)
            raw_length = int(encoded.size)
            difference = raw_length - expected
            if difference == 1:
                encoded = encoded[:expected]
                token_action = "tail_truncate_one"
            elif difference == -1:
                encoded = np.concatenate([encoded, encoded[-1:]])
                token_action = "repeat_final_token_once"
            elif difference == 0:
                token_action = "none"
            else:
                raise ValueError(
                    f"fatal evidence/mixture token length difference {difference}: {row['trial_id']} {pool}"
                )
            if encoded.size != expected or encoded.min() < 0 or encoded.max() >= 6561:
                raise ValueError(f"invalid aligned tokens: {row['trial_id']} {pool}")
            np.save(token_path, encoded.astype(np.int32, copy=False))
            paths[pool] = str(token_path)
            candidate_names[pool] = name
            records.append({
                "trial_id": row["trial_id"],
                "split": "dev",
                "pool": pool,
                "selected_candidate": name,
                "source_waveform": source,
                "aligned_waveform": str(wav_path),
                "mixture_num_samples": mixture_samples,
                "source_num_samples": source_samples,
                "waveform_alignment_action": waveform_action,
                "raw_evidence_num_tokens": raw_length,
                "expected_num_tokens": expected,
                "raw_minus_expected_tokens": difference,
                "token_adjustment_action": token_action,
                "evidence_token_path": str(token_path),
            })
        inference.append({
            "trial_id": row["trial_id"],
            "split": "dev",
            "mixture_wav": row["mixture_wav"],
            "enrollment_wav": row["enrollment_wav"],
            "speaker_embedding_path": row["speaker_embedding_path"],
            "cdcs2_evidence_token_path": paths["cdcs2"],
            "cdcs5_evidence_token_path": paths["cdcs5"],
            "cdcs2_direct_candidate": candidate_names["cdcs2"],
            "cdcs5_direct_candidate": candidate_names["cdcs5"],
        })
        source_eval = prepared[row["trial_id"]]
        evaluation.append(dict(source_eval) | {
            "cohort": row["cohort"],
            "cdcs2_direct_candidate": candidate_names["cdcs2"],
            "cdcs5_direct_candidate": candidate_names["cdcs5"],
            "cdcs2_direct_waveform": next(
                item["aligned_waveform"] for item in records[-2:] if item["pool"] == "cdcs2"
            ),
            "cdcs5_direct_waveform": next(
                item["aligned_waveform"] for item in records[-2:] if item["pool"] == "cdcs5"
            ),
        })
        if index == 1 or index % 200 == 0 or index == len(rows):
            print(f"tokenized={index}/{len(rows)} rate={index / max(time.monotonic() - started, 1e-6):.2f}/s", flush=True)

    forbidden = (
        "target", "interferer", "transcript", "sisdr", "si_sdr", "qc", "reference", "label"
    )
    for index, row in enumerate(inference):
        leaked = sorted(key for key in row if any(value in key.lower() for value in forbidden))
        if leaked:
            raise ValueError(f"fail-closed inference manifest row {index} leaked fields: {leaked}")
    write_jsonl(ANALYSIS / "evidence_alignment.jsonl", records)
    write_jsonl(ROOT / "manifests/selected_evidence_dev_inference.jsonl", inference)
    write_jsonl(ROOT / "manifests/selected_evidence_dev_evaluation.jsonl", evaluation)
    smoke_ids = json.loads((ANALYSIS / "preregistered_smoke_subset.json").read_text())["ordered_trial_ids"]
    inference_by_id = {row["trial_id"]: row for row in inference}
    evaluation_by_id = {row["trial_id"]: row for row in evaluation}
    write_jsonl(
        ROOT / "manifests/selected_evidence_dev_smoke_inference.jsonl",
        [inference_by_id[trial_id] for trial_id in smoke_ids],
    )
    write_jsonl(
        ROOT / "manifests/selected_evidence_dev_smoke_evaluation.jsonl",
        [evaluation_by_id[trial_id] for trial_id in smoke_ids],
    )
    summary = {
        "status": "COMPLETE",
        "trials": 6000,
        "evidence_rows": len(records),
        "provider": tokenizer.provider,
        "waveform_actions": dict(Counter(row["waveform_alignment_action"] for row in records)),
        "token_length_differences": dict(Counter(str(row["raw_minus_expected_tokens"]) for row in records)),
        "token_actions": dict(Counter(row["token_adjustment_action"] for row in records)),
        "inference_fields": sorted(inference[0]),
        "target_length_used": False,
        "clean_reference_used": False,
        "test_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (ANALYSIS / "evidence_preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "assemble":
        return assemble(args)
    return tokenize(args)


if __name__ == "__main__":
    raise SystemExit(main())
