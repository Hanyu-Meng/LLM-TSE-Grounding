#!/usr/bin/env python3
"""Embed frozen candidate outputs with the frozen WeSep ECAPA speaker encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wesep-repo", type=Path, default=PROJECT_ROOT / "external/wesep-real-tse"
    )
    parser.add_argument(
        "--wesep-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "pretrained/wesep/spk_emb_100",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--expected-trials", type=int, default=5991)
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
        raise ValueError(f"expected 16 kHz audio: {path}")
    return values.mean(axis=1, dtype=np.float64).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        raise ValueError("zero-norm speaker embedding")
    return float(np.dot(a, b) / denominator)


def embedding_path(directory: Path, audio_path: str) -> Path:
    digest = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()
    return directory / f"{digest}.npy"


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.acoustic_metrics)
    if len(rows) != args.expected_trials or len({row["trial_id"] for row in rows}) != args.expected_trials:
        raise ValueError(
            f"acoustic metrics must contain {args.expected_trials:,} unique DEV trials"
        )
    if any(row.get("split") != args.split for row in rows):
        raise ValueError(f"input is not exclusively split={args.split}")

    import torch

    compatibility()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    repo = str(args.wesep_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import wesep

    extractor = wesep.load_model_local(str(args.wesep_checkpoint.resolve()))
    extractor.set_resample_rate(16000)
    extractor.set_vad(False)
    extractor.set_device(args.device)
    speaker_model = extractor.model.spk_ft.spkemb.eval()
    for parameter in speaker_model.parameters():
        parameter.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = args.output_dir / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    tasks: list[tuple[str, str, str]] = []
    for row in rows:
        for candidate, metrics in row["candidates"].items():
            tasks.append((row["trial_id"], candidate, metrics["output_wav"]))

    groups: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    for task in tasks:
        info = sf.info(task[2])
        if int(info.samplerate) != 16000:
            raise ValueError(f"expected 16 kHz audio: {task[2]}")
        groups[int(info.frames)].append(task)

    completed = 0
    started = time.monotonic()
    for frame_count in sorted(groups):
        group = groups[frame_count]
        for start in range(0, len(group), args.batch_size):
            batch = group[start : start + args.batch_size]
            missing = [item for item in batch if not embedding_path(embedding_dir, item[2]).is_file()]
            if missing:
                waveforms = np.stack([load_mono16(item[2]) for item in missing])
                if waveforms.shape[1] != frame_count:
                    raise ValueError("audio length changed after metadata scan")
                tensor = torch.from_numpy(waveforms).to(extractor.device)
                with torch.inference_mode():
                    values = speaker_model.compute(tensor).detach().cpu().float().numpy()
                values = values.reshape(len(missing), -1)
                for item, embedding in zip(missing, values):
                    if embedding.size != 192 or not np.isfinite(embedding).all():
                        raise ValueError(f"invalid candidate embedding: {item[0]} {item[1]}")
                    np.save(embedding_path(embedding_dir, item[2]), embedding.astype(np.float32, copy=False))
            completed += len(batch)
            if completed == len(tasks) or completed // args.progress_interval != (completed - len(batch)) // args.progress_interval:
                print(
                    f"embedded={completed}/{len(tasks)} rate={completed / max(time.monotonic() - started, 1e-6):.2f}/s",
                    flush=True,
                )

    records = []
    for row in rows:
        target_enrollment = np.load(row["target_enrollment_embedding_path"], allow_pickle=False).reshape(-1)
        interferer_enrollment = np.load(
            row["interferer_enrollment_embedding_path"], allow_pickle=False
        ).reshape(-1)
        for candidate, metrics in row["candidates"].items():
            path = embedding_path(embedding_dir, metrics["output_wav"])
            candidate_embedding = np.load(path, allow_pickle=False).reshape(-1)
            target_similarity = cosine(candidate_embedding, target_enrollment)
            interferer_similarity = cosine(candidate_embedding, interferer_enrollment)
            if not all(math.isfinite(value) for value in (target_similarity, interferer_similarity)):
                raise ValueError(f"non-finite similarity: {row['trial_id']} {candidate}")
            records.append(
                {
                    "trial_id": row["trial_id"],
                    "split": args.split,
                    "cohort": row["cohort"],
                    "candidate": candidate,
                    "output_wav": metrics["output_wav"],
                    "candidate_embedding_path": str(path.resolve()),
                    "speaker_similarity_to_enrollment": target_similarity,
                    "speaker_similarity_to_interferer_enrollment": interferer_similarity,
                    "speaker_embedding_margin": target_similarity - interferer_similarity,
                }
            )
    index_path = args.output_dir / "candidate_speaker_metrics.jsonl"
    index_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    summary = {
        "status": "COMPLETE",
        "split": args.split,
        "trials": len(rows),
        "candidate_outputs": len(records),
        "embedding_dim": 192,
        "speaker_encoder": "frozen WeSep ECAPA-TDNN from spk_emb_100",
        "seed": args.seed,
        "speaker_margin_definition": "cos(candidate,target full enrollment)-cos(candidate,paired interferer full enrollment)",
        "elapsed_seconds": time.monotonic() - started,
        "test_used": args.split == "test",
    }
    (args.output_dir / "speaker_embedding_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
