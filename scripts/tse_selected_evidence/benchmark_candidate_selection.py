#!/usr/bin/env python3
"""Benchmark the frozen ECAPA enrollment-cosine selector on the locked smoke set."""

from __future__ import annotations

import json
import resource
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/selected_evidence"
ORDER = ("full", "first", "middle", "final", "tfmap_context_full")
POOLS = {
    "Pool A": ("full",),
    "Pool B": ("full", "tfmap_context_full"),
    "Pool C": ("full", "first", "middle", "final"),
    "Pool D": ORDER,
}


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
        raise ValueError(path)
    return values.mean(axis=1, dtype=np.float64).astype(np.float32)


def main() -> int:
    import torch
    compatibility()
    repo = str((ROOT / "external/wesep-real-tse").resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import wesep

    inputs = read_jsonl(ANALYSIS / "smoke_candidate_inputs.jsonl")
    if len(inputs) != 100:
        raise ValueError("cost benchmark requires the locked 100-trial smoke set")
    components = {}
    for key, path in {
        "full": ANALYSIS / "cost_benchmark/full/candidate_paths.jsonl",
        "segments": ANALYSIS / "cost_benchmark/segments/candidate_paths.jsonl",
        "tfmap": ANALYSIS / "cost_benchmark/tfmap/candidate_paths.jsonl",
    }.items():
        components[key] = {row["trial_id"]: row for row in read_jsonl(path)}
        if set(components[key]) != {row["trial_id"] for row in inputs}:
            raise ValueError(f"cost component coverage mismatch: {key}")
    paths = {}
    for row in inputs:
        trial_id = row["trial_id"]
        candidate_paths = dict(components["full"][trial_id]["candidate_paths"])
        candidate_paths.update(components["segments"][trial_id]["candidate_paths"])
        candidate_paths.update(components["tfmap"][trial_id]["candidate_paths"])
        if tuple(candidate_paths) != ORDER:
            raise ValueError(f"candidate order mismatch: {trial_id}")
        paths[trial_id] = candidate_paths

    extractor = wesep.load_model_local(str((ROOT / "pretrained/wesep/spk_emb_100").resolve()))
    extractor.set_resample_rate(16000)
    extractor.set_vad(False)
    extractor.set_device("cuda")
    model = extractor.model.spk_ft.spkemb.eval()
    model.requires_grad_(False)
    total_audio_seconds = sum(sf.info(row["mixture_wav"]).frames / 16000 for row in inputs)
    output = {}
    for pool_name, names in POOLS.items():
        groups: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
        for row in inputs:
            for name in names:
                path = paths[row["trial_id"]][name]
                groups[int(sf.info(path).frames)].append((row["trial_id"], name, path))
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        cpu_started = time.process_time()
        batch_calls = 0
        embeddings = {}
        for frame_count, tasks in sorted(groups.items()):
            for start in range(0, len(tasks), 64):
                batch = tasks[start:start + 64]
                waveforms = np.stack([load_mono16(path) for _, _, path in batch])
                if waveforms.shape[1] != frame_count:
                    raise ValueError("candidate length changed")
                with torch.inference_mode():
                    values = model.compute(
                        torch.from_numpy(waveforms).to(extractor.device)
                    ).detach().cpu().float().numpy().reshape(len(batch), -1)
                batch_calls += 1
                for (trial_id, name, _path), embedding in zip(batch, values):
                    embeddings[(trial_id, name)] = embedding
        selected = []
        for row in inputs:
            enrollment = np.load(row["speaker_embedding_path"], allow_pickle=False).reshape(-1)
            scores = {}
            for name in names:
                candidate = embeddings[(row["trial_id"], name)]
                scores[name] = float(
                    np.dot(candidate, enrollment)
                    / (np.linalg.norm(candidate) * np.linalg.norm(enrollment))
                )
            selected.append(max(names, key=lambda name: (scores[name], -ORDER.index(name))))
        elapsed = time.monotonic() - started
        cpu_seconds = time.process_time() - cpu_started
        output[pool_name] = {
            "candidate_names": names,
            "trials": len(inputs),
            "candidate_embeddings": len(embeddings),
            "embedding_batch_calls": batch_calls,
            "selected_candidate_counts": {
                name: selected.count(name) for name in names
            },
            "elapsed_seconds": elapsed,
            "latency_seconds_per_trial": elapsed / len(inputs),
            "candidate_selection_rtf": elapsed / total_audio_seconds,
            "cpu_process_seconds": cpu_seconds,
            "average_process_cpu_percent": 100 * cpu_seconds / max(elapsed, 1e-9),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "peak_torch_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_torch_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        print(json.dumps({pool_name: output[pool_name]}, indent=2), flush=True)
    summary = {
        "status": "COMPLETE",
        "scope": "locked 100-trial DEV smoke subset",
        "model_load_excluded": True,
        "speaker_encoder": "frozen WeSep ECAPA-TDNN from spk_emb_100",
        "total_mixture_audio_seconds": total_audio_seconds,
        "pools": output,
        "test_used": False,
    }
    destination = ANALYSIS / "cost_benchmark/selection_cost.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
