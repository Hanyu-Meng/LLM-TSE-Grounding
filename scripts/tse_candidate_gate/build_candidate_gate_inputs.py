#!/usr/bin/env python3
"""Build locked natural-DEV cohorts and deterministic enrollment views."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("first", "middle", "final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-manifest",
        type=Path,
        default=ROOT / "manifests/tse_dev_prepared.jsonl",
    )
    parser.add_argument(
        "--natural-audit",
        type=Path,
        default=ROOT / "analysis/wesep_speaker_selection/dev_all_trials.jsonl",
    )
    parser.add_argument(
        "--primary-metrics",
        type=Path,
        default=ROOT / "dev_outputs/WeSep/per_trial_metrics.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "analysis/candidate_gate"
    )
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--start-grid-seconds", type=float, default=0.01)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_key(path: str) -> str:
    stem = Path(path).stem
    readable = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    digest = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]
    return f"{readable[:72]}-{digest}"


def load_mono16(path: str) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if int(sample_rate) != 16000:
        raise ValueError(f"enrollment must be 16 kHz: {path}")
    waveform = values.mean(axis=1, dtype=np.float64).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid enrollment waveform: {path}")
    return waveform


def choose_starts(
    waveform: np.ndarray, window: int, step: int
) -> tuple[dict[str, int], bool]:
    if waveform.size < window:
        return {view: 0 for view in VIEWS}, True
    maximum = waveform.size - window
    starts = list(range(0, maximum + 1, step))
    if starts[-1] != maximum:
        starts.append(maximum)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(waveform.astype(np.float64) ** 2)]
    )

    def energy(start: int) -> float:
        return float((cumulative[start + window] - cumulative[start]) / window)

    groups: dict[str, list[int]] = {view: [] for view in VIEWS}
    for start in starts:
        fraction = start / maximum if maximum else 0.5
        if fraction < 1.0 / 3.0:
            groups["first"].append(start)
        elif fraction < 2.0 / 3.0:
            groups["middle"].append(start)
        else:
            groups["final"].append(start)
    targets = {"first": 0.0, "middle": 0.5, "final": 1.0}
    result = {}
    for view in VIEWS:
        candidates = groups[view]
        if not candidates:
            candidates = [
                min(starts, key=lambda start: abs(start / max(1, maximum) - targets[view]))
            ]
        result[view] = max(candidates, key=lambda start: (energy(start), -start))
    return result, False


def materialize_views(
    enrollment_path: str, root: Path, segment_seconds: float, grid_seconds: float
) -> dict[str, Any]:
    waveform = load_mono16(enrollment_path)
    window = int(round(segment_seconds * 16000))
    step = int(round(grid_seconds * 16000))
    if window <= 0 or step <= 0:
        raise ValueError("segment duration and grid must be positive")
    starts, short = choose_starts(waveform, window, step)
    directory = root / safe_key(enrollment_path)
    directory.mkdir(parents=True, exist_ok=True)
    view_paths = {"full": str(Path(enrollment_path).resolve())}
    for view in VIEWS:
        start = starts[view]
        segment = waveform[start : start + window]
        if segment.size < window:
            segment = np.pad(segment, (0, window - segment.size))
        destination = directory / f"{view}.wav"
        if not destination.is_file():
            sf.write(destination, segment, 16000, subtype="PCM_16")
        view_paths[view] = str(destination.resolve())
    return {
        "enrollment_path": str(Path(enrollment_path).resolve()),
        "duration_seconds": waveform.size / 16000.0,
        "segment_seconds": segment_seconds,
        "short_utterance_fallback": short,
        "start_samples": starts,
        "start_seconds": {key: value / 16000.0 for key, value in starts.items()},
        "view_paths": view_paths,
    }


def main() -> int:
    args = parse_args()
    prepared_rows = read_jsonl(args.prepared_manifest)
    audit_rows = read_jsonl(args.natural_audit)
    primary_rows = read_jsonl(args.primary_metrics)
    prepared = {row["trial_id"]: row for row in prepared_rows}
    audit = {row["trial_id"]: row for row in audit_rows}
    primary = {row["trial_id"]: row for row in primary_rows}
    if not (
        len(prepared_rows)
        == len(prepared)
        == len(audit_rows)
        == len(audit)
        == len(primary_rows)
        == len(primary)
        == 6000
    ):
        raise ValueError("expected three aligned sources with 6,000 unique DEV trials")
    if set(prepared) != set(audit) or set(prepared) != set(primary):
        raise ValueError("DEV sources have mismatched trial IDs")
    by_mixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared_rows:
        by_mixture[Path(row["mixture_wav"]).stem].append(row)
    if any(len(rows) != 2 for rows in by_mixture.values()):
        raise ValueError("expected exactly two target trials per DEV mixture")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    view_root = args.output_dir / "enrollment_views"
    selected_ids = [
        row["trial_id"]
        for row in audit_rows
        if row["high_confidence_wrong"] or not row["wrong_margin_0"]
    ]
    enrollment_paths = sorted({prepared[trial_id]["enrollment_wav"] for trial_id in selected_ids})
    view_index: dict[str, dict[str, Any]] = {}
    for index, enrollment_path in enumerate(enrollment_paths, 1):
        view_index[enrollment_path] = materialize_views(
            enrollment_path,
            view_root,
            args.segment_seconds,
            args.start_grid_seconds,
        )
        if index == 1 or index % 200 == 0 or index == len(enrollment_paths):
            print(f"enrollment_views={index}/{len(enrollment_paths)}", flush=True)

    output = []
    for trial_id in selected_ids:
        row = prepared[trial_id]
        natural = audit[trial_id]
        mixture_id = Path(row["mixture_wav"]).stem
        counterpart = next(
            item for item in by_mixture[mixture_id] if item["trial_id"] != trial_id
        )
        if str(counterpart["target_speaker"]) != str(row["interferer_speakers"][0]):
            raise ValueError(f"paired interferer mismatch: {trial_id}")
        primary_row = primary[trial_id]
        if primary_row.get("decode_status") != "ok":
            raise ValueError(f"failed saved primary: {trial_id}")
        output.append(
            {
                "trial_id": trial_id,
                "split": "dev",
                "cohort": (
                    "natural_primary_swap"
                    if natural["high_confidence_wrong"]
                    else "primary_correct_control"
                ),
                "mixture_id": mixture_id,
                "mixture_wav": row["mixture_wav"],
                "target_wav": row["target_wav"],
                "interferer_wav": row["interferer_wavs"][0],
                "target_speaker": row["target_speaker"],
                "interferer_speaker": row["interferer_speakers"][0],
                "enrollment_wav": row["enrollment_wav"],
                "target_enrollment_embedding_path": row["speaker_embedding_path"],
                "interferer_enrollment_embedding_path": counterpart[
                    "speaker_embedding_path"
                ],
                "enrollment_views": view_index[row["enrollment_wav"]]["view_paths"],
                "enrollment_view_start_seconds": view_index[row["enrollment_wav"]][
                    "start_seconds"
                ],
                "enrollment_short_fallback": view_index[row["enrollment_wav"]][
                    "short_utterance_fallback"
                ],
                "primary_output_wav": primary_row["output_wav"],
                "frozen_primary_high_confidence_wrong": bool(
                    natural["high_confidence_wrong"]
                ),
                "frozen_primary_margin_db": float(natural["speaker_selection_margin"]),
            }
        )
    counts = Counter(row["cohort"] for row in output)
    if counts != Counter(
        {"natural_primary_swap": 405, "primary_correct_control": 5586}
    ):
        raise ValueError(f"unexpected locked cohort counts: {counts}")
    manifest_path = args.output_dir / "candidate_input_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in output), encoding="utf-8"
    )
    index_path = args.output_dir / "enrollment_view_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(view_index[path]) + "\n" for path in enrollment_paths),
        encoding="utf-8",
    )
    summary = {
        "status": "COMPLETE",
        "split": "dev",
        "trials": len(output),
        "cohorts": dict(counts),
        "excluded_ambiguous_primary_wrong": 9,
        "unique_enrollment_utterances": len(enrollment_paths),
        "short_enrollment_fallbacks": sum(
            row["short_utterance_fallback"] for row in view_index.values()
        ),
        "segment_seconds": args.segment_seconds,
        "start_grid_seconds": args.start_grid_seconds,
        "test_used": False,
        "clean_references_used_for_candidate_construction": False,
    }
    (args.output_dir / "candidate_input_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
