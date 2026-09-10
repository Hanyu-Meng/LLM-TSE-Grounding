#!/usr/bin/env python3
"""Build target-speaker trials from Libri2Mix and SpeakerBeam enrollment maps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librimix-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train-100", "train-360", "dev", "test"), required=True)
    parser.add_argument(
        "--enrollment-map",
        type=Path,
        help="SpeakerBeam fixed map (preferred for dev/test)",
    )
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        help="choose a deterministic same-speaker/different-utterance enrollment",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-mixtures",
        type=int,
        default=0,
        help="number of mixtures to emit; zero means all mapped mixtures",
    )
    return parser.parse_args()


def read_mapping(path: Path) -> dict[str, list[tuple[str, str]]]:
    mapping: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 fields, got {len(fields)}")
            mixture_id, target_utt, enrollment_relpath = fields
            mapping[mixture_id].append((target_utt, enrollment_relpath))
    return mapping


def require_wav(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty WAV: {path}")
    return path.resolve()


def enrollment_utterance(enrollment_path: Path, source_index: int) -> str:
    utterances = enrollment_path.stem.split("_")
    if len(utterances) != 2:
        raise ValueError(f"unexpected Libri2Mix enrollment name: {enrollment_path.name}")
    return utterances[source_index]


def trial_base(split_dir: Path, split: str, mixture_id: str,
               target_utt: str) -> tuple[dict, int]:
    source_utts = mixture_id.split("_")
    if len(source_utts) != 2:
        raise ValueError(f"unexpected Libri2Mix mixture id: {mixture_id}")
    if target_utt not in source_utts:
        raise ValueError(f"target {target_utt} is not a source of {mixture_id}")
    target_index = source_utts.index(target_utt)
    interferer_index = 1 - target_index
    target_speaker = target_utt.split("-", 1)[0]
    interferer_utt = source_utts[interferer_index]
    interferer_speaker = interferer_utt.split("-", 1)[0]
    if target_speaker == interferer_speaker:
        raise ValueError(f"same-speaker mixture is not a valid TSE trial: {mixture_id}")
    row = {
        "trial_id": f"{split}:{mixture_id}:{target_speaker}",
        "split": split,
        "mixture_wav": str(require_wav(split_dir / "mix_clean" / f"{mixture_id}.wav")),
        "target_wav": str(require_wav(split_dir / f"s{target_index + 1}" / f"{mixture_id}.wav")),
        "interferer_wavs": [
            str(require_wav(split_dir / f"s{interferer_index + 1}" / f"{mixture_id}.wav"))
        ],
        "target_speaker": target_speaker,
        "interferer_speakers": [interferer_speaker],
        "target_utterance": target_utt,
    }
    return row, target_index


def build_trial(split_dir: Path, split: str, mixture_id: str,
                target_utt: str, enrollment_relpath: str,
                librispeech_root: Path | None = None,
                cache: dict[str, list[Path]] | None = None) -> dict:
    row, target_index = trial_base(split_dir, split, mixture_id, target_utt)
    target_speaker = row["target_speaker"]

    relative = Path(enrollment_relpath)
    if relative.suffix != ".wav":
        relative = relative.with_suffix(".wav")
    if relative.parts[0] not in ("s1", "s2"):
        raise ValueError(f"enrollment path must begin with s1/ or s2/: {relative}")
    enrollment_index = int(relative.parts[0][1:]) - 1
    enrollment_path = require_wav(split_dir / relative)
    enrollment_utt = enrollment_utterance(enrollment_path, enrollment_index)
    enrollment_speaker = enrollment_utt.split("-", 1)[0]
    if enrollment_speaker != target_speaker:
        raise ValueError(
            f"speaker mismatch for {mixture_id}: target={target_speaker}, "
            f"enrollment={enrollment_speaker} ({enrollment_path})"
        )
    if enrollment_utt == target_utt:
        if librispeech_root is not None and cache is not None:
            fallback = build_librispeech_trial(
                split_dir, split, mixture_id, target_utt, librispeech_root, cache
            )
            fallback["enrollment_source"] = "librispeech_leakage_fallback"
            return fallback
        raise ValueError(f"enrollment leakage for {mixture_id}: {target_utt}")

    row["enrollment_wav"] = str(enrollment_path)
    row["enrollment_utterance"] = enrollment_utt
    row["enrollment_source"] = "speakerbeam_fixed_map"
    return row


def librispeech_split_name(split: str) -> str:
    return {
        "train-100": "train-clean-100",
        "train-360": "train-clean-360",
        "dev": "dev-clean",
        "test": "test-clean",
    }[split]


def build_librispeech_trial(split_dir: Path, split: str, mixture_id: str,
                            target_utt: str, librispeech_root: Path,
                            cache: dict[str, list[Path]]) -> dict:
    row, _ = trial_base(split_dir, split, mixture_id, target_utt)
    speaker = row["target_speaker"]
    if speaker not in cache:
        speaker_dir = librispeech_root / librispeech_split_name(split) / speaker
        cache[speaker] = sorted(speaker_dir.rglob("*.flac"))
    candidates = [path for path in cache[speaker] if path.stem != target_utt]
    if not candidates:
        raise ValueError(f"no different enrollment utterance for speaker {speaker}")
    enrollment_path = require_wav(candidates[0])
    row["enrollment_wav"] = str(enrollment_path)
    row["enrollment_utterance"] = enrollment_path.stem
    row["enrollment_source"] = "librispeech_same_speaker"
    return row


def main() -> int:
    args = parse_args()
    split_dir = args.librimix_root.resolve() / "wav16k" / "min" / args.split
    if not args.enrollment_map and not args.librispeech_root:
        raise ValueError("provide --enrollment-map, --librispeech-root, or both")
    trials: list[dict] = []
    mixtures_written = 0

    if args.enrollment_map:
        mapping = read_mapping(args.enrollment_map.resolve())
        cache: dict[str, list[Path]] = {}
        fallback_root = args.librispeech_root.resolve() if args.librispeech_root else None
        for mixture_id, targets in mapping.items():
            if args.num_mixtures and mixtures_written >= args.num_mixtures:
                break
            if len(targets) != 2:
                raise ValueError(f"expected two target rows for {mixture_id}, found {len(targets)}")
            trials.extend([
                build_trial(
                    split_dir,
                    args.split,
                    mixture_id,
                    target_utt,
                    enrollment_relpath,
                    fallback_root,
                    cache,
                )
                for target_utt, enrollment_relpath in targets
            ])
            mixtures_written += 1
    else:
        cache: dict[str, list[Path]] = {}
        mixture_paths = sorted((split_dir / "mix_clean").glob("*.wav"))
        if args.num_mixtures:
            mixture_paths = mixture_paths[:args.num_mixtures]
        for mixture_path in mixture_paths:
            mixture_id = mixture_path.stem
            targets = mixture_id.split("_")
            if len(targets) != 2:
                raise ValueError(f"unexpected Libri2Mix mixture id: {mixture_id}")
            trials.extend([
                build_librispeech_trial(
                    split_dir,
                    args.split,
                    mixture_id,
                    target_utt,
                    args.librispeech_root.resolve(),
                    cache,
                )
                for target_utt in targets
            ])
            mixtures_written += 1

    if not trials:
        raise RuntimeError("no trials were generated")
    expected = mixtures_written * 2
    if len(trials) != expected:
        raise RuntimeError(f"expected {expected} trials, generated {len(trials)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(json.dumps(trial, ensure_ascii=True) + "\n")
    print(f"Wrote {len(trials)} trials from {mixtures_written} mixtures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
