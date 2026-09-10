#!/usr/bin/env python3
"""Materialize WeSep evidence, speaker embeddings, and CosyVoice3 tokens."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from se_align.data.store import read_manifest, write_manifest
from se_align.utils.audio import load_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("wesep", "tokens", "finalize"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--wesep-repo", type=Path)
    parser.add_argument("--wesep-checkpoint", type=Path)
    parser.add_argument("--cosyvoice-model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def trial_key(trial_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", trial_id).strip("_")[:96]
    digest = hashlib.sha1(trial_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def paths_for(root: Path, row: dict) -> dict[str, Path]:
    key = trial_key(row["trial_id"])
    return {
        "evidence_wav": root / "evidence_wav" / f"{key}.wav",
        "speaker_embedding": root / "speaker_embeddings" / f"{key}.npy",
        "evidence_token": root / "evidence_tokens" / f"{key}.npy",
        "target_token": root / "target_tokens" / f"{key}.npy",
    }


def selected_rows(args: argparse.Namespace) -> list[dict]:
    rows = read_manifest(args.manifest)
    return rows[:args.limit] if args.limit else rows


def run_wesep(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.wesep_checkpoint is None or args.wesep_repo is None:
        raise ValueError("--wesep-repo and --wesep-checkpoint are required")
    wesep_repo = str(args.wesep_repo.resolve())
    if wesep_repo not in sys.path:
        sys.path.insert(0, wesep_repo)
    import soundfile as sf
    import torchaudio

    # s3prl 0.4 still calls this legacy no-op during import. Torchaudio 2.10
    # removed the symbol, so provide it without changing the vendored package.
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *_args, **_kwargs: None
    try:
        import torchcodec  # noqa: F401
        has_torchcodec = True
    except ImportError:
        has_torchcodec = False
    torchaudio_version = tuple(
        int(part) for part in torchaudio.__version__.split("+", 1)[0].split(".")[:2]
    )
    if torchaudio_version >= (2, 10) and not has_torchcodec:
        def soundfile_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                           channels_first=True, **_kwargs):
            dtype = "float32" if normalize else "int16"
            with sf.SoundFile(uri) as audio:
                audio.seek(int(frame_offset))
                frames = -1 if int(num_frames) < 0 else int(num_frames)
                values = audio.read(frames=frames, dtype=dtype, always_2d=True)
                sample_rate = int(audio.samplerate)
            waveform = torch.from_numpy(values.copy())
            if channels_first:
                waveform = waveform.transpose(0, 1).contiguous()
            return waveform, sample_rate

        torchaudio.load = soundfile_load
    if "torchaudio.sox_effects" not in sys.modules:
        import types

        sox_effects = types.ModuleType("torchaudio.sox_effects")

        def unavailable_sox_effect(*_args, **_kwargs):
            raise RuntimeError("torchaudio 2.10 removed the unused sox_effects API")

        sox_effects.apply_effects_file = unavailable_sox_effect
        sox_effects.apply_effects_tensor = unavailable_sox_effect
        sys.modules["torchaudio.sox_effects"] = sox_effects
    import wesep

    extractor = wesep.load_model_local(str(args.wesep_checkpoint))
    extractor.set_resample_rate(16000)
    extractor.set_vad(False)
    extractor.set_device(args.device)
    extractor.set_output_norm(True)
    for index, row in enumerate(rows, 1):
        paths = paths_for(args.output_root, row)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not paths["evidence_wav"].is_file():
            speech = extractor.extract_speech(row["mixture_wav"], row["enrollment_wav"])
            if speech is None or speech.numel() == 0 or not torch.isfinite(speech).all():
                raise RuntimeError(f"WeSep failed for {row['trial_id']}")
            sf.write(paths["evidence_wav"], speech[0].cpu().numpy(), 16000)
        if args.overwrite or not paths["speaker_embedding"].is_file():
            enrollment, sample_rate = torchaudio.load(row["enrollment_wav"])
            enrollment = enrollment.mean(dim=0, keepdim=True).float()
            if sample_rate != 16000:
                enrollment = torchaudio.functional.resample(enrollment, sample_rate, 16000)
            with torch.no_grad():
                embedding = extractor.model.spk_ft.spkemb.compute(
                    enrollment.to(extractor.device)
                ).reshape(-1)
            if embedding.numel() != 192 or not torch.isfinite(embedding).all():
                raise RuntimeError(f"invalid WeSep speaker embedding for {row['trial_id']}")
            np.save(paths["speaker_embedding"], embedding.cpu().float().numpy())
        print(f"[{index}/{len(rows)}] wesep {row['trial_id']}")


def run_tokens(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.cosyvoice_model is None:
        raise ValueError("--cosyvoice-model is required for the tokens stage")
    from se_align.codec.cosyvoice3_codec import CosyVoice3S3Tokenizer

    if args.device == "cpu":
        provider = "CPUExecutionProvider"
    elif args.device.startswith("cuda"):
        provider = "CUDAExecutionProvider"
    else:
        raise ValueError("tokens stage --device must be cpu or cuda")
    tokenizer = CosyVoice3S3Tokenizer(str(args.cosyvoice_model), provider=provider)
    print(f"CosyVoice3 tokenizer provider={tokenizer.provider}", flush=True)
    for index, row in enumerate(rows, 1):
        paths = paths_for(args.output_root, row)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        pairs = (
            (paths["evidence_wav"], paths["evidence_token"]),
            (Path(row["target_wav"]), paths["target_token"]),
        )
        for audio_path, token_path in pairs:
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if args.overwrite or not token_path.is_file():
                waveform, sample_rate = load_wav(str(audio_path))
                tokens = tokenizer.encode(waveform, sample_rate)
                np.save(token_path, tokens.numpy().astype(np.int32, copy=False))
        print(f"[{index}/{len(rows)}] tokens {row['trial_id']}")


def run_finalize(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.prepared_manifest is None:
        raise ValueError("--prepared-manifest is required for the finalize stage")
    prepared: list[dict] = []
    for row in rows:
        paths = paths_for(args.output_root, row)
        absent = [str(path) for path in paths.values() if not path.is_file()]
        if absent:
            raise FileNotFoundError(
                f"incomplete features for {row['trial_id']}: {', '.join(absent)}"
            )
        evidence = np.load(paths["evidence_token"], allow_pickle=False)
        target = np.load(paths["target_token"], allow_pickle=False)
        speaker = np.load(paths["speaker_embedding"], allow_pickle=False)
        if evidence.size == 0 or target.size == 0 or speaker.reshape(-1).size != 192:
            raise ValueError(f"invalid prepared features for {row['trial_id']}")
        item = dict(row)
        item.update({
            "evidence_wav": str(paths["evidence_wav"].resolve()),
            "evidence_token_path": str(paths["evidence_token"].resolve()),
            "target_token_path": str(paths["target_token"].resolve()),
            "speaker_embedding_path": str(paths["speaker_embedding"].resolve()),
            "evidence_num_tokens": int(evidence.size),
            "target_num_tokens": int(target.size),
        })
        prepared.append(item)
    write_manifest(args.prepared_manifest, prepared)
    print(f"Wrote {len(prepared)} prepared trials to {args.prepared_manifest}")


def main() -> int:
    args = parse_args()
    rows = selected_rows(args)
    if not rows:
        raise ValueError("manifest has no selected trials")
    args.output_root = args.output_root.resolve()
    if args.stage == "wesep":
        run_wesep(args, rows)
    elif args.stage == "tokens":
        run_tokens(args, rows)
    else:
        run_finalize(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
