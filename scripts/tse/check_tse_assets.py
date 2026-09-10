#!/usr/bin/env python3
"""Check local assets required by the TSE preprocessing and smoke tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Result:
    group: str
    name: str
    status: str
    detail: str


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def any_nonempty(paths: Iterable[Path]) -> bool:
    return any(nonempty(path) for path in paths)


def ready(group: str, name: str, detail: str) -> Result:
    return Result(group, name, "READY", detail)


def missing(group: str, name: str, detail: str) -> Result:
    return Result(group, name, "MISSING", detail)


def broken(group: str, name: str, detail: str) -> Result:
    return Result(group, name, "BROKEN", detail)


def optional(group: str, name: str, detail: str) -> Result:
    return Result(group, name, "OPTIONAL", detail)


def check_file_set(group: str, name: str, root: Path,
                   relative_paths: Iterable[str]) -> Result:
    required = [root / item for item in relative_paths]
    absent = [str(path) for path in required if not nonempty(path)]
    if not root.is_dir():
        return missing(group, name, f"directory not found: {root}")
    if absent:
        return broken(group, name, "missing or empty: " + ", ".join(absent))
    return ready(group, name, str(root))


def check_split(name: str, split: Path, expected_count: int) -> Result:
    if not split.is_dir():
        return missing("Data", name, f"directory not found: {split}")
    expected = [split / "mix_clean", split / "s1", split / "s2"]
    counts = {path.name: len(list(path.glob("*.wav"))) for path in expected}
    bad = {key: count for key, count in counts.items() if count != expected_count}
    detail = ", ".join(f"{key}={count}" for key, count in counts.items())
    if bad:
        return broken("Data", name, f"expected {expected_count} each; {detail}")
    return ready("Data", name, f"{split} ({detail})")


def check_manifest(name: str, path: Path, expected_count: int,
                   required_fields: Iterable[str]) -> Result:
    if not nonempty(path):
        return missing("Manifest", name, f"file not found or empty: {path}")
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                absent = [field for field in required_fields if not row.get(field)]
                if absent:
                    return broken(
                        "Manifest", name,
                        f"line {line_number} missing fields: {', '.join(absent)}",
                    )
                count += 1
    except (OSError, json.JSONDecodeError) as exc:
        return broken("Manifest", name, str(exc))
    if count != expected_count:
        return broken("Manifest", name, f"expected {expected_count}, found {count}")
    return ready("Manifest", name, f"{path} ({count} trials)")


def build_results(root: Path) -> list[Result]:
    wesep_repo = root / "external" / "wesep-real-tse"
    wesep_ckpt = root / "pretrained" / "wesep" / "spk_emb_100"
    cosy_repo = root / "external" / "CosyVoice"
    cosy_model = root / "pretrained" / "Fun-CosyVoice3-0.5B"
    qwen = root / "pretrained" / "Qwen2.5-0.5B-Instruct"
    wavlm = root / "pretrained" / "wavlm-base-plus"
    librimix = root / "datasets" / "LibriMix" / "wav16k" / "min"
    manifests = root / "manifests"
    debug_manifests = root / "debug_manifests"

    results = [
        check_file_set("WeSep", "repo", wesep_repo,
                       ["wesep/__init__.py", "evaluate.py"]),
        check_file_set("WeSep", "spk_emb_100", wesep_ckpt,
                       ["avg_model.pt", "config.yaml"]),
        check_file_set("CosyVoice3", "repo", cosy_repo,
                       ["README.md", "cosyvoice/cli/cosyvoice.py"]),
        check_file_set(
            "CosyVoice3", "model", cosy_model,
            ["cosyvoice3.yaml", "speech_tokenizer_v3.onnx", "flow.pt", "hift.pt"],
        ),
    ]

    qwen_weights = list(qwen.glob("*.safetensors")) + list(qwen.glob("pytorch_model*.bin"))
    qwen_tokenizers = [qwen / "tokenizer.json", qwen / "tokenizer_config.json"]
    if not qwen.is_dir():
        results.append(missing("Qwen", "Qwen2.5-0.5B-Instruct", str(qwen)))
    elif not nonempty(qwen / "config.json") or not any_nonempty(qwen_tokenizers) or not any_nonempty(qwen_weights):
        results.append(broken("Qwen", "Qwen2.5-0.5B-Instruct", "config, tokenizer, or weights missing/empty"))
    else:
        results.append(ready("Qwen", "Qwen2.5-0.5B-Instruct", str(qwen)))

    wavlm_weights = list(wavlm.glob("*.safetensors")) + list(wavlm.glob("pytorch_model*.bin"))
    if not wavlm.is_dir():
        results.append(missing("WavLM", "wavlm-base-plus", str(wavlm)))
    elif not nonempty(wavlm / "config.json") or not any_nonempty(wavlm_weights):
        results.append(broken("WavLM", "wavlm-base-plus", "config or weights missing/empty"))
    else:
        results.append(ready("WavLM", "wavlm-base-plus", str(wavlm)))

    results.extend([
        check_split("train-100", librimix / "train-100", 13900),
        check_split("dev", librimix / "dev", 3000),
        check_split("test", librimix / "test", 3000),
    ])

    base_fields = (
        "trial_id", "mixture_wav", "enrollment_wav", "target_wav",
        "target_speaker", "target_utterance", "enrollment_utterance",
    )
    prepared_fields = base_fields + (
        "evidence_token_path", "target_token_path", "speaker_embedding_path",
    )
    results.extend([
        check_manifest("train", manifests / "tse_train.jsonl", 27800, base_fields),
        check_manifest("dev", manifests / "tse_dev.jsonl", 6000, base_fields),
        check_manifest("test", manifests / "tse_test.jsonl", 6000, base_fields),
        check_manifest(
            "debug prepared",
            debug_manifests / "tse_debug_4trials_prepared.jsonl",
            4,
            prepared_fields,
        ),
        check_manifest(
            "train prepared",
            manifests / "tse_train_prepared.jsonl",
            27800,
            prepared_fields,
        ),
        check_manifest(
            "dev prepared",
            manifests / "tse_dev_prepared.jsonl",
            6000,
            prepared_fields,
        ),
        check_manifest(
            "test prepared",
            manifests / "tse_test_prepared.jsonl",
            6000,
            prepared_fields,
        ),
    ])

    for name, path in [("se_align/tse", root / "se_align" / "tse"),
                       ("se_align/data", root / "se_align" / "data")]:
        init_file = path / "__init__.py"
        if path.is_dir() and nonempty(init_file):
            results.append(ready("Code", name, str(path)))
        elif path.is_dir():
            results.append(broken("Code", name, f"missing or empty: {init_file}"))
        else:
            results.append(missing("Code", name, f"directory not found: {path}"))

    evase_dir = root / "pretrained" / "evase"
    evase_weights = []
    if evase_dir.is_dir():
        for pattern in ("*.pt", "*.pth", "*.ckpt"):
            evase_weights.extend(evase_dir.rglob(pattern))
    if any_nonempty(evase_weights):
        results.append(ready("Optional", "EvaSE checkpoint", str(evase_weights[0])))
    else:
        results.append(optional("Optional", "EvaSE checkpoint", "not found; Qwen initialization remains available"))

    generator_dir = root / "exp" / "tse_qwen_wavlm_fsq"
    generator_weights = []
    if generator_dir.is_dir():
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
            generator_weights.extend(generator_dir.rglob(pattern))
    if any_nonempty(generator_weights):
        results.append(ready("Optional", "TSE generator checkpoint", str(generator_weights[0])))
    else:
        results.append(optional("Optional", "TSE generator checkpoint", "not trained yet; decoding quality is not meaningful"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TSE project root (defaults to the repository containing this script)",
    )
    parser.add_argument("--strict", action="store_true",
                        help="return a nonzero exit code when required assets are not ready")
    args = parser.parse_args()
    root = args.project_root.resolve()
    results = build_results(root)

    groups: list[str] = []
    for result in results:
        if result.group not in groups:
            groups.append(result.group)
    for group in groups:
        print(f"[{group}]")
        for result in results:
            if result.group == group:
                print(f"{result.status:8s} {result.name}: {result.detail}")

    status = {(result.group, result.name): result.status for result in results}
    preprocessing_keys = [
        ("WeSep", "repo"),
        ("WeSep", "spk_emb_100"),
        ("CosyVoice3", "repo"),
        ("CosyVoice3", "model"),
        ("Data", "train-100"),
        ("Data", "dev"),
        ("Data", "test"),
        ("Manifest", "train"),
        ("Manifest", "dev"),
        ("Manifest", "test"),
    ]
    preprocessing = all(status.get(key) == "READY" for key in preprocessing_keys)
    smoke = preprocessing and status.get(("Manifest", "debug prepared")) == "READY"
    training_keys = [
        ("Qwen", "Qwen2.5-0.5B-Instruct"),
        ("WavLM", "wavlm-base-plus"),
        ("Code", "se_align/tse"),
        ("Code", "se_align/data"),
        ("Manifest", "train prepared"),
        ("Manifest", "dev prepared"),
    ]
    training = preprocessing and all(status.get(key) == "READY" for key in training_keys)
    decoding = (
        training
        and status.get(("Manifest", "test prepared")) == "READY"
        and status.get(("Optional", "TSE generator checkpoint")) == "READY"
    )

    print()
    print(f"READY_FOR_PREPROCESSING={'yes' if preprocessing else 'no'}")
    print(f"READY_FOR_TRAINING_SMOKE={'yes' if smoke else 'no'}")
    print(f"READY_FOR_TRAINING={'yes' if training else 'no'}")
    print(f"READY_FOR_DECODING={'yes' if decoding else 'no'}")
    if args.strict and not (preprocessing and training and decoding):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
