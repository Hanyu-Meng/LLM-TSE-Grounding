"""Command-line interface for the portable reference operators."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_selection import POOL_D, select_from_candidate_record
from .csg import select_csg_tokens
from .difficulty import residual_difficulty, source_threshold_lambda
from .gnr import refine_gnr


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select(args: argparse.Namespace) -> int:
    order = tuple(part.strip() for part in args.pool.split(",") if part.strip())
    rows = _read_jsonl(args.input)
    output = []
    for row in rows:
        decision = select_from_candidate_record(row["candidates"], order, args.score_key)
        output.append({
            "trial_id": row["trial_id"],
            "selected_candidate": decision.candidate,
            "selected_score": decision.score,
            "candidate_order": list(decision.candidate_order),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    temporary.replace(args.output)
    return 0


def _csg(args: argparse.Namespace) -> int:
    logits = np.load(args.logits, allow_pickle=False)
    evidence = np.load(args.evidence, allow_pickle=False).reshape(-1)
    tokens = select_csg_tokens(logits, evidence, args.grounding_lambda, args.temporal_tolerance)
    _atomic_npy(args.output, tokens)
    return 0


def _gnr(args: argparse.Namespace) -> int:
    logits = np.load(args.logits, allow_pickle=False)
    anchor = np.load(args.anchor, allow_pickle=False).reshape(-1)
    result = refine_gnr(logits, anchor, args.top_k, args.radius)
    _atomic_npy(args.output, result.tokens)
    payload = asdict(result)
    payload.pop("tokens")
    _atomic_json(args.stats, payload)
    return 0


def _difficulty(args: argparse.Namespace) -> int:
    mixture = np.load(args.mixture, allow_pickle=False)
    evidence = np.load(args.evidence, allow_pickle=False)
    result = residual_difficulty(mixture, evidence)
    result["selected_lambda"] = source_threshold_lambda(result["difficulty_value_db"])
    _atomic_json(args.output, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-tse-grounding",
        description="Portable reference operators for Repair Before Grounding.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="run target-enrollment cosine selection")
    select.add_argument("--input", type=Path, required=True, help="candidate JSONL")
    select.add_argument("--output", type=Path, required=True, help="selection JSONL")
    select.add_argument("--pool", default=",".join(POOL_D), help="ordered candidate names")
    select.add_argument("--score-key", default="speaker_similarity_to_enrollment")
    select.set_defaults(func=_select)

    csg = sub.add_parser("csg", help="apply CSG to a supplied logit matrix")
    csg.add_argument("--logits", type=Path, required=True, help="[T,6561] .npy")
    csg.add_argument("--evidence", type=Path, required=True, help="evidence token .npy")
    csg.add_argument("--lambda", dest="grounding_lambda", type=float, required=True)
    csg.add_argument("--temporal-tolerance", type=int, default=0)
    csg.add_argument("--output", type=Path, required=True, help="selected token .npy")
    csg.set_defaults(func=_csg)

    gnr = sub.add_parser("gnr", help="refine an immutable anchor from teacher-forced logits")
    gnr.add_argument("--logits", type=Path, required=True, help="[T,6561] .npy")
    gnr.add_argument("--anchor", type=Path, required=True, help="anchor token .npy")
    gnr.add_argument("--top-k", type=int, default=20)
    gnr.add_argument("--radius", type=int, default=2)
    gnr.add_argument("--output", type=Path, required=True, help="refined token .npy")
    gnr.add_argument("--stats", type=Path, required=True, help="audit statistics JSON")
    gnr.set_defaults(func=_gnr)

    difficulty = sub.add_parser("difficulty", help="compute residual-ratio difficulty")
    difficulty.add_argument("--mixture", type=Path, required=True, help="waveform .npy")
    difficulty.add_argument("--evidence", type=Path, required=True, help="waveform .npy")
    difficulty.add_argument("--output", type=Path, required=True, help="summary JSON")
    difficulty.set_defaults(func=_difficulty)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
