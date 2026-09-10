"""Deployable candidate selection for confusion repair.

The selector sees only similarity to the target enrollment.  Clean targets,
interferer references, transcripts, WER, and SI-SDR are evaluation-only and
must never be passed to this module at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

CDCS5_CANDIDATES = ("full", "first", "middle", "final", "tfmap_context_full")


@dataclass(frozen=True)
class Selection:
    """A deterministic candidate-selection decision."""

    candidate: str
    score: float
    candidate_order: tuple[str, ...]


def select_by_enrollment_similarity(
    scores: Mapping[str, float],
    candidate_order: Sequence[str] = CDCS5_CANDIDATES,
) -> Selection:
    """Select the candidate with maximum target-enrollment similarity.

    Ties are resolved by the declared candidate order, matching the frozen
    CDCS-5 protocol.  All requested candidates must have a finite score.
    """

    order = tuple(candidate_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("candidate_order must contain unique candidate names")
    missing = [name for name in order if name not in scores]
    if missing:
        raise KeyError(f"missing enrollment-similarity scores: {missing}")
    values = {name: float(scores[name]) for name in order}
    invalid = [name for name, value in values.items() if not isfinite(value)]
    if invalid:
        raise ValueError(f"non-finite enrollment-similarity scores: {invalid}")

    index = max(range(len(order)), key=lambda i: (values[order[i]], -i))
    winner = order[index]
    return Selection(candidate=winner, score=values[winner], candidate_order=order)


def select_from_candidate_record(
    candidates: Mapping[str, Mapping[str, float]],
    candidate_order: Sequence[str] = CDCS5_CANDIDATES,
    score_key: str = "speaker_similarity_to_enrollment",
) -> Selection:
    """Select directly from a JSON-like candidate metric record."""

    scores = {name: candidates[name][score_key] for name in candidate_order}
    return select_by_enrollment_similarity(scores, candidate_order)
