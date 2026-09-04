"""Grounded neighborhood refinement (GNR) over an immutable anchor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fsq import VOCAB_SIZE, hamming_distance, ids_to_digits


@dataclass(frozen=True)
class GNRResult:
    """Refined tokens and audit-friendly summary statistics."""

    tokens: NDArray[np.int64]
    edit_count: int
    edit_rate: float
    candidate_set_size_min: int
    candidate_set_size_mean: float
    candidate_set_size_max: int
    mean_hamming_edit_distance: float
    max_hamming_edit_distance: int
    refined_tokens_fed_back: bool = False
    teacher_forced_history: str = "immutable_anchor_prefix"


def refine_gnr(
    teacher_forced_logits: ArrayLike,
    anchor_ids: ArrayLike,
    top_k: int = 20,
    radius: int = 2,
) -> GNRResult:
    """Refine an anchor with ``Top-K ∩ HammingBall(radius) ∪ {anchor}``.

    ``teacher_forced_logits`` must be computed once using the complete,
    original anchor prefix.  Refined tokens are never fed back into later
    positions.  The anchor wins exact score ties.
    """

    logits = np.asarray(teacher_forced_logits)
    anchor = np.asarray(anchor_ids)
    if logits.ndim != 2 or logits.shape[1] != VOCAB_SIZE:
        raise ValueError(f"teacher_forced_logits must have shape [T, {VOCAB_SIZE}]")
    if not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
        raise ValueError("teacher_forced_logits must be finite floating values")
    if anchor.ndim != 1 or anchor.shape[0] != logits.shape[0]:
        raise ValueError("anchor_ids must be one-dimensional with length T")
    ids_to_digits(anchor)
    anchor = anchor.astype(np.int64, copy=False)
    if not 1 <= top_k <= VOCAB_SIZE:
        raise ValueError(f"top_k must be in [1, {VOCAB_SIZE}]")
    if not 0 <= radius <= 8:
        raise ValueError("radius must be in [0, 8]")

    refined = anchor.copy()
    candidate_sizes: list[int] = []
    for step, row in enumerate(logits):
        # Stable full ordering keeps tie behavior reproducible for the reference code.
        top_ids = np.argsort(-row, kind="stable")[:top_k].astype(np.int64)
        allowed = top_ids[hamming_distance(top_ids, anchor[step]) <= radius]
        candidate_sizes.append(int(np.union1d(allowed, anchor[step]).size))

        best_id = int(anchor[step])
        best_score = float(row[best_id])
        for candidate_id in allowed:
            score = float(row[int(candidate_id)])
            if score > best_score:
                best_id = int(candidate_id)
                best_score = score
        refined[step] = best_id

    edit_mask = refined != anchor
    edit_distances = hamming_distance(refined, anchor)
    return GNRResult(
        tokens=refined,
        edit_count=int(edit_mask.sum()),
        edit_rate=float(edit_mask.mean()) if edit_mask.size else 0.0,
        candidate_set_size_min=min(candidate_sizes),
        candidate_set_size_mean=float(np.mean(candidate_sizes)),
        candidate_set_size_max=max(candidate_sizes),
        mean_hamming_edit_distance=float(edit_distances.mean()),
        max_hamming_edit_distance=int(edit_distances.max(initial=0)),
    )
