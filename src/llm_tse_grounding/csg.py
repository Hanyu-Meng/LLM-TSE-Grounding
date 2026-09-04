"""Code-space grounding (CSG) operators.

At output step ``t`` CSG subtracts ``lambda * d`` from every audio-token
logit, where ``d`` is the minimum FSQ Hamming distance to the evidence token
at ``t +/- temporal_tolerance``.  The operator is intentionally independent
of a particular language-model implementation so it can be inserted into an
online autoregressive decoder without changing the model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fsq import VOCAB_SIZE, codebook_digits, ids_to_digits


def _validate_logits(logits: ArrayLike) -> NDArray[np.floating]:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != VOCAB_SIZE:
        raise ValueError(f"logits must have shape [T, {VOCAB_SIZE}]")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("logits must use a floating dtype")
    if not np.isfinite(values).all():
        raise ValueError("logits contain NaN or infinity")
    return values


def _validate_evidence(evidence_ids: ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(evidence_ids)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("evidence_ids must be a non-empty one-dimensional sequence")
    # ids_to_digits performs type/range validation.
    ids_to_digits(values)
    return values.astype(np.int64, copy=False)


def csg_distance_rows(
    evidence_ids: ArrayLike,
    output_steps: int,
    temporal_tolerance: int = 0,
) -> NDArray[np.int64]:
    """Build the ``[T, 6561]`` minimum-Hamming penalty table."""

    evidence = _validate_evidence(evidence_ids)
    if output_steps <= 0 or output_steps > evidence.size:
        raise ValueError("output_steps must be in [1, len(evidence_ids)]")
    if temporal_tolerance < 0:
        raise ValueError("temporal_tolerance must be non-negative")

    candidates = codebook_digits()
    distances = np.empty((output_steps, VOCAB_SIZE), dtype=np.int8)
    for step in range(output_steps):
        positions = np.arange(
            step - temporal_tolerance,
            step + temporal_tolerance + 1,
            dtype=np.int64,
        )
        positions = np.clip(positions, 0, evidence.size - 1)
        references = ids_to_digits(evidence[positions])
        per_reference = np.sum(
            candidates[None, :, :] != references[:, None, :], axis=-1
        )
        distances[step] = per_reference.min(axis=0)
    return distances


def csg_penalize_logits(
    logits: ArrayLike,
    evidence_ids: ArrayLike,
    grounding_lambda: float | ArrayLike,
    temporal_tolerance: int = 0,
) -> NDArray[np.floating]:
    """Return a copy of logits after the CSG Hamming penalty."""

    values = _validate_logits(logits)
    lambdas = np.asarray(grounding_lambda, dtype=np.float64)
    if lambdas.ndim == 0:
        lambdas = np.full(values.shape[0], float(lambdas), dtype=np.float64)
    if lambdas.shape != (values.shape[0],) or not np.isfinite(lambdas).all():
        raise ValueError("grounding_lambda must be finite and scalar or length T")
    if np.any(lambdas < 0):
        raise ValueError("grounding_lambda must be non-negative")

    if np.all(lambdas == 0):
        return values.copy()
    distances = csg_distance_rows(evidence_ids, values.shape[0], temporal_tolerance)
    return values - lambdas[:, None].astype(values.dtype) * distances.astype(values.dtype)


def select_csg_tokens(
    logits: ArrayLike,
    evidence_ids: ArrayLike,
    grounding_lambda: float | ArrayLike,
    temporal_tolerance: int = 0,
) -> NDArray[np.int64]:
    """Greedily select one grounded raw FSQ ID per supplied logit row.

    In a real autoregressive decoder, call the penalty at each step before
    producing the next model state.  A precomputed logit matrix is suitable
    for operator audits and teacher-forced analyses.
    """

    grounded = csg_penalize_logits(
        logits, evidence_ids, grounding_lambda, temporal_tolerance
    )
    return grounded.argmax(axis=1).astype(np.int64)
