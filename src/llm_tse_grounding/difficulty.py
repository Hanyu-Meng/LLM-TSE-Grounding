"""Deployable residual-ratio difficulty feature used in the noisy ablation.

This feature did *not* pass the preregistered validity gate as an SNR
estimator.  It is therefore named a difficulty feature, not estimated SNR.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

SOURCE_THRESHOLDS = np.asarray(
    [-3.013, -1.640, -0.267, 1.179, 2.697, 4.216, 7.515, 13.158],
    dtype=np.float64,
)
SOURCE_LAMBDAS = np.asarray(
    [2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0],
    dtype=np.float64,
)


def residual_difficulty(
    mixture: ArrayLike,
    evidence: ArrayLike,
    eps: float = 1e-10,
) -> dict[str, float]:
    """Compute the projection/residual energy-ratio feature in dB."""

    x = np.asarray(mixture, dtype=np.float64).reshape(-1)
    e = np.asarray(evidence, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.shape != e.shape:
        raise ValueError("mixture and evidence must be non-empty and have equal length")
    if not np.isfinite(x).all() or not np.isfinite(e).all():
        raise ValueError("mixture and evidence must be finite")
    if eps <= 0:
        raise ValueError("eps must be positive")

    evidence_energy = float(np.dot(e, e))
    alpha = max(0.0, float(np.dot(x, e)) / (evidence_energy + eps))
    projected = alpha * e
    residual = x - projected
    projected_energy = float(np.dot(projected, projected))
    residual_energy = float(np.dot(residual, residual))
    value_db = float(10.0 * np.log10((projected_energy + eps) / (residual_energy + eps)))
    return {
        "difficulty_value_db": value_db,
        "projection_scale": alpha,
        "evidence_energy": evidence_energy,
        "projected_energy": projected_energy,
        "residual_energy": residual_energy,
    }


def source_threshold_lambda(difficulty_value_db: float) -> float:
    """Apply the preregistered source-threshold lambda mapping."""

    value = float(difficulty_value_db)
    if not np.isfinite(value):
        raise ValueError("difficulty_value_db must be finite")
    index = int(np.searchsorted(SOURCE_THRESHOLDS, value, side="right"))
    return float(SOURCE_LAMBDAS[index])
