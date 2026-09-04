"""Small statistical utilities used by the frozen evaluation protocol."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial rate."""

    if not 0 <= successes <= total or total <= 0:
        raise ValueError("require 0 <= successes <= total and total > 0")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return center - half, center + half


def paired_bootstrap_difference(
    method: ArrayLike,
    baseline: ArrayLike,
    resamples: int = 10_000,
    seed: int = 1986,
) -> dict[str, float | int]:
    """Paired bootstrap CI for ``mean(method - baseline)``.

    Non-finite pairs are removed jointly.  Negative deltas are improvements
    for error metrics; positive deltas are improvements for quality metrics.
    """

    a = np.asarray(method, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("method and baseline must be non-empty and equally shaped")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    mask = np.isfinite(a) & np.isfinite(b)
    delta = a[mask] - b[mask]
    if delta.size == 0:
        raise ValueError("no finite paired observations")

    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    batch = 256
    for start in range(0, resamples, batch):
        stop = min(start + batch, resamples)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        samples[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "pairs": int(delta.size),
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_difference": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }
