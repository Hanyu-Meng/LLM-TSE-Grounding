"""Ternary FSQ geometry used by the CosyVoice3 S3 token stream.

The vocabulary contains ``3**8 = 6561`` raw audio IDs.  IDs use the
little-endian basis ``[1, 3, 9, ..., 3**7]``.  Hamming distance in this
eight-coordinate space is the grounding geometry used by CSG and GNR.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

BASE = 3
N_DIGITS = 8
VOCAB_SIZE = BASE**N_DIGITS
POWERS = BASE ** np.arange(N_DIGITS, dtype=np.int64)


def _validated_ids(ids: ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(ids)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("FSQ token IDs must be integers")
    values = values.astype(np.int64, copy=False)
    if values.size and (int(values.min()) < 0 or int(values.max()) >= VOCAB_SIZE):
        raise ValueError(f"FSQ token IDs must be in [0, {VOCAB_SIZE - 1}]")
    return values


def ids_to_digits(ids: ArrayLike) -> NDArray[np.int64]:
    """Convert raw token IDs to little-endian ternary coordinates.

    The returned shape is ``input_shape + (8,)``.
    """

    values = _validated_ids(ids)
    flat = values.reshape(-1).copy()
    digits = np.empty((flat.size, N_DIGITS), dtype=np.int64)
    for position in range(N_DIGITS):
        digits[:, position] = flat % BASE
        flat //= BASE
    return digits.reshape(values.shape + (N_DIGITS,))


def digits_to_ids(digits: ArrayLike) -> NDArray[np.int64]:
    """Convert ternary coordinates back to raw token IDs."""

    values = np.asarray(digits)
    if values.ndim == 0 or values.shape[-1] != N_DIGITS:
        raise ValueError(f"last dimension must contain {N_DIGITS} FSQ digits")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("FSQ digits must be integers")
    values = values.astype(np.int64, copy=False)
    if values.size and (int(values.min()) < 0 or int(values.max()) >= BASE):
        raise ValueError(f"FSQ digits must be in [0, {BASE - 1}]")
    return np.sum(values * POWERS, axis=-1, dtype=np.int64)


def hamming_distance(a: ArrayLike, b: ArrayLike) -> NDArray[np.int64]:
    """Return elementwise FSQ Hamming distance with NumPy broadcasting."""

    return np.sum(ids_to_digits(a) != ids_to_digits(b), axis=-1, dtype=np.int64)


def codebook_digits() -> NDArray[np.int64]:
    """Return the complete ``[6561, 8]`` FSQ codebook."""

    return ids_to_digits(np.arange(VOCAB_SIZE, dtype=np.int64))
