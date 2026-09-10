"""FSQ Hamming-neighborhood cache for the 6561-token (3^8) S3 vocabulary.

For every token id, decompose into its 8 ternary FSQ digits and precompute the
ids within Hamming distance <= r. Used by the evidence trust-region training
loss (and reusable at decode time).

Neighborhood sizes: r=1 -> 17, r=2 -> 129, r=3 -> 577 (incl. the token itself).

Note on vocab mapping: the SEModel audio head is a dedicated slice of the LM
head indexed by RAW audio ids (0..6560 + specials), so neighbor ids can be
used directly as gather indices into that slice. If a caller ever works with
full-tokenizer logits instead, shift ids by `vocab.audio_shift` first.
"""
from __future__ import annotations

import os
from itertools import combinations, product

import numpy as np
import torch

N_DIGITS = 8
BASE = 3
VOCAB = BASE ** N_DIGITS  # 6561


def _digits(ids: np.ndarray) -> np.ndarray:
    """[N] token ids -> [N, 8] ternary digits (least-significant first)."""
    d = np.empty((len(ids), N_DIGITS), dtype=np.int64)
    x = ids.copy()
    for k in range(N_DIGITS):
        d[:, k] = x % BASE
        x //= BASE
    return d


def neighborhood_size(r: int) -> int:
    from math import comb
    return sum(comb(N_DIGITS, k) * (BASE - 1) ** k for k in range(r + 1))


def build_neighbors(r: int) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """neighbor_ids [6561, K], neighbor_mask [6561, K] for Hamming <= r.

    All rows are full (every token has the same neighborhood size in a
    complete 3^8 code space), so the mask is all-True; it is kept for
    interface stability (callers must respect it if the space ever changes).
    """
    K = neighborhood_size(r)
    ids = np.arange(VOCAB, dtype=np.int64)
    digits = _digits(ids)                                   # [V, 8]
    pow3 = BASE ** np.arange(N_DIGITS, dtype=np.int64)       # [8]

    out = np.empty((VOCAB, K), dtype=np.int64)
    out[:, 0] = ids
    col = 1
    for k in range(1, r + 1):
        for pos in combinations(range(N_DIGITS), k):
            pos = list(pos)
            base_contrib = (digits[:, pos] * pow3[pos]).sum(axis=1)  # [V]
            rest = ids - base_contrib
            for deltas in product(range(1, BASE), repeat=k):
                newd = (digits[:, pos] + np.array(deltas)) % BASE     # [V, k]
                out[:, col] = rest + (newd * pow3[pos]).sum(axis=1)
                col += 1
    assert col == K, (col, K)
    return torch.from_numpy(out), torch.ones(VOCAB, K, dtype=torch.bool)


def load_neighbors(r: int, cache_path: str = "") -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Build or load the cached neighborhood tables."""
    path = cache_path or f"results/.fsq_neighbors_r{r}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return torch.from_numpy(z["ids"]), torch.from_numpy(z["mask"])
    ids, mask = build_neighbors(r)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, ids=ids.numpy(), mask=mask.numpy())
    os.replace(tmp, path)
    return ids, mask


def hamming_digits(a: torch.LongTensor, b: torch.LongTensor) -> torch.LongTensor:
    """Elementwise FSQ Hamming distance between two id tensors (same shape).
    Invalid ids (<0 or >=6561) must be masked by the caller."""
    d = torch.zeros_like(a)
    x, y = a.clone(), b.clone()
    for _ in range(N_DIGITS):
        d += (x % BASE != y % BASE).long()
        x //= BASE
        y //= BASE
    return d
