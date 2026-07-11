"""Low-level linear-algebra helpers shared across models."""

from __future__ import annotations

import numpy as np


def truncated_svd(A: np.ndarray, r: int):
    """Return U (d x r), s (r,), Vt (r x d) of the best rank-r approximation of A.

    r == 0 returns empty factors. r >= min(shape) returns the full SVD.
    """
    if r <= 0:
        d0, d1 = A.shape
        return (np.zeros((d0, 0)), np.zeros((0,)), np.zeros((0, d1)))
    U, s, Vt = np.linalg.svd(A.astype(np.float64), full_matrices=False)
    r = min(r, s.shape[0])
    return U[:, :r], s[:r], Vt[:r]


def best_rank_r(A: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r approximation of A in Frobenius/spectral norm (Eckart-Young)."""
    if r <= 0:
        return np.zeros_like(A)
    U, s, Vt = truncated_svd(A, r)
    return (U * s) @ Vt


def lowrank_factors(A: np.ndarray, r: int):
    """Return (U, V) with U V^T the best rank-r approx; U=(d,r) folds in sqrt(s)."""
    U, s, Vt = truncated_svd(A, r)
    root = np.sqrt(s)
    return U * root, (Vt.T * root)


def singular_values(A: np.ndarray) -> np.ndarray:
    return np.linalg.svd(A.astype(np.float64), compute_uv=False)


def stable_rank(A: np.ndarray) -> float:
    """||A||_F^2 / ||A||_2^2 -- a soft, noise-robust rank estimate."""
    s = singular_values(A)
    top = s[0] ** 2 if s.size else 0.0
    return float((s ** 2).sum() / top) if top > 0 else 0.0
