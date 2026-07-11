"""Null models: control groups with *no genuine cross-layer structure*.

If a shared-structure model improves over the independent baseline by the *same* margin
on a null group as on real weights, the apparent 'sharing' is an artifact of each layer's
spectrum, not real cross-layer algebra. Every headline gain must survive these controls.

Nulls provided:
  * spectrum_matched_random  -- each layer: same singular values, independent Haar U,V
  * rotated_matched          -- each layer: W_l conjugated by an independent rotation
  * permuted_layers          -- shuffle which matrix is 'layer l' (identity for a single
                                group's internal stats, but used across paired groups)
  * gaussian_matched_fro     -- i.i.d. Gaussian with each layer's Frobenius norm
"""

from __future__ import annotations

import numpy as np

from . import linalg


def _haar(d: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-random orthogonal matrix via QR of a Gaussian."""
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))  # fix signs for a proper Haar measure
    return Q


def spectrum_matched_random(W: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same per-layer singular values, but independent random singular subspaces."""
    rng = np.random.default_rng(seed)
    k, d, _ = W.shape
    out = np.empty_like(W, dtype=np.float64)
    for i in range(k):
        s = linalg.singular_values(W[i])
        U = _haar(d, rng)
        V = _haar(d, rng)
        out[i] = (U * s) @ V.T
    return out.astype(W.dtype)


def rotated_matched(W: np.ndarray, seed: int = 0) -> np.ndarray:
    """Conjugate each layer by an independent rotation: Q_i W_i Q_i^T."""
    rng = np.random.default_rng(seed)
    k, d, _ = W.shape
    out = np.empty_like(W, dtype=np.float64)
    for i in range(k):
        Q = _haar(d, rng)
        out[i] = Q @ W[i].astype(np.float64) @ Q.T
    return out.astype(W.dtype)


def gaussian_matched_fro(W: np.ndarray, seed: int = 0) -> np.ndarray:
    """i.i.d. Gaussian matrices matching each layer's Frobenius norm."""
    rng = np.random.default_rng(seed)
    k, d, _ = W.shape
    out = np.empty_like(W, dtype=np.float64)
    for i in range(k):
        G = rng.standard_normal((d, d))
        fro = np.linalg.norm(W[i])
        out[i] = G * (fro / np.linalg.norm(G))
    return out.astype(W.dtype)


def permuted_layers(W: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffle the layer order (a control for pairing across two groups)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(W.shape[0])
    return W[idx].copy()


NULLS = {
    "spectrum_matched": spectrum_matched_random,
    "rotated_matched": rotated_matched,
    "gaussian_fro": gaussian_matched_fro,
}
