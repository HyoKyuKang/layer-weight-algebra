"""Shared matrix dictionary model.

    W_l ~= sum_{j in S_l} c_lj D_j + R_l,   |S_l| <= s,  rank(R_l) <= r

The dictionary atoms D_1..D_q are *unconstrained* d x d matrices shared across layers.
This is the key competitor the generator-word model must beat at a matched budget:
words re-use g generators via products, a dictionary spends a full d^2 per atom.

Fit: alternating between
  (a) sparse coding of (W_l - R_l) over the current atoms (OMP, <= s nonzeros);
  (b) dictionary update by least squares over the codes;
  (c) residual R_l = best-rank-r(W_l - D c_l).
Initialized from the top-q principal directions of the flattened stack.
"""

from __future__ import annotations

import numpy as np

from . import linalg
from .metrics import ParamCost, lowrank_params


def _omp_codes(X: np.ndarray, D: np.ndarray, s: int) -> np.ndarray:
    """Sparse codes C (k x q) s.t. X ~= C D, <= s nonzeros/row. X:(k,P), D:(q,P)."""
    from sklearn.linear_model import orthogonal_mp

    # orthogonal_mp solves min ||y - Dic gamma|| with Dic columns as atoms.
    Dic = D.T  # (P, q)
    C = orthogonal_mp(Dic, X.T, n_nonzero_coefs=min(s, D.shape[0])).T  # (k,q)
    return np.asarray(C)


def fit_dictionary(
    W: np.ndarray, q: int, s: int, r: int, n_iter: int = 15, seed: int = 0
) -> dict:
    k, d, _ = W.shape
    P = d * d
    X = W.reshape(k, P).astype(np.float64)
    q = min(q, k)

    # init atoms from top-q right singular vectors of the stack
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    D = Vt[:q].copy()  # (q, P)

    R = np.zeros_like(X)
    C = np.zeros((k, q))
    prev = np.inf
    for _ in range(n_iter):
        target = X - R
        C = _omp_codes(target, D, s)
        # dictionary update: D = argmin ||target - C D||  -> lstsq
        D, *_ = np.linalg.lstsq(C, target, rcond=None)
        recon = C @ D
        # residual update
        resid_full = (X - recon)
        for i in range(k):
            R[i] = linalg.best_rank_r(resid_full[i].reshape(d, d), r).reshape(P)
        err = float(((X - recon - R) ** 2).sum())
        if abs(prev - err) <= 1e-9 * max(prev, 1.0):
            break
        prev = err

    What = (C @ D + R).reshape(k, d, d).astype(W.dtype)
    # per-layer floats: s coefficients + rank-r residual. (atom indices cost
    # s*ceil(log2 q) bits, tracked separately in `index_bits`.)
    per_layer = s + lowrank_params(r, d)
    cost = ParamCost(shared=q * P, per_layer=per_layer, k=k, d=d)
    index_bits = k * s * int(np.ceil(np.log2(max(q, 2))))
    return {
        "name": "dict", "What": What, "cost": cost,
        "q": q, "s": s, "r": r, "index_bits": index_bits,
        "D": D.reshape(q, d, d), "C": C,
    }
