"""Baseline reconstruction models (the bar every generator model must clear).

Every ``fit_*`` returns a dict with at least:
    What  : (k, d, d) reconstruction
    cost  : ParamCost
    name  : str
plus model-specific factors. Errors are computed separately via ``metrics.summarize``.
"""

from __future__ import annotations

import numpy as np

from . import linalg
from .metrics import ParamCost, lowrank_params


# ---------------------------------------------------------------------------
# 1. Independent low-rank   E_ind(r) = sum_l min_{rank<=r} ||W_l - R_l||^2
# ---------------------------------------------------------------------------
def fit_independent_lowrank(W: np.ndarray, r: int) -> dict:
    k, d, _ = W.shape
    What = np.stack([linalg.best_rank_r(W[i], r) for i in range(k)], axis=0)
    cost = ParamCost(shared=0, per_layer=lowrank_params(r, d), k=k, d=d)
    return {"name": "ind", "What": What.astype(W.dtype), "cost": cost, "r": r}


# ---------------------------------------------------------------------------
# 2. Shared mean   B = mean_l W_l  (== shared base, r=0)
# ---------------------------------------------------------------------------
def fit_shared_mean(W: np.ndarray) -> dict:
    k, d, _ = W.shape
    B = W.mean(axis=0)
    What = np.broadcast_to(B, W.shape).copy()
    cost = ParamCost(shared=d * d, per_layer=0, k=k, d=d)
    return {"name": "mean", "What": What.astype(W.dtype), "cost": cost, "B": B}


# ---------------------------------------------------------------------------
# 3. Shared base + low-rank residual
#    E_shared(r) = min_B sum_l min_{rank(R_l)<=r} ||W_l - B - R_l||^2
# ---------------------------------------------------------------------------
def fit_shared_base_lowrank(
    W: np.ndarray, r: int, n_iter: int = 30, tol: float = 1e-7
) -> dict:
    """Alternating minimization: B = mean_l(W_l - R_l); R_l = SVD_r(W_l - B)."""
    k, d, _ = W.shape
    Wd = W.astype(np.float64)
    B = Wd.mean(axis=0)
    R = np.zeros_like(Wd)
    prev = np.inf
    for _ in range(max(1, n_iter)):
        # residual step
        for i in range(k):
            R[i] = linalg.best_rank_r(Wd[i] - B, r)
        # base step
        B = (Wd - R).mean(axis=0)
        err = float(((Wd - B - R) ** 2).sum())
        if abs(prev - err) <= tol * max(prev, 1.0):
            break
        prev = err
        if r == 0:
            break  # R stays 0; B=mean is the exact optimum in one step
    What = (B[None] + R).astype(W.dtype)
    cost = ParamCost(shared=d * d, per_layer=lowrank_params(r, d), k=k, d=d)
    return {"name": "shared", "What": What, "cost": cost, "B": B, "r": r}


# ---------------------------------------------------------------------------
# 4. Clustering + low-rank residual
#    Partition layers into c clusters, each with its own base B_g = cluster mean.
# ---------------------------------------------------------------------------
def fit_clustering_lowrank(
    W: np.ndarray, r: int, n_clusters: int, seed: int = 0
) -> dict:
    from sklearn.cluster import KMeans

    k, d, _ = W.shape
    n_clusters = min(n_clusters, k)
    feats = W.reshape(k, -1).astype(np.float64)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(feats)

    What = np.empty_like(W, dtype=np.float64)
    bases = {}
    for g in range(n_clusters):
        idx = np.where(labels == g)[0]
        if idx.size == 0:
            continue
        Bg = W[idx].astype(np.float64).mean(axis=0)
        bases[g] = Bg
        for i in idx:
            What[i] = Bg + linalg.best_rank_r(W[i].astype(np.float64) - Bg, r)
    cost = ParamCost(
        shared=n_clusters * d * d, per_layer=lowrank_params(r, d), k=k, d=d
    )
    return {
        "name": "cluster", "What": What.astype(W.dtype), "cost": cost,
        "labels": labels.tolist(), "n_clusters": n_clusters, "r": r,
    }
