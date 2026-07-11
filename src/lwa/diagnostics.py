"""Cross-layer heterogeneity diagnostics.

These measure *whether a shared algebraic structure is even plausible* before we try to
fit one. Reported for a group W (k, d, d):

  * singular-value decay of each W_l and of W_l - B (centered);
  * pairwise-difference norms and effective ranks of W_i - W_j;
  * principal angles between row / column spaces of layer pairs;
  * normalized commutators ||W_i W_j - W_j W_i|| / (||W_i|| ||W_j||);
  * residual stable rank after removing a shared base.
"""

from __future__ import annotations

import numpy as np

from . import linalg


def spectra(W: np.ndarray) -> np.ndarray:
    """(k, d) array of singular values, one row per layer."""
    return np.stack([linalg.singular_values(W[i]) for i in range(W.shape[0])], axis=0)


def centered_spectra(W: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Singular values of W_l - B (B defaults to the layerwise mean)."""
    if B is None:
        B = W.mean(axis=0)
    return np.stack([linalg.singular_values(W[i] - B) for i in range(W.shape[0])], axis=0)


def stable_ranks(W: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Stable rank of each W_l (or W_l - B if B given)."""
    Wc = W if B is None else W - B[None]
    return np.array([linalg.stable_rank(Wc[i]) for i in range(W.shape[0])])


def pairwise_diff_norm(W: np.ndarray) -> np.ndarray:
    """Symmetric matrix D[i,j] = ||W_i - W_j||_F / sqrt(||W_i||_F ||W_j||_F)."""
    k = W.shape[0]
    fro = np.sqrt((W.astype(np.float64) ** 2).sum(axis=(1, 2)))
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            n = np.linalg.norm(W[i] - W[j])
            denom = np.sqrt(fro[i] * fro[j])
            D[i, j] = D[j, i] = n / denom if denom > 0 else 0.0
    return D


def pairwise_diff_stable_rank(W: np.ndarray) -> np.ndarray:
    """Stable rank of each W_i - W_j (upper triangle; diagonal = 0)."""
    k = W.shape[0]
    S = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            S[i, j] = S[j, i] = linalg.stable_rank(W[i] - W[j])
    return S


def normalized_commutator(A: np.ndarray, B: np.ndarray) -> float:
    """||AB - BA||_F / (||A||_F ||B||_F). 0 => commute; ~sqrt(2) => 'generic'."""
    na = np.linalg.norm(A)
    nb = np.linalg.norm(B)
    if na == 0 or nb == 0:
        return 0.0
    C = A @ B - B @ A
    return float(np.linalg.norm(C) / (na * nb))


def commutator_matrix(W: np.ndarray) -> np.ndarray:
    """Symmetric matrix of normalized commutators for all layer pairs."""
    k = W.shape[0]
    Wd = W.astype(np.float64)
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            C[i, j] = C[j, i] = normalized_commutator(Wd[i], Wd[j])
    return C


def principal_angles(A: np.ndarray, B: np.ndarray, rank: int) -> np.ndarray:
    """Principal angles (radians) between the top-`rank` column subspaces of A and B."""
    Ua, _, _ = linalg.truncated_svd(A, rank)
    Ub, _, _ = linalg.truncated_svd(B, rank)
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def mean_principal_angle_matrix(W: np.ndarray, rank: int, use: str = "col") -> np.ndarray:
    """Pairwise mean principal angle (radians) between top-`rank` subspaces.

    use='col' -> column spaces (left singular vectors of W);
    use='row' -> row spaces    (via W^T).
    """
    k = W.shape[0]
    mats = W if use == "col" else np.transpose(W, (0, 2, 1))
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            ang = principal_angles(mats[i], mats[j], rank)
            M[i, j] = M[j, i] = float(ang.mean()) if ang.size else 0.0
    return M


def summarize_group(W: np.ndarray, rank_for_angles: int = 16) -> dict:
    """A compact scalar summary of a group's heterogeneity."""
    B = W.mean(axis=0)
    sp = spectra(W)
    csp = centered_spectra(W, B)
    D = pairwise_diff_norm(W)
    C = commutator_matrix(W)
    iu = np.triu_indices(W.shape[0], k=1)
    energy_in_mean = float((B ** 2).sum() * W.shape[0]) / float((W.astype(np.float64) ** 2).sum())
    return {
        "k": int(W.shape[0]),
        "d": int(W.shape[1]),
        "mean_stable_rank": float(np.mean(stable_ranks(W))),
        "centered_mean_stable_rank": float(np.mean(stable_ranks(W, B))),
        "energy_fraction_in_mean": energy_in_mean,
        "pairwise_diff_norm_mean": float(D[iu].mean()),
        "pairwise_diff_norm_min": float(D[iu].min()),
        "pairwise_diff_norm_max": float(D[iu].max()),
        "commutator_mean": float(C[iu].mean()),
        "commutator_max": float(C[iu].max()),
        "top_sv_mean": float(sp[:, 0].mean()),
        "centered_top_sv_mean": float(csp[:, 0].mean()),
    }
