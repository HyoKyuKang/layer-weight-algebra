"""Error metrics and storage / compute accounting.

All models in this project reconstruct a stack ``What`` (k, d, d) of a same-role group
``W`` (k, d, d). To compare models *fairly* we always report, alongside the error:

  * ``params``: a ``ParamCost`` splitting floats into shared-once vs per-layer;
  * derived stored bits at a chosen precision;
  * a materialized inference multiply count.

The success criterion for this project is a *rate-distortion* one: lower error at the
**same stored bits**, so error must never be read without its cost.
"""

from __future__ import annotations

import dataclasses

import numpy as np


# ---------------------------------------------------------------------------
# Storage / compute accounting
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ParamCost:
    """Float-parameter accounting for a reconstruction of k matrices of size d x d."""

    shared: int          # floats stored once for the whole group (B, generators, atoms)
    per_layer: int       # floats stored for each layer (coeffs, U_l, V_l)
    k: int               # number of layers
    d: int               # matrix dimension

    @property
    def total(self) -> int:
        return self.shared + self.per_layer * self.k

    @property
    def dense_total(self) -> int:
        """Cost of just storing every matrix densely (the do-nothing baseline)."""
        return self.k * self.d * self.d

    def bits(self, bits_per_param: int = 16) -> int:
        return self.total * bits_per_param

    def compression_ratio(self) -> float:
        return self.dense_total / max(self.total, 1)

    def as_dict(self) -> dict:
        return {
            "shared_params": self.shared,
            "per_layer_params": self.per_layer,
            "total_params": self.total,
            "dense_params": self.dense_total,
            "compression_ratio": self.compression_ratio(),
        }


def lowrank_params(r: int, d: int) -> int:
    """Floats to store a rank-r residual U V^T with U,V in R^{d x r}."""
    return 2 * d * r if r > 0 else 0


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def frob_sq(A: np.ndarray) -> float:
    return float(np.sum(A.astype(np.float64) ** 2))


def normalized_frob_error(W: np.ndarray, What: np.ndarray) -> float:
    """||W - What||_F^2 / ||W||_F^2 over the whole stack."""
    num = frob_sq(W - What)
    den = frob_sq(W)
    return num / den if den > 0 else 0.0


def per_matrix_frob_error(W: np.ndarray, What: np.ndarray) -> np.ndarray:
    """Normalized squared Frobenius error for each matrix in the stack."""
    diff = ((W - What).astype(np.float64) ** 2).sum(axis=(1, 2))
    den = (W.astype(np.float64) ** 2).sum(axis=(1, 2))
    return diff / np.where(den > 0, den, 1.0)


def spectral_error(W: np.ndarray, What: np.ndarray) -> np.ndarray:
    """Per-matrix ||W-What||_2 / ||W||_2 (top singular value ratio)."""
    out = np.empty(W.shape[0])
    for i in range(W.shape[0]):
        s_diff = np.linalg.norm(W[i] - What[i], 2)
        s_w = np.linalg.norm(W[i], 2)
        out[i] = s_diff / s_w if s_w > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(
    W: np.ndarray,
    What: np.ndarray,
    cost: ParamCost,
    bits_per_param: int = 16,
    extra: dict | None = None,
) -> dict:
    """Assemble a full metrics record for one fitted model."""
    rec = {
        "nfe": normalized_frob_error(W, What),
        "nfe_per_layer": per_matrix_frob_error(W, What).tolist(),
        "spectral_per_layer": spectral_error(W, What).tolist(),
        "bits": cost.bits(bits_per_param),
        "bits_per_param": bits_per_param,
    }
    rec.update(cost.as_dict())
    if extra:
        rec.update(extra)
    return rec
