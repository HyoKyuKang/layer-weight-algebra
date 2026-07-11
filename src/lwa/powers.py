"""Single-generator power model.

    W_l ~= B + alpha_l * A^{a_l} + R_l          (with_base=True)
    W_l ~= A^{a_l} + R_l                         (with_base=False)

A is a shared d x d generator; a_l is a small (possibly negative) integer exponent;
alpha_l a scalar; R_l a rank-r residual. This tests whether same-role layers lie on a
*one-parameter multiplicative orbit* of a single matrix.

Fit (coordinate descent):
  * hold A: for each layer pick the exponent a in [-max_exp..max_exp] and closed-form
    alpha that best explain M_l = W_l - B, then set R_l = best-rank-r of the leftover;
  * hold exponents/alphas: gradient step on A (and B) in float64 torch.
A is kept invertible/bounded by a light spectral penalty; negative exponents use A^{-1}.
"""

from __future__ import annotations

import numpy as np

from . import linalg
from .metrics import ParamCost, lowrank_params


def _powers(A, max_exp):
    """Dict exp -> A^exp for exp in [-max_exp..max_exp] (torch tensors)."""
    import torch
    d = A.shape[0]
    out = {0: torch.eye(d, dtype=A.dtype)}
    P = torch.eye(d, dtype=A.dtype)
    for e in range(1, max_exp + 1):
        P = P @ A
        out[e] = P
    Ainv = torch.linalg.inv(A)
    P = torch.eye(d, dtype=A.dtype)
    for e in range(1, max_exp + 1):
        P = P @ Ainv
        out[-e] = P
    return out


def _select_exponents(M, powers_np, max_exp):
    """For each layer, choose (a, alpha) minimizing ||M_l - alpha A^a||_F. Closed form."""
    k = M.shape[0]
    a_sel = np.zeros(k, dtype=int)
    alpha = np.zeros(k)
    for i in range(k):
        best = np.inf
        Mi = M[i]
        for e, Pe in powers_np.items():
            denom = float((Pe * Pe).sum())
            if denom <= 1e-12:
                continue
            c = float((Mi * Pe).sum()) / denom
            resid = float(((Mi - c * Pe) ** 2).sum())
            if resid < best:
                best, a_sel[i], alpha[i] = resid, e, c
    return a_sel, alpha


def fit_power(
    W: np.ndarray, max_exp: int = 6, with_base: bool = True, r: int = 0,
    steps: int = 150, lr: float = 5e-2, reselect_every: int = 15,
    spec_penalty: float = 1e-3, seed: int = 0,
) -> dict:
    import torch

    torch.manual_seed(seed)
    k, d, _ = W.shape
    Wt = torch.tensor(W.astype(np.float64))
    # Base is FIXED at the layerwise mean (a valid shared base). Jointly training B with a
    # near-singular A destabilizes: A^a can blow up, dragging B far from the data so the
    # error exceeds even ||W-mean|| (an unfair artifact, not a real failure of the orbit
    # hypothesis). Fixing B guarantees error <= ||W - mean|| and gives the model its fair
    # best. Only the shared generator A is optimized.
    B = Wt.mean(0).clone() if with_base else torch.zeros(d, d, dtype=torch.float64)
    # init A near identity (stable, invertible)
    A = (torch.eye(d, dtype=torch.float64)
         + 1e-2 * torch.randn(d, d, dtype=torch.float64)).requires_grad_(True)

    opt = torch.optim.Adam([A], lr=lr)

    a_sel = np.zeros(k, dtype=int)
    alpha = np.ones(k)
    for step in range(steps):
        if step % reselect_every == 0:
            with torch.no_grad():
                pw = _powers(A, max_exp)
                pw_np = {e: P.numpy() for e, P in pw.items()}
                M = (Wt - B).numpy()
                a_sel, alpha = _select_exponents(M, pw_np, max_exp)
        opt.zero_grad()
        pw = _powers(A, max_exp)
        recon = torch.stack([B + float(alpha[i]) * pw[int(a_sel[i])] for i in range(k)])
        loss = ((Wt - recon) ** 2).sum()
        # keep A bounded (spectral norm near 1)
        loss = loss + spec_penalty * (torch.linalg.matrix_norm(A, 2) - 1.0) ** 2
        loss.backward()
        opt.step()

    with torch.no_grad():
        pw = _powers(A, max_exp)
        pw_np = {e: P.numpy() for e, P in pw.items()}
        Bn = B.numpy()
        M = W.astype(np.float64) - Bn
        a_sel, alpha = _select_exponents(M, pw_np, max_exp)
        What = np.empty_like(W, dtype=np.float64)
        for i in range(k):
            comp = Bn + alpha[i] * pw_np[int(a_sel[i])]
            R = linalg.best_rank_r(W[i].astype(np.float64) - comp, r)
            What[i] = comp + R

    shared = d * d + (d * d if with_base else 0)  # A (+ B)
    per_layer = 1 + lowrank_params(r, d)          # alpha_l (+ exponent, few bits)
    cost = ParamCost(shared=shared, per_layer=per_layer, k=k, d=d)
    return {
        "name": "power" + ("_base" if with_base else ""),
        "What": What.astype(W.dtype), "cost": cost,
        "A": A.detach().numpy(), "B": Bn if with_base else None,
        "exponents": a_sel.tolist(), "alpha": alpha.tolist(),
        "max_exp": max_exp, "r": r,
    }
