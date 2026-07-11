"""Multiple-generator short-word model (the central hypothesis).

    W_l ~= B + sum_{t=1}^{s} c_lt * w_lt(G_1,...,G_g) + R_l

where each w is a *noncommutative word* of length <= L over the shared generators, e.g.
G_1, G_2 G_1, G_1 G_2 G_1. The word-atoms are generated from only g shared d x d
generators, so the shared budget is g*d^2 regardless of how many distinct words exist --
this is what could let words beat a plain q-atom dictionary at a matched budget.

Fit (alternating):
  1. build the atom set {w(G)} for all words of length <= L;
  2. sparse-code M_l = W_l - B over the atoms (OMP, <= s nonzeros);
  3. gradient step on B, G_1..G_g through the selected word products (torch);
  4. R_l = best-rank-r of the leftover.
Generators are parameterized near identity, G_j = I + eps H_j, for stability.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import linalg
from .metrics import ParamCost, lowrank_params


def enumerate_words(g: int, max_len: int, include_identity: bool = False):
    """All words (tuples of generator indices) of length in [lo..max_len]."""
    words = []
    if include_identity:
        words.append(())
    for L in range(1, max_len + 1):
        words.extend(itertools.product(range(g), repeat=L))
    return words


def _atoms_from_generators(Gs, words):
    """Materialize each word as a matrix product of the generators (torch)."""
    import torch
    d = Gs[0].shape[0]
    atoms = []
    for w in words:
        if len(w) == 0:
            atoms.append(torch.eye(d, dtype=Gs[0].dtype))
            continue
        M = Gs[w[0]]
        for idx in w[1:]:
            M = M @ Gs[idx]
        atoms.append(M)
    return atoms


def _omp_codes(target, atom_mat, s):
    """target:(k,P), atom_mat:(P,W). Codes (k,W) with <= s nonzeros/row."""
    from sklearn.linear_model import orthogonal_mp
    C = orthogonal_mp(atom_mat, target.T, n_nonzero_coefs=min(s, atom_mat.shape[1])).T
    return np.asarray(C)


def fit_words(
    W: np.ndarray, g: int = 2, max_len: int = 3, s: int = 2, r: int = 0,
    eps: float = 0.1, steps: int = 120, lr: float = 2e-2, reselect_every: int = 15,
    spec_penalty: float = 1e-3, seed: int = 0, include_identity: bool = False,
) -> dict:
    import torch

    torch.manual_seed(seed)
    k, d, _ = W.shape
    P = d * d
    Wt = torch.tensor(W.astype(np.float64))
    words = enumerate_words(g, max_len, include_identity)
    n_words = len(words)

    B = Wt.mean(0).clone().requires_grad_(True)
    # generators near identity: G_j = I + eps H_j
    H = [(1e-2 * torch.randn(d, d, dtype=torch.float64)).requires_grad_(True)
         for _ in range(g)]
    eye = torch.eye(d, dtype=torch.float64)

    opt = torch.optim.Adam([B] + H, lr=lr)

    codes = np.zeros((k, n_words))
    for step in range(steps):
        Gs = [eye + eps * Hj for Hj in H]
        if step % reselect_every == 0:
            with torch.no_grad():
                atoms = _atoms_from_generators(Gs, words)
                atom_mat = torch.stack([a.reshape(P) for a in atoms], dim=1).numpy()  # (P,W)
                target = (Wt - B).reshape(k, P).numpy()
                codes = _omp_codes(target, atom_mat, s)
        opt.zero_grad()
        Gs = [eye + eps * Hj for Hj in H]
        atoms = _atoms_from_generators(Gs, words)
        atom_stack = torch.stack([a.reshape(P) for a in atoms], dim=0)  # (W,P)
        recon = B.reshape(P)[None] + torch.tensor(codes) @ atom_stack     # (k,P)
        loss = ((Wt.reshape(k, P) - recon) ** 2).sum()
        for Hj in H:
            loss = loss + spec_penalty * (eps * Hj).pow(2).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        Gs = [eye + eps * Hj for Hj in H]
        atoms = _atoms_from_generators(Gs, words)
        atom_mat = torch.stack([a.reshape(P) for a in atoms], dim=1).numpy()
        Bn = B.numpy()
        target = (W.reshape(k, P) - Bn.reshape(P)[None]).astype(np.float64)
        codes = _omp_codes(target, atom_mat, s)
        recon_flat = codes @ atom_mat.T  # (k,P)
        What = np.empty_like(W, dtype=np.float64)
        for i in range(k):
            comp = Bn + recon_flat[i].reshape(d, d)
            R = linalg.best_rank_r(W[i].astype(np.float64) - comp, r)
            What[i] = comp + R

    shared = d * d + g * d * d            # B + g generators
    per_layer = s + lowrank_params(r, d)  # s coefficients (+ word indices, few bits)
    cost = ParamCost(shared=shared, per_layer=per_layer, k=k, d=d)
    index_bits = k * s * int(np.ceil(np.log2(max(n_words, 2))))
    return {
        "name": "word", "What": What.astype(W.dtype), "cost": cost,
        "g": g, "max_len": max_len, "s": s, "r": r, "n_words": n_words,
        "index_bits": index_bits,
        "generators": [ (eye + eps * Hj).detach().numpy() for Hj in H ],
        "B": Bn, "codes": codes,
    }
