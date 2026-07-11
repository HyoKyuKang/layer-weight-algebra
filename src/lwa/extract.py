"""Weight extraction and same-role grouping.

We collect *same-role, same-shape, square* weight matrices across the layers of a
pretrained Transformer and group them so each group is a stack

    W  of shape  (k, d, d)     # k layers, each a d x d operator

All matrices are normalized to the **operator convention** ``y = M @ x`` (shape
``(out, in)``), regardless of whether the source module stored them as ``nn.Linear``
(weight is already ``(out, in)``) or as a GPT-2 ``Conv1D`` (weight is ``(in, out)``,
so we transpose). Fused QKV projections are split into their per-role blocks.

We deliberately keep W_Q, W_K, W_V, W_O and the two MLP projections in *separate*
groups and never mix them.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Group container
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class WeightGroup:
    """A stack of same-role square matrices across layers."""

    role: str                 # e.g. "attn.q"
    matrices: np.ndarray      # (k, d, d), operator convention (out, in)
    layers: list[int]         # source layer index for each matrix
    module_type: str          # "attn" | "mlp"
    model: str                # source model id
    d: int                    # dimension (square)

    @property
    def k(self) -> int:
        return self.matrices.shape[0]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"WeightGroup(role={self.role!r}, k={self.k}, d={self.d}, "
            f"model={self.model!r})"
        )


# ---------------------------------------------------------------------------
# Architecture-specific extraction
# ---------------------------------------------------------------------------
# Each extractor returns dict[role -> list[(layer_idx, matrix_out_in)]].
#
# matrix_out_in is a numpy array (out, in) in operator convention.


def _get(state, *candidates: str) -> np.ndarray:
    """Fetch a param by name, tolerating an optional ``transformer.``/``model.`` prefix
    (``AutoModel`` drops it; ``AutoModelForCausalLM`` keeps it)."""
    for name in candidates:
        for key in (name, f"transformer.{name}", f"model.{name}"):
            if key in state:
                return np.asarray(state[key])
    raise KeyError(f"none of {candidates} (with/without prefix) in state dict")


def _extract_gpt2(state, config) -> dict[str, list[tuple[int, np.ndarray]]]:
    """GPT-2 family (gpt2, gpt2-medium/large/xl, distilgpt2).

    Modules are ``Conv1D`` (weight stored as ``(in, out)``). c_attn fuses Q,K,V
    along the output dim in contiguous [Q|K|V] blocks.
    """
    d = config.n_embd
    n_layer = config.n_layer
    groups: dict[str, list[tuple[int, np.ndarray]]] = {
        "attn.q": [], "attn.k": [], "attn.v": [], "attn.o": [],
    }
    for i in range(n_layer):
        c_attn = _get(state, f"h.{i}.attn.c_attn.weight")  # (d, 3d)
        assert c_attn.shape == (d, 3 * d), c_attn.shape
        wq, wk, wv = c_attn[:, 0:d], c_attn[:, d:2 * d], c_attn[:, 2 * d:3 * d]
        # Conv1D: y = x @ W  ->  operator (out,in) is W.T
        groups["attn.q"].append((i, wq.T.copy()))
        groups["attn.k"].append((i, wk.T.copy()))
        groups["attn.v"].append((i, wv.T.copy()))

        c_proj = _get(state, f"h.{i}.attn.c_proj.weight")  # (d, d)
        assert c_proj.shape == (d, d), c_proj.shape
        groups["attn.o"].append((i, c_proj.T.copy()))
    return groups


def _extract_gpt_neox(state, config) -> dict[str, list[tuple[int, np.ndarray]]]:
    """GPT-NeoX / Pythia family.

    query_key_value is ``nn.Linear`` with weight ``(3*hidden, hidden)`` but the rows
    are grouped **per head**: reshaped as (num_heads, 3*head_size, hidden). We
    de-interleave to recover full Q, K, V operators of shape (hidden, hidden).
    """
    h = config.hidden_size
    n_layer = config.num_hidden_layers
    n_head = config.num_attention_heads
    head = h // n_head
    groups: dict[str, list[tuple[int, np.ndarray]]] = {
        "attn.q": [], "attn.k": [], "attn.v": [], "attn.o": [],
    }
    for i in range(n_layer):
        qkv = _get(state, f"gpt_neox.layers.{i}.attention.query_key_value.weight",
                   f"layers.{i}.attention.query_key_value.weight")
        assert qkv.shape == (3 * h, h), qkv.shape
        # (n_head, 3*head, hidden) -> split the 3*head axis into q|k|v
        qkv = qkv.reshape(n_head, 3 * head, h)
        wq = qkv[:, 0:head, :].reshape(h, h)
        wk = qkv[:, head:2 * head, :].reshape(h, h)
        wv = qkv[:, 2 * head:3 * head, :].reshape(h, h)
        # nn.Linear weight is already (out, in) == operator convention
        groups["attn.q"].append((i, wq.copy()))
        groups["attn.k"].append((i, wk.copy()))
        groups["attn.v"].append((i, wv.copy()))
        dense = _get(state, f"gpt_neox.layers.{i}.attention.dense.weight",
                     f"layers.{i}.attention.dense.weight")  # (h,h)
        assert dense.shape == (h, h), dense.shape
        groups["attn.o"].append((i, dense.copy()))
    return groups


def _extract_llama(state, config) -> dict[str, list[tuple[int, np.ndarray]]]:
    """LLaMA / Mistral / Qwen2 style: separate q/k/v/o projections.

    Only q_proj and o_proj are guaranteed square (k/v may be GQA-reduced). We keep
    every projection that is square and same-shape as hidden_size.
    """
    h = config.hidden_size
    n_layer = config.num_hidden_layers
    groups: dict[str, list[tuple[int, np.ndarray]]] = {}
    role_map = {
        "attn.q": "self_attn.q_proj",
        "attn.k": "self_attn.k_proj",
        "attn.v": "self_attn.v_proj",
        "attn.o": "self_attn.o_proj",
    }
    for role, sub in role_map.items():
        mats = []
        for i in range(n_layer):
            try:
                w = _get(state, f"layers.{i}.{sub}.weight")
            except KeyError:
                mats = []
                break
            if w.shape[0] == w.shape[1] == h:  # square only
                mats.append((i, w.copy()))
        if len(mats) == n_layer:
            groups[role] = mats
    return groups


_EXTRACTORS: dict[str, Callable] = {
    "gpt2": _extract_gpt2,
    "gpt_neox": _extract_gpt_neox,
    "llama": _extract_llama,
    "mistral": _extract_llama,
    "qwen2": _extract_llama,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_groups(
    model_id: str,
    cache_dir: str | None = None,
    dtype: str = "float32",
    roles: list[str] | None = None,
) -> dict[str, WeightGroup]:
    """Load ``model_id`` from HuggingFace and return same-role square weight groups."""
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    model_type = config.model_type
    if model_type not in _EXTRACTORS:
        raise NotImplementedError(
            f"No extractor for model_type={model_type!r}. "
            f"Supported: {sorted(_EXTRACTORS)}"
        )

    model = AutoModel.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.float32
    )
    model.eval()
    np_dtype = np.dtype(dtype)
    # Materialize state dict as float32 numpy (analysis dtype).
    state = {k: v.detach().to(torch.float32).cpu().numpy() for k, v in model.state_dict().items()}

    raw = _EXTRACTORS[model_type](state, config)

    groups: dict[str, WeightGroup] = {}
    for role, entries in raw.items():
        if not entries:
            continue
        if roles and role not in roles:
            continue
        layers = [i for i, _ in entries]
        mats = np.stack([m.astype(np_dtype) for _, m in entries], axis=0)
        d = mats.shape[1]
        if mats.shape[1] != mats.shape[2]:
            continue  # square only
        module_type = role.split(".")[0]
        groups[role] = WeightGroup(
            role=role, matrices=mats, layers=layers,
            module_type=module_type, model=model_id, d=d,
        )
    return groups


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_groups(groups: dict[str, WeightGroup], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for role, g in groups.items():
        fname = os.path.join(out_dir, f"{role.replace('.', '_')}.npz")
        np.savez_compressed(
            fname,
            matrices=g.matrices,
            layers=np.asarray(g.layers),
            role=g.role,
            module_type=g.module_type,
            model=g.model,
            d=g.d,
        )


def load_groups(in_dir: str, roles: list[str] | None = None) -> dict[str, WeightGroup]:
    groups: dict[str, WeightGroup] = {}
    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith(".npz"):
            continue
        role = fname[:-4].replace("_", ".", 1)
        if roles and role not in roles:
            continue
        z = np.load(os.path.join(in_dir, fname), allow_pickle=False)
        groups[str(z["role"])] = WeightGroup(
            role=str(z["role"]),
            matrices=z["matrices"],
            layers=z["layers"].tolist(),
            module_type=str(z["module_type"]),
            model=str(z["model"]),
            d=int(z["d"]),
        )
    return groups
