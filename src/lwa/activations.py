"""Activation-aware metrics.

Weight-space error alone is insufficient: a large ``||W_l - What_l||_F`` may act on
directions the model never uses. With calibration inputs X_l (the actual inputs each
weight sees), we report

    ||(W_l - What_l) X_l||_F^2 / ||W_l X_l||_F^2

which weights the error by the input covariance. X_l is captured with forward hooks on
the *same-role* module (e.g. the input to c_attn feeds W_Q/W_K/W_V; the input to
c_proj feeds W_O). Operator convention: X_l is (d, N), N calibration tokens.
"""

from __future__ import annotations

import os

import numpy as np


# role -> (module suffix producing that role's input). GPT-2 specific for now.
_GPT2_ROLE_MODULE = {
    "attn.q": "attn.c_attn",
    "attn.k": "attn.c_attn",
    "attn.v": "attn.c_attn",
    "attn.o": "attn.c_proj",
}


def collect_activations(
    model_id: str, roles: list[str], cache_dir: str | None = None,
    dataset: str = "wikitext", dataset_config: str = "wikitext-2-raw-v1",
    n_samples: int = 64, seq_len: int = 256, n_tokens: int = 4096, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Return {role: X (k, d, N)} of calibration inputs per layer (subsampled tokens)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, dtype=torch.float32
    )
    model.eval()

    # module-name resolution (handles GPT-2 base/LM prefixes)
    name2mod = dict(model.named_modules())

    def find_layer_module(layer: int, suffix: str):
        for cand in (f"transformer.h.{layer}.{suffix}", f"h.{layer}.{suffix}"):
            if cand in name2mod:
                return name2mod[cand]
        raise KeyError(f"module {suffix} for layer {layer} not found")

    n_layer = model.config.num_hidden_layers
    unique_suffixes = sorted({_GPT2_ROLE_MODULE[r] for r in roles})

    captured: dict[tuple[int, str], list] = {}
    handles = []

    def make_hook(layer, suffix):
        def hook(mod, inp, out):
            x = inp[0].detach()               # (batch, seq, d)
            captured.setdefault((layer, suffix), []).append(
                x.reshape(-1, x.shape[-1]).to(torch.float32))
        return hook

    for i in range(n_layer):
        for suffix in unique_suffixes:
            h = find_layer_module(i, suffix).register_forward_hook(make_hook(i, suffix))
            handles.append(h)

    # calibration corpus (datasets>=5 needs canonical namespace/name repo ids)
    candidates = [dataset]
    if "/" not in dataset:
        candidates += [f"Salesforce/{dataset}", f"mirror/{dataset}"]
    ds = None
    last_err = None
    for cand in candidates:
        try:
            ds = load_dataset(cand, dataset_config, split="train")
            break
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
    if ds is None:
        raise RuntimeError(f"could not load calibration dataset {dataset!r}: {last_err}")
    rng = np.random.default_rng(seed)
    texts = [t for t in ds["text"] if t and len(t) > 64]
    idx = rng.choice(len(texts), size=min(n_samples, len(texts)), replace=False)

    with torch.no_grad():
        for j in idx:
            enc = tok(texts[int(j)], return_tensors="pt", truncation=True,
                      max_length=seq_len)
            model(**enc)

    for h in handles:
        h.remove()

    # assemble per role, subsample tokens
    out: dict[str, np.ndarray] = {}
    for role in roles:
        suffix = _GPT2_ROLE_MODULE[role]
        d = model.config.hidden_size
        per_layer = []
        for i in range(n_layer):
            X = np.concatenate([t.numpy() for t in captured[(i, suffix)]], axis=0)  # (T, d)
            if X.shape[0] > n_tokens:
                sel = rng.choice(X.shape[0], size=n_tokens, replace=False)
                X = X[sel]
            per_layer.append(X.T)  # (d, N)
        # trim to common N
        N = min(x.shape[1] for x in per_layer)
        out[role] = np.stack([x[:, :N] for x in per_layer], axis=0)  # (k, d, N)
    return out


def activation_error(W: np.ndarray, What: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-layer ||(W-What) X||_F^2 / ||W X||_F^2."""
    k = W.shape[0]
    out = np.empty(k)
    for i in range(k):
        WX = W[i].astype(np.float64) @ X[i].astype(np.float64)
        EX = (W[i] - What[i]).astype(np.float64) @ X[i].astype(np.float64)
        den = float((WX ** 2).sum())
        out[i] = float((EX ** 2).sum()) / den if den > 0 else 0.0
    return out


def output_cosine(W: np.ndarray, What: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-layer mean cosine similarity between W x and What x over calibration tokens."""
    k = W.shape[0]
    out = np.empty(k)
    for i in range(k):
        WX = W[i].astype(np.float64) @ X[i].astype(np.float64)      # (d, N)
        HX = What[i].astype(np.float64) @ X[i].astype(np.float64)
        num = (WX * HX).sum(axis=0)
        den = np.linalg.norm(WX, axis=0) * np.linalg.norm(HX, axis=0)
        good = den > 0
        out[i] = float((num[good] / den[good]).mean()) if good.any() else 0.0
    return out


def save_activations(acts: dict[str, np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for role, X in acts.items():
        np.savez_compressed(
            os.path.join(out_dir, f"act_{role.replace('.', '_')}.npz"), X=X)


def load_activations(in_dir: str, roles: list[str] | None = None) -> dict[str, np.ndarray]:
    out = {}
    if not os.path.isdir(in_dir):
        return out
    for fname in sorted(os.listdir(in_dir)):
        if not fname.startswith("act_") or not fname.endswith(".npz"):
            continue
        role = fname[4:-4].replace("_", ".", 1)
        if roles and role not in roles:
            continue
        out[role] = np.load(os.path.join(in_dir, fname))["X"]
    return out
