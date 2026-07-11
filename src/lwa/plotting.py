"""Matplotlib figures: reconstruction curves, residual spectra, rate-distortion."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def reconstruction_curve(records: list[dict], title: str, path: str) -> None:
    """NFE vs total stored params, one line per model family.

    records: [{'name','r','total_params','nfe'}, ...]
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    by_name: dict[str, list[dict]] = {}
    for rec in records:
        by_name.setdefault(rec["name"], []).append(rec)
    for name, recs in sorted(by_name.items()):
        recs = sorted(recs, key=lambda x: x["total_params"])
        xs = [r["total_params"] for r in recs]
        ys = [r["nfe"] for r in recs]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("total stored params")
    ax.set_ylabel("normalized Frobenius error")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, path)


def residual_spectra(spectra: np.ndarray, title: str, path: str) -> None:
    """Overlay singular-value curves (one per layer), normalized to top SV."""
    fig, ax = plt.subplots(figsize=(6, 4))
    k = spectra.shape[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, k))
    for i in range(k):
        s = spectra[i]
        s = s / s[0] if s[0] > 0 else s
        ax.plot(np.arange(1, s.size + 1), s, color=cmap[i], alpha=0.7, lw=1)
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("normalized singular value")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, k - 1))
    fig.colorbar(sm, ax=ax, label="layer index")
    _save(fig, path)


def heatmap(M: np.ndarray, title: str, path: str, cbar_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(M, cmap="magma", aspect="equal")
    ax.set_xlabel("layer j")
    ax.set_ylabel("layer i")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    _save(fig, path)


def error_by_depth(nfe_per_layer: dict[str, list[float]], title: str, path: str) -> None:
    """nfe_per_layer: {model_name: [nfe_layer0, ...]}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in sorted(nfe_per_layer.items()):
        ax.plot(np.arange(len(ys)), ys, marker="o", label=name)
    ax.set_xlabel("layer index (depth)")
    ax.set_ylabel("normalized Frobenius error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)
