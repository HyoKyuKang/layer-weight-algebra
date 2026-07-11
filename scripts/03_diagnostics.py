"""Stage 03: cross-layer heterogeneity diagnostics + figures.

Usage:
    python scripts/03_diagnostics.py --model gpt2
"""

import argparse
import json
import os

import numpy as np

import _bootstrap  # noqa: F401
from lwa import config as cfg_mod, diagnostics, extract, plotting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    overrides = dict(cfg_mod.parse_override(s) for s in args.overrides)
    cfg = cfg_mod.load_config(args.config, overrides)
    model = args.model or cfg["model"]

    data_dir = os.path.join(_bootstrap.ROOT, cfg["data_dir"], model.replace("/", "__"))
    out_dir = os.path.join(_bootstrap.ROOT, cfg["results_dir"], model.replace("/", "__"))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    groups = extract.load_groups(data_dir, roles=cfg.get("roles") or None)
    if not groups:
        raise SystemExit(f"No groups in {data_dir}; run 01_extract_weights.py first.")

    summary = {}
    for role, g in groups.items():
        W = g.matrices
        s = diagnostics.summarize_group(W, rank_for_angles=16)
        summary[role] = s
        tag = role.replace(".", "_")

        # centered residual spectra W_l - B
        csp = diagnostics.centered_spectra(W)
        plotting.residual_spectra(
            csp, title=f"{model} / {role}: spectra of W_l - mean",
            path=os.path.join(fig_dir, f"spectra_centered_{tag}.png"),
        )
        # raw spectra
        plotting.residual_spectra(
            diagnostics.spectra(W), title=f"{model} / {role}: spectra of W_l",
            path=os.path.join(fig_dir, f"spectra_raw_{tag}.png"),
        )
        # pairwise diff + commutator heatmaps
        plotting.heatmap(
            diagnostics.pairwise_diff_norm(W),
            title=f"{model} / {role}: ||W_i-W_j||/sqrt(...)",
            path=os.path.join(fig_dir, f"pairdiff_{tag}.png"),
            cbar_label="normalized diff",
        )
        plotting.heatmap(
            diagnostics.commutator_matrix(W),
            title=f"{model} / {role}: normalized commutator",
            path=os.path.join(fig_dir, f"commutator_{tag}.png"),
            cbar_label="||[Wi,Wj]||/(||Wi|| ||Wj||)",
        )

        print(f"[{role}] " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in s.items()))

    with open(os.path.join(out_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[diagnostics] wrote {out_dir}/diagnostics.json and figures/")


if __name__ == "__main__":
    main()
