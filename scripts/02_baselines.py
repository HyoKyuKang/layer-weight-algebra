"""Stage 02: fit baseline models and emit reconstruction curves.

Fits per group, per residual rank r:
    ind      independent low-rank
    shared   shared base + low-rank residual
    cluster  clustering (2 groups) + low-rank residual
plus the r-independent shared-mean point.

Usage:
    python scripts/02_baselines.py --model gpt2
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401
from lwa import baselines, config as cfg_mod, extract, plotting
from lwa.metrics import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    overrides = dict(cfg_mod.parse_override(s) for s in args.overrides)
    cfg = cfg_mod.load_config(args.config, overrides)
    model = args.model or cfg["model"]
    ranks = cfg["ranks"]
    bpp = 16

    data_dir = os.path.join(_bootstrap.ROOT, cfg["data_dir"], model.replace("/", "__"))
    out_dir = os.path.join(_bootstrap.ROOT, cfg["results_dir"], model.replace("/", "__"))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    groups = extract.load_groups(data_dir, roles=cfg.get("roles") or None)
    if not groups:
        raise SystemExit(f"No groups in {data_dir}; run 01_extract_weights.py first.")

    all_results = {}
    for role, g in groups.items():
        W = g.matrices
        records = []

        # r-independent shared mean
        fit = baselines.fit_shared_mean(W)
        rec = summarize(W, fit["What"], fit["cost"], bpp, {"name": "mean", "r": 0})
        records.append(rec)

        for r in ranks:
            for fitter, kw in [
                (baselines.fit_independent_lowrank, {"r": r}),
                (baselines.fit_shared_base_lowrank, {"r": r}),
                (baselines.fit_clustering_lowrank, {"r": r, "n_clusters": 2}),
            ]:
                fit = fitter(W, **kw)
                rec = summarize(W, fit["What"], fit["cost"], bpp,
                                {"name": fit["name"], "r": r})
                records.append(rec)

        all_results[role] = records

        # reconstruction curve (drop nfe_per_layer / spectral from plotting payload)
        plotting.reconstruction_curve(
            records, title=f"{model} / {role}",
            path=os.path.join(fig_dir, f"recon_{role.replace('.', '_')}.png"),
        )
        # error-by-depth at r=8 (or largest available <=8)
        r_show = max([r for r in ranks if r <= 8], default=ranks[0])
        depth = {}
        for name, fitter, kw in [
            ("ind", baselines.fit_independent_lowrank, {"r": r_show}),
            ("shared", baselines.fit_shared_base_lowrank, {"r": r_show}),
        ]:
            fit = fitter(W, **kw)
            from lwa.metrics import per_matrix_frob_error
            depth[f"{name} r={r_show}"] = per_matrix_frob_error(W, fit["What"]).tolist()
        plotting.error_by_depth(
            depth, title=f"{model} / {role}: error by depth",
            path=os.path.join(fig_dir, f"depth_{role.replace('.', '_')}.png"),
        )

        print(f"[{role}] d={g.d} k={g.k}")
        _print_table(records, ranks)

    with open(os.path.join(out_dir, "baselines.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[baselines] wrote {out_dir}/baselines.json and figures/")


def _print_table(records, ranks) -> None:
    header = f"    {'model':8s}" + "".join(f"  r={r:<3d}" for r in ranks)
    print(header)
    by_name = {}
    for rec in records:
        by_name.setdefault(rec["name"], {})[rec["r"]] = rec["nfe"]
    for name in ("mean", "ind", "shared", "cluster"):
        if name not in by_name:
            continue
        cells = []
        for r in ranks:
            v = by_name[name].get(r)
            cells.append(f"  {v:6.4f}" if v is not None else "  ------")
        print(f"    {name:8s}" + "".join(cells))


if __name__ == "__main__":
    main()
