"""Stage 06: null-model controls.

Refits `shared` and `dict` on spectrum-matched / rotated / gaussian nulls and compares
the gain-over-independent-lowrank on real vs null groups. If the gain is the same, the
apparent cross-layer 'sharing' is a spectral artifact, not real structure.

Usage:
    python scripts/06_nulls.py --model gpt2
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401
from lwa import baselines, config as cfg_mod, dictionary, extract, nulls
from lwa.metrics import normalized_frob_error


def gain(W, r):
    """Improvement of shared/dict over independent low-rank (positive = structure)."""
    ind = normalized_frob_error(W, baselines.fit_independent_lowrank(W, r)["What"])
    sh = normalized_frob_error(W, baselines.fit_shared_base_lowrank(W, r)["What"])
    dc = normalized_frob_error(W, dictionary.fit_dictionary(W, q=4, s=2, r=r)["What"])
    return {"ind": ind, "shared": sh, "dict_q4s2": dc,
            "gain_shared": ind - sh, "gain_dict": ind - dc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    overrides = dict(cfg_mod.parse_override(s) for s in args.overrides)
    cfg = cfg_mod.load_config(args.config, overrides)
    model = args.model or cfg["model"]
    r = args.rank

    data_dir = os.path.join(_bootstrap.ROOT, cfg["data_dir"], model.replace("/", "__"))
    out_dir = os.path.join(_bootstrap.ROOT, cfg["results_dir"], model.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)

    groups = extract.load_groups(data_dir, roles=cfg.get("roles") or None)
    report = {}
    for role, g in groups.items():
        W = g.matrices
        entry = {"real": gain(W, r)}
        for nname, nfn in nulls.NULLS.items():
            entry[nname] = gain(nfn(W, seed=cfg["seed"]), r)
        report[role] = entry
        print(f"[{role}] r={r}")
        print(f"  {'variant':16s} gain_shared  gain_dict")
        for k, v in entry.items():
            print(f"  {k:16s} {v['gain_shared']:+.4f}     {v['gain_dict']:+.4f}")

    with open(os.path.join(out_dir, "nulls.json"), "w", encoding="utf-8") as f:
        json.dump({"rank": r, "groups": report}, f, indent=2)
    print(f"\n[nulls] wrote {out_dir}/nulls.json")


if __name__ == "__main__":
    main()
