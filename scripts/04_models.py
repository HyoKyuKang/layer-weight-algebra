"""Stage 04: fit the algebraic models (dict / power / word) and compare, at matched
budget, against the strongest baselines.

Emits results/<model>/models.json and a combined rate-distortion figure per group.

Usage:
    python scripts/04_models.py --model gpt2
    python scripts/04_models.py --model gpt2 --set roles=["attn.q"]
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401
from lwa import baselines, config as cfg_mod, dictionary, extract, plotting, powers, words
from lwa.metrics import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--ranks", default=None, help="comma list overriding cfg ranks")
    args = ap.parse_args()

    overrides = dict(cfg_mod.parse_override(s) for s in args.overrides)
    cfg = cfg_mod.load_config(args.config, overrides)
    model = args.model or cfg["model"]
    ranks = ([int(x) for x in args.ranks.split(",")] if args.ranks else cfg["ranks"])
    bpp = 16

    data_dir = os.path.join(_bootstrap.ROOT, cfg["data_dir"], model.replace("/", "__"))
    out_dir = os.path.join(_bootstrap.ROOT, cfg["results_dir"], model.replace("/", "__"))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    groups = extract.load_groups(data_dir, roles=cfg.get("roles") or None)
    if not groups:
        raise SystemExit(f"No groups in {data_dir}; run 01 first.")

    all_results = {}
    for role, g in groups.items():
        W = g.matrices
        records = []
        print(f"\n=== {role}  (d={g.d}, k={g.k}) ===")

        def add(fit, meta):
            rec = summarize(W, fit["What"], fit["cost"], bpp, meta)
            for key in ("index_bits", "g", "s", "q", "max_len", "max_exp", "n_words"):
                if key in fit:
                    rec[key] = fit[key]
            records.append(rec)
            print(f"  {meta['name']:12s} r={meta.get('r',0):<3d} "
                  f"nfe={rec['nfe']:.4f}  params={rec['total_params']:>8d}  "
                  f"cr={rec['compression_ratio']:.2f}x")
            return rec

        for r in ranks:
            add(baselines.fit_independent_lowrank(W, r), {"name": "ind", "r": r})
            add(baselines.fit_shared_base_lowrank(W, r), {"name": "shared", "r": r})
            # dictionary: match generator shared budget region (q in {2,4})
            for q in cfg["dict"]["q"]:
                for s in cfg["dict"]["s"]:
                    add(dictionary.fit_dictionary(W, q=q, s=s, r=r),
                        {"name": f"dict_q{q}s{s}", "r": r})
            # powers
            add(powers.fit_power(W, max_exp=cfg["power"]["max_exp"],
                                 with_base=True, r=r, steps=120),
                {"name": "power_base", "r": r})
            # words (g=2, L<=3)
            for s in cfg["word"]["s"]:
                add(words.fit_words(W, g=cfg["word"]["g"], max_len=cfg["word"]["max_len"],
                                    s=s, r=r, eps=0.1, steps=100),
                    {"name": f"word_s{s}", "r": r})

        all_results[role] = records
        plotting.reconstruction_curve(
            records, title=f"{model} / {role}: rate-distortion",
            path=os.path.join(fig_dir, f"rd_{role.replace('.', '_')}.png"),
        )

    with open(os.path.join(out_dir, "models.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[models] wrote {out_dir}/models.json and figures/rd_*.png")


if __name__ == "__main__":
    main()
