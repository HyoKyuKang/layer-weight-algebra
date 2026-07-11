"""Stage 05: capture calibration activations and re-score models activation-aware.

Usage:
    python scripts/05_activations.py --model gpt2
"""

import argparse
import json
import os

import numpy as np

import _bootstrap  # noqa: F401
from lwa import activations, baselines, config as cfg_mod, dictionary, extract, words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    overrides = dict(cfg_mod.parse_override(s) for s in args.overrides)
    cfg = cfg_mod.load_config(args.config, overrides)
    model = args.model or cfg["model"]
    acfg = cfg["activation"]

    data_dir = os.path.join(_bootstrap.ROOT, cfg["data_dir"], model.replace("/", "__"))
    act_dir = os.path.join(data_dir, "activations")
    out_dir = os.path.join(_bootstrap.ROOT, cfg["results_dir"], model.replace("/", "__"))
    cache_dir = os.path.join(_bootstrap.ROOT, cfg["cache_dir"])
    os.makedirs(out_dir, exist_ok=True)

    groups = extract.load_groups(data_dir, roles=cfg.get("roles") or None)
    roles = list(groups.keys())

    acts = activations.load_activations(act_dir, roles)
    if set(acts) != set(roles):
        print("[act] collecting calibration activations ...", flush=True)
        # default: network-free builtin corpus. Set activation.use_dataset: true to pull
        # the HF dataset instead (needs network / cached files).
        use_ds = acfg.get("use_dataset", False)
        acts = activations.collect_activations(
            model, roles, cache_dir=cache_dir,
            dataset=(acfg["dataset"] if use_ds else None),
            dataset_config=acfg["dataset_config"],
            n_samples=acfg["n_samples"], seq_len=acfg["seq_len"])
        activations.save_activations(acts, act_dir)
        print(f"[act] captured activations for {len(acts)} roles", flush=True)

    r = max([x for x in cfg["ranks"] if x <= 16], default=cfg["ranks"][0])
    report = {}
    for role, g in groups.items():
        W, X = g.matrices, acts[role]
        fits = {
            "ind": baselines.fit_independent_lowrank(W, r),
            "shared": baselines.fit_shared_base_lowrank(W, r),
            "dict_q4s2": dictionary.fit_dictionary(W, q=4, s=2, r=r),
            "word_s2": words.fit_words(W, g=2, max_len=3, s=2, r=r, steps=100),
        }
        rec = {}
        for name, fit in fits.items():
            ae = activations.activation_error(W, fit["What"], X)
            cos = activations.output_cosine(W, fit["What"], X)
            from lwa.metrics import normalized_frob_error
            rec[name] = {
                "weight_nfe": normalized_frob_error(W, fit["What"]),
                "act_error_mean": float(np.mean(ae)),
                "act_error_per_layer": ae.tolist(),
                "output_cosine_mean": float(np.mean(cos)),
            }
        report[role] = {"rank": r, "models": rec}
        print(f"[{role}] r={r}")
        for name, m in rec.items():
            print(f"  {name:10s} wnfe={m['weight_nfe']:.3f} "
                  f"act_err={m['act_error_mean']:.3f} cos={m['output_cosine_mean']:.3f}")

    with open(os.path.join(out_dir, "activation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[act] wrote {out_dir}/activation_report.json")


if __name__ == "__main__":
    main()
