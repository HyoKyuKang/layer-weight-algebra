"""Stage 07: the airtight matched-budget test of the generator-word restriction.

For a single group (default attn.q), lay every structured model on one axis: number of
shared full d*d matrices stored. The question the whole project turns on:

    at the SAME shared-matrix budget, does a generator-word model beat an unconstrained
    dictionary?

We include the *fair* (non-near-identity) word variant so a word failure cannot be
blamed on the I+eps H handicap.

Usage:
    python scripts/07_word_fairness.py --model gpt2 --role attn.q
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401
from lwa import config as cfg_mod, dictionary, extract, powers, words
from lwa.metrics import normalized_frob_error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--role", default="attn.q")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    overrides = dict(cfg_mod.parse_override(s) for s in args.overrides)
    cfg = cfg_mod.load_config(args.config, overrides)
    model = args.model or cfg["model"]
    data_dir = os.path.join(_bootstrap.ROOT, cfg["data_dir"], model.replace("/", "__"))
    out_dir = os.path.join(_bootstrap.ROOT, cfg["results_dir"], model.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)

    groups = extract.load_groups(data_dir, roles=[args.role])
    g = groups[args.role]
    W = g.matrices
    d = g.d
    rows = []

    def shared_matrices(rec):
        return rec["cost"].shared / (d * d)

    def add(label, fit):
        nfe = normalized_frob_error(W, fit["What"])
        sm = fit["cost"].shared / (d * d)
        rows.append({"label": label, "shared_matrices": round(sm, 2),
                     "total_params": fit["cost"].total, "nfe": round(nfe, 4),
                     "r": fit.get("r", 0)})
        print(f"  {label:22s} shared={sm:4.1f} mats  params={fit['cost'].total:>8d} "
              f"nfe={nfe:.4f}")

    for r in (0, 8):
        print(f"\n=== {model} / {args.role}  r={r} ===")
        for q in (2, 3, 4, 6):
            add(f"dict q{q}s2 r{r}", dictionary.fit_dictionary(W, q=q, s=2, r=r))
        add(f"word nearId s2 r{r}",
            words.fit_words(W, g=2, max_len=3, s=2, r=r, near_identity=True,
                            eps=0.1, steps=100))
        add(f"word FREE s2 r{r}",
            words.fit_words(W, g=2, max_len=3, s=2, r=r, near_identity=False,
                            steps=140, lr=1e-2))
        add(f"power_base r{r}",
            powers.fit_power(W, max_exp=6, with_base=True, r=r, steps=120))

    out = {"model": model, "role": args.role, "d": d, "k": g.k, "rows": rows}
    with open(os.path.join(out_dir, f"fairness_{args.role.replace('.', '_')}.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[fairness] wrote fairness_{args.role.replace('.', '_')}.json")
    print("\nInterpretation: compare each word row against the dict row at the SAME "
          "shared-matrix count. Word must win there to matter.")


if __name__ == "__main__":
    main()
