"""Stage 01: extract same-role square weight groups and cache them to disk.

Usage:
    python scripts/01_extract_weights.py --model gpt2
    python scripts/01_extract_weights.py --model gpt2-medium --set roles=["attn.q","attn.o"]
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401  (sets sys.path)
from lwa import config as cfg_mod
from lwa import extract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted override key=value (repeatable)")
    args = ap.parse_args()

    overrides = dict(cfg_mod.parse_override(s) for s in args.overrides)
    cfg = cfg_mod.load_config(args.config, overrides)
    model = args.model or cfg["model"]
    roles = cfg.get("roles") or None

    out_dir = os.path.join(_bootstrap.ROOT, cfg["data_dir"], model.replace("/", "__"))
    cache_dir = os.path.join(_bootstrap.ROOT, cfg["cache_dir"])

    print(f"[extract] model={model}  -> {out_dir}")
    groups = extract.extract_groups(
        model, cache_dir=cache_dir, dtype=cfg["dtype"], roles=roles
    )
    extract.save_groups(groups, out_dir)

    summary = {
        role: {"k": g.k, "d": g.d, "layers": g.layers, "module_type": g.module_type}
        for role, g in groups.items()
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"model": model, "groups": summary}, f, indent=2)

    for role, g in groups.items():
        print(f"  {role:10s}  k={g.k:3d}  d={g.d:5d}  "
              f"(stack {g.matrices.shape}, {g.matrices.dtype})")
    print(f"[extract] wrote {len(groups)} groups + manifest.json")


if __name__ == "__main__":
    main()
