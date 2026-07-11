"""Tiny config loader: YAML defaults + dotted CLI overrides."""

from __future__ import annotations

import ast
import copy
import os

import yaml

_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    path = path or _DEFAULT
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            _set_dotted(cfg, k, v)
    return cfg


def parse_override(s: str) -> tuple[str, object]:
    """Parse ``key=value`` where value is python-literal-ish (int, list, bool, str)."""
    key, _, raw = s.partition("=")
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        val = raw
    return key.strip(), val


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def merge(base: dict, **kw) -> dict:
    out = copy.deepcopy(base)
    for k, v in kw.items():
        if v is not None:
            out[k] = v
    return out
