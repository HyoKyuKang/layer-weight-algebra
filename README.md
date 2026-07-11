# Layer Weight Algebra

Investigating whether pretrained Transformer weight matrices across layers contain
**compressible shared algebraic structure**.

## Central hypothesis

> Can same-role weight matrices from different Transformer layers be approximated by a
> small shared algebraic system — shared matrices, matrix powers, or short *words* over a
> few matrix generators — with only low-rank layer-specific residuals?

For same-role matrices $W_1,\dots,W_k \in \mathbb{R}^{m\times n}$ we study

$$\widehat W_\ell = B + \sum_{t=1}^{s} c_{\ell t}\, w_{\ell t}(G_1,\dots,G_g) + U_\ell V_\ell^\top$$

- $B$: shared full-rank base matrix
- $G_1,\dots,G_g$: shared matrix generators
- $w_{\ell t}$: short matrix word (e.g. $G_1^a G_2^b$, $G_2 G_1$, $G_1 G_2 G_1$)
- $c_{\ell t}$: scalar coefficient
- $U_\ell V_\ell^\top$: rank-$r$ layer-specific residual

Initially restricted to **square, same-shape, same-role** matrices. $W_Q, W_K, W_V, W_O$ and
distinct MLP projections are **never** mixed into one group.

## Objectives compared (at matched storage budgets)

| Key | Model | Formula |
|-----|-------|---------|
| `ind`    | Independent low-rank            | $\sum_\ell \min_{\mathrm{rank}(R_\ell)\le r}\|W_\ell-R_\ell\|_F^2$ |
| `shared` | Shared base + low-rank residual | $\min_B \sum_\ell \min_{R_\ell}\|W_\ell-B-R_\ell\|_F^2$ |
| `dict`   | Shared matrix dictionary        | $\sum_\ell\|W_\ell-\sum_j c_{\ell j}D_j-R_\ell\|_F^2$, $\le s$ nonzeros/layer |
| `power`  | Single-generator powers         | $\sum_\ell\|W_\ell-A^{a_\ell}-R_\ell\|_F^2$ (and $B+\alpha_\ell A^{a_\ell}+R_\ell$) |
| `word`   | Two-generator short words       | $\sum_\ell\|W_\ell-B-\sum_t c_{\ell t} w_{\ell t}(G_1,G_2)-R_\ell\|_F^2$ |

Starting regime: $g=2$, $L\le 3$, $s\in\{1,2\}$, $r\in\{0,4,8,16,32\}$.

## Success criterion (important)

A generator model is **not** a success merely because it beats raw weight sharing.
It must beat an **unconstrained dictionary / shared-basis model under the same storage budget**,
and improvements must survive **activation-aware** metrics — not just Frobenius weight error.

A well-supported **negative result is an acceptable outcome**.

## Repo layout

```
config/            experiment configs (yaml)
src/lwa/           library
  extract.py       weight extraction + same-role grouping
  metrics.py       error metrics, bit/FLOP accounting
  baselines.py     ind, shared (+ mean, clustering)
  dictionary.py    shared matrix dictionary
  powers.py        single-generator powers
  words.py         two-generator short words
  diagnostics.py   SV decay, principal angles, commutators, stable rank
  nulls.py         null models (permutation, spectrum-matched randoms)
  activations.py   activation-aware metrics + calibration
  plotting.py      reconstruction / rate-distortion / SV plots
scripts/           runnable pipeline stages (01_..., 02_..., ...)
data/              extracted weights (gitignored)
results/           metrics + figures (gitignored)
docs/              literature map, notes, conclusions
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/01_extract_weights.py --model gpt2          # -> data/gpt2/*.npz
python scripts/02_baselines.py       --model gpt2          # -> results/gpt2/baselines.*
python scripts/03_diagnostics.py     --model gpt2
```

## Status

See [docs/PROGRESS.md](docs/PROGRESS.md) for the running log and
[docs/literature_map.md](docs/literature_map.md) for related work.
