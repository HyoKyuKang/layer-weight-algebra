# Results summary — GPT-2 (session 1)

Model: `gpt2` (124M). Groups: `attn.{q,k,v,o}`, each **k=12** layers of **d=768** square
matrices. Metric: normalized Frobenius error (NFE) = ‖W−Ŵ‖²/‖W‖², lower better.
Precision for bits: 16-bit. **All results in this file are weight-space only** (see the
activation-aware section for the functional test that can overturn them).

## 1. Baselines (NFE)

attn.q (representative; others within ±0.05):

| model | r=0 | r=4 | r=8 | r=16 | r=32 |
|-------|----:|----:|----:|-----:|-----:|
| independent low-rank | 1.000 | 0.941 | 0.899 | 0.831 | 0.724 |
| shared base + LR     | 0.917 | 0.862 | 0.823 | 0.759 | 0.659 |
| clustering(2) + LR   | 0.728 | 0.686 | 0.655 | 0.605 | 0.527 |

Take-aways:
- Even rank-32 per layer leaves **~70%** of energy unexplained → matrices are high-rank.
- Shared base beats independent LR by only ~0.06–0.08 NFE.
- Clustering(2) beats a single shared base a lot → suggests **1–2 outlier layers**
  (likely layer 0), not smooth shared structure.

## 2. Diagnostics

| role | energy in mean | stable rank | pairwise diff (√2≈1.414) | commutator (√2/√d≈0.051) |
|------|---------------:|------------:|-------------------------:|--------------------------:|
| attn.q | 0.083 | 57 | 1.45 | 0.051 |
| attn.k | 0.083 | 49 | 1.45 | 0.051 |
| attn.v | 0.084 | 99 | 1.46 | 0.051 |
| attn.o | 0.084 | 21 | 1.46 | 0.051 |

- Shared mean holds only **~8%** of energy.
- Layers are **nearly mutually orthogonal** (pairwise diff ≈ √2; no pair is close).
- Commutator sits at the **generic random-matrix scale** — NOT evidence of commuting.

## 3. Structured models (attn.q)

| model | shared d² mats | r=0 NFE | r=8 NFE |
|-------|---------------:|--------:|--------:|
| shared base | 1 | 0.917 | 0.823 |
| dict q2 s2 | 2 | 0.676 | 0.606 |
| dict q4 s2 | 4 | **0.480** | **0.433** |
| single-generator power (+base) | 2 | **4.25 ✗** | 0.936 |
| word (2-gen, L≤3, near-identity) | 3 | 0.668 | 0.606 |

- **Power model fails outright** (r=0 NFE > 1 means it adds more error than signal).
- Word model (3 stored matrices) only matches dict q2 (2 stored matrices); a matched-
  budget dict q3 (≈0.58 interpolated) beats it → the word restriction buys nothing.
  *(Fair, non-near-identity word variant: see fairness_attn_q.json.)*

## 4. Null-model controls (r=8, gain over independent LR)

The improvement shared/dict give over independent low-rank, on real vs controls that
**destroy all cross-layer relationships but keep each layer's spectrum**:

| variant | gain_shared (attn.q) | gain_dict (attn.q) |
|---------|---------------------:|-------------------:|
| **real weights** | +0.076 | +0.467 |
| spectrum-matched random | +0.076 | +0.467 |
| rotated (Qᵢ Wᵢ Qᵢᵀ) | +0.076 | +0.467 |
| gaussian (Fro-matched) | +0.083 | +0.500 |

**The dict/shared gain is identical on real weights and spectrum-matched randoms** (to 4
decimals for all four roles). Interpretation: the entire weight-space "sharing" gain is a
function of each layer's **spectrum**, not of any genuine shared subspace — because the
real layers are as mutually orthogonal as random matrices with the same spectra. This is
the single strongest weight-space result and it applies even to the *dictionary*.

## 5. Activation-aware (the decisive test)

> The weight-space null result could still be overturned if real weights are aligned with
> the actual activation subspace in a way random matrices are not. Filled in by
> `scripts/05_activations.py` → `results/gpt2/activation_report.json`.

_(pending — see activation_report.json)_

## Provisional conclusion (weight-space)

Of the spec's four outcome forms, the evidence points to **"arbitrary dictionaries reduce
Frobenius error but generator words do not — and even the dictionary gain is a spectral
artifact, not genuine cross-layer algebra."** Confirmation requires the activation-aware
section and replication on gpt2-medium / Pythia.
