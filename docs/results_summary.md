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

## 3. Structured models — matched-budget test (attn.q)

Budget axis = number of shared d² matrices stored. This is the project's decisive
comparison. From `fairness_attn_q.json`:

| model | shared d² mats | r=0 NFE | r=8 NFE |
|-------|---------------:|--------:|--------:|
| shared base | 1 | 0.917 | 0.823 |
| dict q2 s2 | 2 | 0.676 | 0.606 |
| dict q3 s2 | 3 | 0.575 | 0.516 |
| dict q4 s2 | 4 | 0.480 | 0.433 |
| dict q6 s2 | 6 | 0.331 | 0.298 |
| word near-identity s2 | 3 | 0.627 | — |
| **word FREE-generators s2** | **3** | **0.529** | **0.493** |
| single-generator power (+base) | 2 | (unstable, refit) | 0.936 |

**Headline (revised):** the *near-identity* word constraint cripples the model (0.627,
worse than dict q3=0.575), BUT with **free generators** the word model beats the
matched-budget dict q3 at **both** ranks — **r=0: 0.529 < 0.575** and **r=8: 0.493 <
0.516** (3 stored matrices each). Short noncommutative words over 2 free generators expose
~15 usable atoms from 3 stored matrices, a small but *consistent* matched-budget edge over
an unconstrained 3-atom dictionary (attn.q). This is the first positive signal and
**overturns the earlier near-identity-based negative read**; still one group/seed and
dict q4 (4 mats) wins overall, so confirm across roles, seeds, and models before any claim.

Caveats: single group / rank / seed; margin 0.046; word fit is non-convex torch vs dict's
clean SVD (optimization-effort parity to check). The `power` r=0 value in the first run was
a training blowup (trainable base diverged); powers.py now fixes B at the mean — refit
pending.

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

Calibration: builtin 36-sentence corpus through gpt2, ~556 tokens/layer captured as the
real input X to each weight. Metrics at **r=16**: `act_err = ‖(W−Ŵ)X‖²/‖WX‖²`,
`cos` = mean output cosine. (`results/gpt2/activation_report.json`.)

| role | model | weight NFE | act_err | out cosine |
|------|-------|-----------:|--------:|-----------:|
| attn.q | ind | 0.831 | 0.788 | 0.439 |
| attn.q | shared | 0.759 | 0.729 | 0.508 |
| attn.q | **dict q4s2** | 0.399 | **0.534** | **0.618** |
| attn.q | word s2 | 0.524 | 0.631 | 0.577 |
| attn.o | ind | 0.768 | 0.610 | 0.577 |
| attn.o | shared | 0.703 | 0.691 | 0.533 |
| attn.o | **dict q4s2** | 0.375 | **0.486** | **0.660** |
| attn.o | word s2 | 0.530 | 0.658 | 0.565 |

Findings:
- **Model ranking is preserved through activations**: dict < word < shared < ind on
  `act_err` (in 3/4 roles). The activation-aware test does **not** rescue generator words —
  they still lose to the unconstrained dictionary functionally.
- The dictionary yields a *genuine* functional gain (attn.q act_err 0.79→0.53, cos
  0.44→0.62), but that is the known MASA/Basis-Sharing dictionary, not our word/power model.
- **Inversion at attn.o**: shared/word have *higher* act_err than plain independent
  low-rank (0.69/0.66 vs 0.61) — a shared component can hurt functionally there.
- Interesting nuance: for the dictionary, `act_err > weight NFE` (errors sit in
  activation-relevant directions), while for ind/shared `act_err < weight NFE` (errors
  hide in unused directions).

_Open refinement:_ an activation-space null (does dict's act_err gain survive spectrum-
matched weights measured through the same X?) would fully separate "real functional
structure" from "spectral artifact" for the dictionary. Words are already settled: they
lose to dict in both spaces.

## Provisional conclusion (weight-space)

Of the spec's four outcome forms, the evidence points to **"arbitrary dictionaries reduce
Frobenius error but generator words do not — and even the dictionary gain is a spectral
artifact, not genuine cross-layer algebra."** Confirmation requires the activation-aware
section and replication on gpt2-medium / Pythia.
