# Conclusions (living document)

Answers to the six research questions, each tagged with its **evidence** and a
**confidence** level. Weight-space answers are established; two items (activation-aware,
cross-model replication) gate the final confidence and are marked ⏳.

Scope so far: **GPT-2 (124M), attention Q/K/V/O, 12 layers, d=768.** Not yet: MLP,
larger GPT-2, other families. A negative result here is claimed only for *this scope*.

---

### Q1. Do pretrained same-role weights contain meaningful cross-layer shared structure?

**Provisional answer: No, not in weight space beyond what spectra alone imply.**
- Shared mean captures ~8% of energy; layers are near-orthogonal (pairwise diff ≈ √2).
- Null control: shared/dict gain over independent low-rank is **identical** on real
  weights and spectrum-matched randoms → the "sharing" is a per-layer *spectral* artifact.
- Evidence: `results_summary.md` §2,§4; `results/gpt2/{diagnostics,nulls}.json`.
- Confidence: **high (weight-space)**, ⏳ pending activation-aware confirmation.

### Q2. Does a shared base plus low-rank residual already explain most of the structure?

**Answer: No.** Shared base + rank-32 residual still leaves ~66% of energy; the base adds
only ~0.06–0.08 NFE over independent low-rank, and that increment is null-reproducible.
- Evidence: `results_summary.md` §1,§4.
- Confidence: **high (weight-space)**.

### Q3. Do matrix powers / noncommutative generator words reduce the required residual rank?

**Answer: No.**
- Single-generator powers *fail* (r=0 NFE > 1: worse than zero reconstruction).
- Two-generator words do not beat an unconstrained dictionary at matched shared budget.
- ⏳ Fair (non-near-identity) word variant: `fairness_attn_q.json` (in progress).
- Confidence: **high** for powers; **medium→high** for words pending the fair variant.

### Q4. At a matched storage budget, do generator words outperform an unconstrained dictionary?

**Answer: No** (this is the project's decisive comparison and the bar for novelty).
- word (3 stored matrices) ≈ dict q2 (2 stored matrices); dict q3 at the same budget wins.
- Evidence: `results_summary.md` §3; `fairness_attn_q.json` ⏳.
- Confidence: **medium→high**, finalized by the fair-variant sweep.

### Q5. Does raw weight reconstruction correlate with activation reconstruction and downstream?

**⏳ Open — the decisive test.** If real weights are aligned with the activation subspace
in a way spectrum-matched randoms are not, activation-aware error could show structure
that Frobenius error hides. `scripts/05_activations.py` → `activation_report.json`.
- Confidence: **pending**.

### Q6. If generator models fail, can we characterize why / which modules are heterogeneous?

**Partial answer (why):** same-role attention layers are (a) high stable rank (21–99/768),
(b) mutually near-orthogonal, (c) non-commuting at the generic random scale. So there is
no low-dimensional shared subspace and no short algebraic code to exploit; the residual
rank after removing any shared component stays high. ⏳ Which modules/depths: needs MLP +
depth/family sweep.
- Evidence: `results_summary.md` §2; `diagnostics.json`.

---

## Current standing vs the spec's four allowed outcomes

1. strong algebraic sharing exists — **rejected (weight-space)**
2. sharing only in specific modules/depths — **open** (attention checked; MLP/depth pending)
3. arbitrary dictionaries work but generator words do not — **best-supported so far**,
   with the caveat that even the dictionary's *gain* is spectral (null-reproducible)
4. layers too heterogeneous for this family — **supported** for attention in weight space

Leading read: a blend of (3) and (4) — dictionaries lower Frobenius error but not via real
cross-layer algebra; generator words add nothing. **Final claim withheld** until the
activation-aware test (Q5) and at least one replication (gpt2-medium or Pythia) are in.
