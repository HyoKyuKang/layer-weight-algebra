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

**Answer: powers no; words — early yes (with free generators).**
- Single-generator powers do not help (worse than shared base; the r=0 blowup was a
  trainable-base instability, since fixed — refit pending).
- Two-generator **free** words DO beat a matched-budget dictionary at r=0 on attn.q (see Q4).
  The near-identity parameterization the spec suggested is what made words look useless.
- Evidence: `results_summary.md` §3; `fairness_attn_q.json`.
- Confidence: **high** for powers; **low→medium (positive)** for words — one datapoint.

### Q4. At a matched storage budget, do generator words outperform an unconstrained dictionary?

**Answer: preliminary YES for free-generator words (reversing the earlier read).**
- At 3 stored matrices, r=0, attn.q: **word FREE = 0.529 < dict q3 = 0.575**. Short words
  over 2 free generators yield ~15 atoms from 3 stored matrices — a real matched-budget edge.
- Near-identity words (0.627) lose; the constraint, not the idea, was the problem.
- Margin is small (0.046), single group/rank/seed, and dict q4 (4 mats) still wins overall.
- Evidence: `results_summary.md` §3; `fairness_attn_q.json`.
- Confidence: **low→medium (positive signal)** — needs r=8 free-word, other roles, seeds,
  models, and optimization-effort parity before it can be claimed.

### Q5. Does raw weight reconstruction correlate with activation reconstruction and downstream?

**Answer: yes, ranking-wise — and it does NOT rescue generator words.** Through real
calibration activations (r=16), model order on activation error is dict < word < shared <
ind in 3/4 roles; the dictionary gives a genuine functional gain (attn.q act_err
0.79→0.53, cosine 0.44→0.62) but generator words still lose to it. At attn.o a shared
component even *raises* activation error above independent low-rank. So the weight-space
verdict survives the functional test for the novel models.
- Evidence: `results_summary.md` §5; `results/gpt2/activation_report.json`.
- Confidence: **medium→high**. One refinement left: an activation-space null for the
  *dictionary* (to label its functional gain as real structure vs spectral). Downstream
  perplexity/accuracy not yet measured.

### Q6. If generator models fail, can we characterize why / which modules are heterogeneous?

**Partial answer (why):** same-role attention layers are (a) high stable rank (21–99/768),
(b) mutually near-orthogonal, (c) non-commuting at the generic random scale. So there is
no low-dimensional shared subspace and no short algebraic code to exploit; the residual
rank after removing any shared component stays high. ⏳ Which modules/depths: needs MLP +
depth/family sweep.
- Evidence: `results_summary.md` §2; `diagnostics.json`.

---

## Current standing vs the spec's four allowed outcomes

1. strong algebraic sharing exists — **not supported** (heterogeneous in weight space)
2. sharing only in specific modules/depths — **open** (attention checked; MLP/depth pending)
3. arbitrary dictionaries work but generator words do not — **NO LONGER the leading read**:
   free-generator words just beat a matched-budget dictionary at r=0 (attn.q)
4. layers too heterogeneous for this family — weight-space heterogeneous, yet a short word
   code still buys a small matched-budget edge — so not fully this either

**Revised leading read (session 1):** the honest state is *unsettled and mildly positive
for the novel model*. Weight-space cross-layer "sharing" (shared base / dictionary) is
largely a spectral artifact (null-reproducible), AND the unconstrained dictionary is the
strongest simple model — but **free-generator short words show a first, small matched-budget
win over the dictionary**, which is exactly the effect the project set out to find. Whether
it survives (r=8, all roles, seeds, gpt2-medium/Pythia, activation-aware, optimization
parity) is the whole question for session 2. **No final claim yet** — the negative read was
premature; the positive read is one datapoint. Resolve, do not force, either way.
