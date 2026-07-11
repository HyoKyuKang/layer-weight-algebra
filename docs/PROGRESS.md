# Progress log

## 2026-07-12 — session 1: foundation + first results

### Built
- Repo scaffolding: `config/`, `src/lwa/`, `scripts/`, `docs/`.
- `extract.py` — same-role square-matrix extraction, operator convention, fused-QKV split.
  Supports GPT-2 (done), GPT-NeoX/Pythia, LLaMA/Mistral/Qwen2 (implemented, untested).
- `metrics.py` — normalized Frobenius / spectral error, `ParamCost` (shared vs per-layer
  bits/params), compression ratio.
- `baselines.py` — `ind`, `mean`, `shared` (base + low-rank, alternating min), `cluster`.
- `diagnostics.py` — spectra, stable rank, pairwise-diff norms, normalized commutators.
- `dictionary.py`, `powers.py`, `words.py` — the three algebraic models.
- `nulls.py` — spectrum-matched / rotated / gaussian controls.
- `activations.py` — hook-based calibration capture + activation-aware error.
- `plotting.py` — reconstruction curves, spectra, heatmaps.
- Scripts 01 (extract), 02 (baselines), 03 (diagnostics), 04 (models).

### First results — GPT-2 attention (d=768, k=12 layers per role)

**Baselines (normalized Frobenius error, lower better):**

| role | mean r0 | ind r32 | shared r32 | cluster r32 |
|------|--------:|--------:|-----------:|------------:|
| attn.q | 0.917 | 0.724 | 0.659 | 0.527 |
| attn.k | 0.917 | 0.698 | 0.637 | 0.506 |
| attn.v | 0.916 | 0.778 | 0.707 | 0.598 |
| attn.o | 0.916 | 0.686 | 0.626 | 0.533 |

**Diagnostics:**
- `energy_fraction_in_mean ≈ 0.083` — shared base captures only ~8% of energy.
- `mean_stable_rank` 21–99 (attn.o most compressible, attn.v least) — matrices are NOT
  low-rank; subtracting the mean barely changes stable rank.
- `pairwise_diff_norm ≈ 1.41–1.46 ≈ √2` — same-role layers are **nearly mutually
  orthogonal** in matrix space; no pair is notably similar.
- `commutator_mean ≈ 0.051 ≈ √2/√d` — this is exactly the *generic random-matrix* scale,
  so it is **not** evidence of commuting structure (must compare vs null, not vs 0).

### Reading so far (provisional)
GPT-2 attention weights look **strongly cross-layer heterogeneous** in weight space:
tiny shared base, high stable rank, near-orthogonal layers. `cluster` beating `shared`
suggests 1–2 outlier layers (likely layer 0) rather than smooth shared structure.
**Caveat:** weight-space only; activation-aware metrics + null comparison still pending,
and GPT-2 is small/less-structured than modern LLMs.

### Next
- [ ] Read generator-model smoke test (dict/power/word vs baselines at matched budget).
- [ ] Null-model comparison (does `shared`/`dict` gain survive spectrum-matched controls?).
- [ ] Activation-aware re-scoring.
- [ ] Scale to gpt2-medium (depth) and Pythia (family) if signal warrants.
