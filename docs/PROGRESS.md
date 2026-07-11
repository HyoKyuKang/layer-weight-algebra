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

### Update (later in session 1) — the picture shifted

- **Null controls**: shared/dict gain over independent low-rank is identical on real vs
  spectrum-matched/rotated weights → weight-space "sharing" is a spectral artifact.
- **Activation-aware** (builtin corpus, r=16): model ranking preserved (dict<word<shared<ind);
  dict gives a real functional gain (attn.q act_err 0.79→0.53); words still lose to dict there.
- **Fair word variant is the twist**: near-identity words are useless (0.627 > dict q3 0.575),
  but **FREE-generator words beat dict q3 at matched 3-matrix budget (0.529 < 0.575, r=0,
  attn.q)**. First positive signal → the earlier negative read was premature.
- **Bug fixed**: single-generator power r=0 blew up because the trainable base diverged;
  `powers.py` now fixes B at the mean (fair). Refit pending.

### HANDOFF — to continue in another environment

1. `pip install -r requirements.txt` (needs: numpy scipy scikit-learn torch transformers
   safetensors huggingface_hub datasets pyyaml matplotlib). Note this box was **CPU-only**;
   a CUDA box will make the torch word/power fits ~10-50x faster.
2. Reproduce session 1:
   ```
   python scripts/01_extract_weights.py --model gpt2
   python scripts/02_baselines.py       --model gpt2
   python scripts/03_diagnostics.py     --model gpt2
   python scripts/06_nulls.py           --model gpt2 --rank 8
   python scripts/05_activations.py     --model gpt2      # builtin corpus, no network
   python scripts/07_word_fairness.py   --model gpt2 --role attn.q
   ```
   (`data/`, `results/` are gitignored — regenerate locally.)
3. **Session 2 priorities (in order):**
   - [ ] Finish `07` for r=8 free-word + all 4 roles + 3 seeds (is the 0.046 win real?).
   - [ ] Optimization-effort parity: give dict and word equal compute; try word L≤2 vs L≤3.
   - [ ] Refit powers (B fixed) so its number is fair.
   - [ ] Activation-space null for the dictionary (real structure vs spectral).
   - [ ] Replicate free-word-vs-dict on **gpt2-medium** (depth) and **Pythia-160m** (family).
   - [ ] If the free-word edge holds → measure downstream perplexity with reconstructed weights.
- Source of truth: `docs/CONCLUSIONS.md` (Q-by-Q), `docs/results_summary.md` (numbers).
