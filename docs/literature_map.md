# Literature map — Layer Weight Algebra

Goal: place the *generator-word* hypothesis against existing cross-layer weight-sharing
and post-training LLM compression work. We distinguish four things per work:
**(1)** existing components, **(2)** the exact generator-word restriction, **(3)** the
low-rank correction, **(4)** the experimental question of whether pretrained layers
actually possess this structure.

Status: **draft v1** (first pass; to be deepened). Search date: 2026-07.

## Closest prior work

| Work | What it shares | Atom / structure | Overlap with us | Difference from us |
|------|----------------|------------------|-----------------|--------------------|
| **ResidualTransformer** (Ge et al., ICASSP'24) | consecutive layers | weight-sharing + **residual low-rank** learned during training | our `shared`/`ind` residual idea | trained from scratch; no shared *base* fit to pretrained weights; no generator words |
| **DeltaLLM** (2501.18596) | tie weights of adjacent blocks; add **low-rank delta** between shared weights | shared W + low-rank Δ | exactly our `shared base + low-rank residual` template, applied blockwise | ties whole blocks & retrains ~30–40M tokens; MLP > attn; no dictionary/word algebra |
| **Basis Sharing** (2410.03765) | across layers | each W = linear combo of **shared basis vectors** + per-layer coeffs (SVD) | this *is* our shared-basis / dictionary baseline | unconstrained basis; no product/word structure; no null-model controls |
| **MASA — Share Your Attention** (2508.04581) | Q,K,V,O across layers | **matrix-based dictionary** of shared atoms + per-layer coeffs | *directly* our `dict` model on attention; strong same-role atom-sharing evidence | unconstrained atoms; trained drop-in; no generator words; no algebraic diagnostics |
| **FiPS** (2411.09816, *Learning Parameter Sharing w/ Tensor Decomp. & Sparsity*) | MLP neurons across layers | **shared base + sparse factors**, SVD-init | our `dict`/sparse-coefficient accounting | MLP-focused; tensor+sparsity, not matrix words |
| Tensor-network / TD post-training pipelines (e.g. 2602.01613) | whole model | Tucker/TT factorization | shared-basis / tensor-decomp baseline family | global TD, not same-role cross-layer algebra |

## What appears genuinely under-explored (novelty surface)

A targeted search for *matrix powers* / *noncommutative products of a few shared
generators* as a weight-generation code returns **no direct LLM-compression hits**. The
nearby matrix-polynomial work is about encoding rank as polynomial degree, not about
generating same-role layer weights as short words over shared generators.

So the decomposition of our contribution is:

1. **Existing components** — shared base, low-rank residual, unconstrained shared
   dictionary/basis: all well-established (DeltaLLM, Basis Sharing, MASA, FiPS).
2. **Exact generator-word restriction** — approximating $W_\ell$ by short noncommutative
   words $w_{\ell t}(G_1,\dots,G_g)$ over a *few shared generators*: **no direct prior
   in LLM compression** found so far. This is the candidate novel object.
3. **Low-rank correction** on top of the algebraic term: standard, but its *interaction*
   with the word term (does structure reduce required rank?) is the open question.
4. **Experimental question** — do pretrained same-role layers actually possess this
   structure, tested against matched-budget dictionaries and null models: this is the
   core empirical contribution, independent of whether the answer is yes or no.

## Consequence for claims

Because unconstrained dictionary / shared-basis sharing is *already known to work*
(MASA, Basis Sharing), beating raw weight-sharing is **not** a contribution. The bar is:
the generator-word model must beat an **unconstrained dictionary at matched bits** and
**survive activation-aware evaluation**. Otherwise the honest result is "arbitrary
dictionaries work, generator words do not" — which our pipeline is built to detect.

## Degenerate / boundary cases of our family

| Work | Where it sits in our parameterization |
|------|----------------------------------------|
| **ALBERT** (Lan et al. 2020) | hard weight-tying: $W_\ell = B$ for all $\ell$ (our `mean` with the base *trained*, no residual, no words). Works when trained from scratch; says nothing about post-hoc structure of *independently* trained layers. |
| **Universal Transformer** (Dehghani et al. 2019) | one shared block applied recurrently = full tying across depth; the $g{=}0$, $r{=}0$ corner. |
| **Head-wise Shareable Attention** (2402.11819) | shares whole attention *heads* across layers — coarser than atom/word sharing. |
| **Hypernetworks** (Ha et al. 2016; HyperLoader 2407.01411) | $W_\ell = f_\theta(\ell)$ for a small net $f$ — a *nonlinear* shared code, strictly more general than our linear words; relevant as an upper-bound competitor for "layers are a low-dim function of depth." |

Note ALBERT/UT show tying *works when you train for it*. Our question is the opposite
direction: does structure survive in weights trained **without** any sharing constraint?
The two are not in tension — a negative post-hoc result is fully consistent with ALBERT.

## To deepen (v2+)

- [ ] Exact MASA atom count / budget and whether it reports same-role vs all-role sharing.
- [ ] Monoid/semigroup-generated matrix families in applied math (outside ML).
- [ ] Whether any work reports **negative** cross-layer sharing for attention specifically.
- [ ] Recurrent-depth / LoopFormer (2602.11451) as a trained-in structural prior.

## Sources

- ResidualTransformer — https://www.researchgate.net/publication/379818242
- DeltaLLM — https://arxiv.org/abs/2501.18596
- Basis Sharing — https://arxiv.org/abs/2410.03765
- MASA / Share Your Attention — https://arxiv.org/abs/2508.04581
- FiPS / Learning Parameter Sharing w/ Tensor Decomp. & Sparsity — https://arxiv.org/abs/2411.09816
- Tensor-network compression pipeline — https://arxiv.org/pdf/2602.01613
- ALBERT — https://huggingface.co/docs/transformers/en/model_doc/albert
- Universal Transformer — https://arxiv.org/abs/1807.03819
- Head-wise Shareable Attention — https://arxiv.org/pdf/2402.11819
- Hypernetworks (HyperLoader) — https://arxiv.org/abs/2407.01411
