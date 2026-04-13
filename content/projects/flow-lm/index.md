---
title: "🌊 Flow Map Language Models"
summary: "Reproducing and extending FMLM — a continuous flow approach to one-step text generation. Discovered a novel non-monotonic step-quality curve not reported in the original paper."
tags:
  - NLP
  - Generative AI
  - Flow Matching
  - Research
  - PyTorch
date: 2026-04-11
featured: true
image:
  focal_point: Smart
links:
  - icon: github
    icon_pack: fab
    name: Fork
    url: https://github.com/BrutalCaeser/Flow-Language-Model
---

Language models have always generated text **one token at a time** (left-to-right, like GPT). Flow Map Language Models challenge this assumption — generating coherent text **in a single forward pass** by treating language as a continuous flow problem on the token simplex.

This project reproduces FLM/FMLM on two datasets (LM1B and OpenWebText), extends the evaluation with a pre-registered seed-variance study, and uncovers a **non-monotonic step-quality curve** not characterized in the original paper.

<!--more-->

## The Core Idea

Each token is represented as a **one-hot vector** in ℝ^|V| (50K dimensions for GPT-2 vocabulary). The model learns a continuous probability flow ODE from Gaussian noise to the data distribution:

```
I_t = (1 - t)·x_0  +  t·x_1
```

where x₀ ~ N(0, I) is noise and x₁ is a one-hot token. A transformer denoiser predicts the clean token from the noisy state — with **bidirectional attention** seeing all positions simultaneously.

**Why one-hot works geometrically:** 50K vertices in ℝ^50K are perfectly orthogonal (every pair at distance √2). Decision boundaries are clean hyperplanes. Gaussian initialization gives rich directional diversity that simplex-confined diffusion lacks.

**Why this beats discrete diffusion at few steps:** Both models use the same transformer architecture. The difference is post-network: FLM's ODE step is deterministic and joint (inter-token correlations survive). Discrete ancestral sampling draws each token from its marginal independently — discarding exactly what the network computed.

---

## Results — Reproduction

FLM and FMLM reproduced on Northeastern Explorer HPC (H200/A100), with flash-attn 2.8.3 + torch 2.5.1+cu124.

### FLM Teacher (1024-step ODE)

| Dataset | Our Gen-PPL | Paper | Status |
|---------|------------|-------|--------|
| LM1B    | **99.20**  | ~97   | ✅ within 2% |
| OWT     | **63.13**  | 62.23 | ✅ within 0.9% |

### FMLM Student — LM1B (single-seed sweep)

| Steps | Gen-PPL |
|-------|---------|
| 1     | 115.36 (paper: ~119 — we beat it) |
| 4     | 106.72 |
| 8     | 99.45  |
| **16** | **92.84 ← best** |
| 32    | 102.45 |
| 64    | 105.72 |

**The student at 16 steps (92.84) surpasses the teacher at 1024 steps (99.20).** The paper only reports 1-step and 4-step results — the full curve is new information.

---

## 🔑 Novel Finding: The 16-Step Sweet Spot

The step-quality relationship is **non-monotonic**: quality improves 1→16 steps, then degrades 16→64.

To confirm this is not a lucky seed, we ran a **pre-registered seed-variance experiment** (5 seeds × 5 step counts = 25 runs on LM1B before seeing results):

| Steps | Mean Gen-PPL | Std  | Entropy |
|-------|-------------|------|---------|
| 1     | 115.38      | 1.70 | 4.143   |
| 8     | 99.70       | 1.59 | 4.227   |
| **16** | **93.49**  | **0.75** | 4.230 |
| 32    | 102.41      | 1.55 | 4.263   |
| 64    | 106.14      | 1.03 | 4.293   |

- Step=16 ranks **#1 in 5/5 seeds**, with the lowest variance (std=0.75)
- **Entropy is monotonically increasing** — the improvement at 16 steps is not an entropy collapse artifact
- The student at 16 steps (93.49) surpasses the teacher at 1024 steps (99.20) by **5.71 gen-PPL points**

### ⚠️ Honest Scope

The 16-step finding is **dataset-scoped**: it holds on LM1B (sequence length L=128) but does not replicate on OpenWebText (L=1024). The OWT step sweep showed a confound — the gamma=1 improvements are entropy-driven (output entropy 5.02 vs. dataset entropy 5.44). "Sequence-length scaling" is a working hypothesis, not a confirmed finding.

---

## What Makes This Interesting

The non-monotonic result raises a genuine open question: **why does distillation at 16 steps produce better quality than both the 1-step student and the 1024-step teacher?**

This has no obvious explanation from the paper's theoretical framework, which only compares 1-step FMLM to many-step FLM. Understanding the mechanism could have implications for consistency distillation (multi-step flow map composition) and deployment: 16-step FMLM may be the right operating point, not 1-step.

---

## Paper Context: What FLM Gets Right (and Omits)

| Claim | Honest Verdict |
|-------|---------------|
| Beats discrete at 1 step | **True. No contest.** Factorization error in ancestral sampling is catastrophic. |
| Beats discrete at 4–8 steps | **True vs. ancestral. False vs. Duo+Greedy-Tail.** |
| Enables guidance discrete can't | **True. Structural: differentiable ODE → ∇reward well-defined.** |
| "Continuous flows are the future" | **Premature.** Duo+DCD+Greedy-Tail is competitive at ≥8 steps. |
| Data likelihood (test PPL) | **Unknown — not reported.** |

---

## What's Next: Stage 3

Two research directions, both starting from the same working pipeline:

**Direction A — FLM Inversion:** Implementing the reverse-time ODE to enable text editing, style transfer, and semantic interpolation in noise space. At inference, invert a real sentence into its noise representation, then re-decode with a different style or structure. This would be the first DDIM-style inversion applied to continuous flow language models — no discrete diffusion analogue exists.

**Direction B — Trajectory Analysis:** At each ODE step, the denoiser output D_t(x_t) is a probability distribution over the vocabulary at every position. Nobody has studied what these intermediate states encode. Three concrete questions:
1. **When does each token commit?** Sharp phase transition or gradual sharpening?
2. **What predicts commitment order?** Predictable tokens first, hard tokens last?
3. **Do inter-token correlations appear?** Can we see positions "coordinating" predictions mid-trajectory?

If the intermediate trajectory shows structured, coordinated computation — positions influencing each other's evolution — that's direct evidence of a capability discrete diffusion sampling structurally cannot replicate (it discards all inter-token correlations at the final argmax step). All measurements use the denoiser output D_t already computed at every ODE step — no extra forward passes.

---

*Reproduced on Northeastern HPC (H200/A100) · Pre-registered seed-variance experiment · Novel 16-step finding · MIT License*
