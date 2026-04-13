---
title: "🔍 SIGReg World Model"
summary: "A rigorous investigation into whether distributional regularization (SIGReg) can prevent representational collapse in self-supervised video features — with a mathematical proof of why it can't."
tags:
  - Computer Vision
  - Deep Learning
  - Research
  - World Models
  - Self-Supervised Learning
  - PyTorch
date: 2026-04-06
featured: true
image:
  focal_point: Smart
links:
  - icon: github
    icon_pack: fab
    name: Code (Private)
    url: https://github.com/BrutalCaeser
---

Self-supervised video representations are the foundation of any learned world model — but they suffer from **representational collapse**: the encoder maps all inputs to a near-identical vector, destroying the spatial structure needed for downstream tasks.

SIGReg (a distributional regularizer based on the Epps-Pulley characteristic function test) has been proposed as a way to prevent this collapse. This project asks: **does it actually work?**

The answer is no — but understanding *why* required a full mathematical proof and 10 sessions of controlled experiments on the Northeastern HPC cluster.

<!--more-->

## Architecture

```
Frozen V-JEPA 2.1 (ViT-L, 196 patches × 1024-dim)
  → PatchAdapter (token-wise MLP, ℝ^1024 → ℝ^256)
  → JEPAPredictor (6-layer transformer, AdaLN conditioning)
  → Losses: L_pred + λ₁·SIGReg(z_c) + λ₂·L_info_dense(ẑ, z_t)
```

**Dataset:** Something-Something-v2 (SSv2) — 20K clips of human actions. Features extracted from frozen V-JEPA 2.1 (ViT-L) stored as 40,800 `.pt` files on the Explorer cluster. Baseline: raw erank = 71.92 ✅, raw cross-correlation = 0.683 ✅.

---

## Experimental Design

8 conditions systematically isolated each component:

| Condition | Stop-grad | SIGReg | L_info | Result |
|-----------|-----------|--------|--------|--------|
| A — Collapse baseline | No | No | No | erank → 1.01 by step 2000 (designed to collapse) |
| B — Stop-gradient only | Yes | No | No | erank → 1.04 — collapsed anyway |
| C — Stop-grad + SIGReg | Yes | Global | No | Collapsed |
| D1 — Token SIGReg | No | Per-token | No | Collapsed |
| D2 — Channel SIGReg | No | Per-channel | No | Collapsed |
| D3 — Global SIGReg | No | Global | No | Collapsed |
| E — Full method | No | Global | Yes | Collapsed |
| F — Info only (ablation) | No | No | Yes | Diverged |

**Every condition collapsed.** This is not a matter of hyperparameters — it's structural.

---

## The Mathematical Proof

### Why SIGReg has exactly zero gradient at collapse

The SIGReg forward pass with all-collapsed inputs z_k = c (constant vector):

1. All projected values h_k = c · u are identical
2. After standardization: h̃_k = (h_k − mean) / std = 0 / 0 → **0**
3. EP test on zeros: cos(0) = 1, **sin(0) = 0**
4. Backward pass: all g_k identical → g_k − mean(g) = **0**

SIGReg correctly reports a loss of ~0.204 (it detects the problem). But its **gradient is mathematically zero** — not approximately small, but exactly zero at the collapsed state. The optimizer receives no signal. Collapse is a stable fixed point.

Two independent mechanisms both give zero:
- `∂(EP test) / ∂h̃` = 0 because sin(0) = 0
- Mean-subtraction in the backward pass cancels all gradients by symmetry

### Why high lambda makes things worse

At lambda = 10–50, a new problem emerges: the optimizer discovers the **1D variance trick**. The adapter collapses to a 1D subspace with direction v and variance σ₁² ≈ d. Random projections then see approximately N(0,1) in expectation (by symmetry of random directions on a sphere). SIGReg actively *decreases* while erank collapses — the optimizer satisfies the regularizer by collapsing.

This was confirmed empirically across lambda ∈ {0.1, 10, 25, 50} and both global (K=32) and token-level (K=6272) SIGReg.

### The impossible dilemma for lambda

| λ₁ | Scale explosion? | SIGReg active? | Outcome |
|----|-----------------|----------------|---------|
| 0.1 | Possible | Ignored (< 5% of gradient) | Collapse |
| 10–50 | Controlled | Exploited (1D variance trick) | Collapse |
| Any | Post-collapse | **Gradient = 0** | Stuck |

No setting of lambda solves both problems simultaneously.

---

## Diagnostic Sweep: Isolating the Root Causes

Three factors from the original SIGReg papers that we didn't match:

| Setting | Our setup | LeJEPA / LeWorldModel |
|---------|-----------|----------------------|
| Batch size | 32 | 2048–4096 |
| Init gain | 0.1 (near-identity) | 1.0 (random) |
| BatchNorm on projector | No | Yes ("critical") |

Controlled diagnostic sweep adding each factor:

| Run | Intervention | erank @ step 400 | Verdict |
|-----|-------------|-----------------|---------|
| A (baseline) | None | 1.05 | Collapsed |
| B | Batch size → 128 | 1.03 | Still collapsed |
| C | + Init gain → 1.0 | 1.05 | Still collapsed |
| D | + BatchNorm | 1.05 | Still collapsed — BN *masks* collapse from SIGReg |

**All four failed.** BatchNorm is particularly revealing: it normalizes the single dominant direction at collapse, making the collapsed representation *look* 256-dimensional to SIGReg. The erank (computed on raw z_c) still drops to 1.05. BN masks collapse rather than preventing it.

---

## What This Proved

**SIGReg cannot replace stop-gradient or EMA as a collapse prevention mechanism in a shared-weight JEPA adapter architecture.**

This is a **genuine negative result** — the kind that's hard to publish but important to document. The proof has implications beyond this specific setup: any regularizer whose gradient depends on the distribution of adapter outputs will face the same dead zone at collapse.

---

## What Came Next: EMA (Also Failed)

After the impossibility proof for output-based losses, **EMA (Exponential Moving Average)** was tried as a structural fix: use a slowly-updating target adapter θ̄ (EMA of the online adapter θ) to compute z_t, rather than the online adapter. If θ collapses, θ̄ still produces diverse targets → L_pred stays high → gradient resists collapse. This is the BYOL/DINO mechanism.

7 EMA conditions were prepared (B through F variants + 2 ablations). First test (B_ema v1, decay=0.996) collapsed at step ~1000: near-identity init (gain=0.1) starts the adapter near the collapse basin, and EMA lag of ~250 steps is shorter than the collapse timescale (~500 steps). The online adapter reaches collapse before the EMA target diverges enough to resist.

Fix applied: `init_gain=1.0` (random init, full-rank start) + `ema_decay=0.999` (lag ≈ 1000 steps). **All subsequent EMA conditions also collapsed.** The collapse attractor proved strong enough to pull the online adapter down within the EMA lag window regardless of initial conditions.

**Definitive result (10 sessions, 77 commits):** No combination of output-based regularization or target-encoder structure prevents rank-1 collapse in a shared-weight JEPA adapter setup on pre-extracted frozen features. The project was closed.

---

## What It Motivated

The impossibility proof — that **no loss function computed from adapter outputs can prevent collapse when gradients are confined to the collapsed 1D subspace** — directly motivated Physbridge: if output-based regularization cannot carry spatial structure, use a reference process that carries spatial structure by construction.

Physbridge is a Schrödinger Bridge formulation with Newtonian free-flight dynamics as the reference measure — the reference process itself encodes physical spatial priors, bypassing the gradient dead zone entirely.

---

## Technical Highlights

- **77 commits** across 10 sessions on Northeastern Explorer HPC (V100/A100/H200)
- **SLURM infrastructure**: auto-resubmit, per-condition job naming, WANDB offline mode for compute nodes
- **Full test suite**: 27 unit tests + smoke tests per condition before job submission
- **Metrics frozen** at experiment start — all conditions evaluated identically (metrics.py never modified post-submission)
- **W&B logging**: all metrics, losses, gradient norms at every evaluation step

---

*10 sessions · 77 commits · 8 experimental conditions + EMA phase · Mathematical proof of gradient dead zone · Northeastern Explorer HPC*
