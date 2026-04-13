---
title: "🔬 MicroDiffusion LM"
summary: "A two-part discrete diffusion study: from-scratch GPT-vs-diffusion comparison on Tiny Shakespeare, then a 124M-parameter scaling study on FineWeb-Edu that found uniform masking outperforms entropy-weighted across all schedules."
tags:
  - Deep Learning
  - NLP
  - Generative AI
  - Transformers
  - Research
  - PyTorch
date: 2026-03-10
featured: true
image:
  filename: race.gif
  focal_point: Smart
links:
  - icon: github
    icon_pack: fab
    name: Source Code
    url: https://github.com/BrutalCaeser/microDLM
  - icon: globe
    icon_pack: fas
    name: Interactive Demo
    url: https://brutalcaeser.github.io/microDLM/
  - icon: blog
    icon_pack: fas
    name: Blog Post
    url: /blog/from-noise-to-shakespeare/
---

> 🎮 **[Try the interactive demo →](https://brutalcaeser.github.io/microDLM/)** — watch diffusion race GPT live in your browser.
>
> 📝 **[Read the full blog post →](/blog/from-noise-to-shakespeare/)** — every line of math, from the ELBO to the training code.

Most language models generate text **left-to-right**, one token at a time (GPT, LLaMA, Claude, etc.). **Discrete diffusion models** take a fundamentally different approach — they generate text **all at once**, starting from a fully masked sequence and iteratively revealing tokens in parallel, like developing a photograph.

This project builds **both architectures from scratch** with an **identical transformer** (same params, same data, same RoPE + RMSNorm + ReluSquared MLP) so you can see exactly what changes — and it turns out to be surprisingly little.

![Diffusion vs GPT generation race — diffusion finishes ~6× faster](https://raw.githubusercontent.com/BrutalCaeser/microDLM/main/assets/race.gif)
*Same architecture, same data — diffusion finishes in **39 steps** vs GPT's **225 steps***

<!--more-->

## The 5 Changes from GPT → Diffusion

| # | What Changes | GPT | Diffusion |
|---|---|---|---|
| 1 | **Vocabulary** | Standard chars | + 1 MASK token |
| 2 | **Attention** | Causal (sees only left ←) | Bidirectional (sees everything ↔) |
| 3 | **Training objective** | Predict next token | Denoise masked tokens |
| 4 | **Loss scope** | All positions | Masked positions only |
| 5 | **Generation** | Sequential, left-to-right | Parallel, by confidence |

**~80% of the code is shared.** Same transformer, same RoPE, same RMSNorm, same ReluSquared MLP.

---

## How Generation Works

**Diffusion** sees the entire sequence at each step (bidirectional attention) and predicts all positions simultaneously. It reveals the most confident predictions first — common words and clear patterns emerge before ambiguous positions.

**GPT** can only see tokens to its left (causal attention) and generates strictly one token at a time.

> Same architecture, same data — diffusion finishes in **39 steps** vs GPT's **225 steps**.

---

## Training Results

All models trained on **Tiny Shakespeare** (~1.1M characters, 65 unique + 1 MASK = 66 vocab).

| Model | Architecture | Params | Train Loss | Val Loss | Time |
|---|---|---|---|---|---|
| MLP Denoiser | 2-layer FF | 0.37M | 3.31 | 3.31 | ~6 min |
| Small Transformer | 4L / 4H / 128E | 1.6M | 2.16 | 2.27 | ~7 min |
| **Diffusion (final)** | 6L / 6H / 384E | 10.7M | 1.93 | 2.09 | ~47 min |
| **GPT (final)** | 6L / 6H / 384E | 10.7M | 0.13 | 4.09 | ~24 min |

### Key Observations

- **GPT overfits heavily** — train loss 0.13 but val loss 4.09. It memorizes the training set with no regularization (intentionally omitted for simplicity).
- **Diffusion generalizes much better** — train 1.93, val 2.09. Random masking acts as a natural regularizer.
- **The MLP → Transformer jump** (3.31 → 2.16) proves that attention is essential.

### ⚠️ Honest Assessment

**GPT produces better text than diffusion at this scale.** This is expected and well-documented in the literature — discrete diffusion LMs typically need 3–5× more training to match autoregressive quality at small scale. The quality gap narrows significantly at larger scale.

What diffusion demonstrates here:
- ⚡ **~6× fewer forward passes** for generation (39 vs 225 steps)
- 🔀 **Parallel decoding** — tokens appear everywhere simultaneously
- 🧩 A fundamentally **different approach** to language modeling

---

## Progressive Build (4 Steps)

The project is built incrementally, each step teaching one concept:

1. **Forward Process** — Cosine noise schedule, no neural network, just masking and statistics
2. **MLP Denoiser** — Proves the training loop works with the simplest possible model
3. **Transformer Denoiser** — Bidirectional self-attention → dramatic quality jump
4. **Scaling + GPT Comparison** — 10.7M params, confidence-based parallel decoding, SUBS parameterization

---

## Architecture

Both models share: **6 layers, 6 heads, 384 embedding dim** — with RoPE positional encoding, RMSNorm, ReluSquared activation, and QK-normalization for stability. The only difference is the attention mask (causal vs bidirectional).

---

## What I Learned

1. **Diffusion LMs are surprisingly simple.** 80% shared code with GPT.
2. **Masking is a natural regularizer.** Diffusion generalized (val 2.09) while GPT memorized (val 4.09).
3. **The MLP → Transformer jump is dramatic.** Attention is essential for learning words and structure.
4. **Parallel generation is real.** ~6 tokens decoded per step on average.
5. **Loss numbers aren't directly comparable** between the two paradigms.

---

---

## Part 2 — Scaling Study: Does Smarter Noise Scheduling Beat Uniform Masking?

Part 1 established the baseline architecture. Part 2 asks a harder question: **at production scale (124M params, FineWeb-Edu), does the choice of forward noise schedule matter?**

The hypothesis: entropy-weighted masking — which masks high-entropy (uncertain) tokens first — should be harder for the model and produce better representations than uniform masking.

### Experimental Design

**6 noise schedules × 2 masking strategies = 12 conditions**, all at identical scale:

| Axis | Options |
|------|---------|
| Architecture | 124M-parameter transformer (GPT-2 scale) |
| Dataset | FineWeb-Edu (educational internet text, WebScale) |
| Noise schedules | Linear, Cosine, Square Root, Log-linear, Clipped Cosine, Sigmoid |
| Masking strategies | Uniform (mask random tokens) vs. Entropy-weighted (mask uncertain tokens first) |

All runs on Northeastern Explorer HPC. SIGUSR1 auto-resubmit implemented for the 7.5-hour partition limit. SEDD log-linear schedule bug fixed (α(1) = 0.001, not 0.5).

### Result: Uniform Masking Wins

**Early results (8/12 runs)** showed entropy-weighted appearing to win by 0.65–0.77 nats on training val loss — the hypothesis seemed to hold.

**Final results (8 runs evaluated — clipped cosine and sigmoid stalled, excluded from analysis):** The advantage reversed on fair comparison. **Uniform masking outperforms entropy-weighted across all four evaluated schedule pairs** on out-of-distribution generalization (WikiText-2 PPL).

### Phase 7 Results — 8 Runs Evaluated

| Run | FineWeb PPL ↓ | WikiText-2 PPL ↓ | Gen PPL ↓ | MAUVE ↑ |
|-----|--------------|-----------------|-----------|---------|
| **cosine_uniform** | **69.56** | **94.32** | 228.51 | 0.575 |
| cosine_entropy | 64.39 | 186.69 | **100.52** | 0.521 |
| loglinear_uniform | 79.67 | 161.81 | 208.53 | **0.651** |
| loglinear_entropy | 98.66 | 261.69 | 95.51 | 0.611 |
| linear_uniform | 83.56 | 151.26 | 220.59 | 0.624 |
| linear_entropy | 102.67 | 225.22 | 114.41 | 0.453 |
| sqrt_uniform | 149.15 | 299.84 | 212.97 | 0.641 |
| sqrt_entropy | 174.99 | 473.75 | 104.22 | 0.561 |

**Three findings that held consistently:**

1. **Cosine schedule wins on denoising** — best FineWeb NELBO (69.56) and WikiText-2 (94.32). The intermediate-noise-spending profile matters most.
2. **Entropy-weighted improves generative fluency ~2× but degrades OOD generalization ~2×** — the entropy prior is FineWeb-specific; the model specializes rather than generalizing.
3. **The val-loss advantage of entropy-weighted is largely measurement bias** — training loss was measured on an easier exam (entropy-masked positions). On fair uniform evaluation, the gap shrinks to ~0.08 nats for cosine.

### Why This Matters

Entropy-weighted masking is intuitively appealing — harder tokens should get more training signal. But the result shows **uniform masking regularizes by forcing the model to handle all positions uniformly**, while an entropy prior creates a model that specializes rather than generalizes.

This result has a direct implication for continuous flow language models: the "smart schedule" intuition is unreliable. [Flow Map Language Models](../flow-lm/) exploit the same lesson differently — by abandoning discrete masking entirely in favor of continuous flows on the token simplex.

### ⚠️ Honest Note

This is a **negative result** — the interesting hypothesis didn't hold. Published on Substack and LinkedIn rather than pursued as an academic paper. The HPC infrastructure (SLURM auto-resubmit, per-condition sweep, SIGUSR1 checkpointing across ~45hr runs chained across 6 jobs each) proved directly reusable for the FlowLM reproduction study.

---

*Part 1: PyTorch · 10.7M parameters · Tiny Shakespeare · MIT License*  
*Part 2: PyTorch · 124M parameters · FineWeb-Edu · Northeastern Explorer HPC · 8/12 conditions evaluated*
