---
title: "🔬 MicroDiffusion LM"
summary: "A from-scratch discrete diffusion language model — built step-by-step to understand how text diffusion compares to GPT on Tiny Shakespeare."
tags:
  - Deep Learning
  - NLP
  - Generative AI
  - Transformers
  - PyTorch
date: 2026-03-10
image:
  filename: race.gif
  focal_point: Smart
links:
  - icon: github
    icon_pack: fab
    name: Source Code
    url: https://github.com/BrutalCaeser/microDLM
---

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

*Built with PyTorch · 10.7M parameters · Trained on Tiny Shakespeare · MIT License*
