---
title: "🧞 MiniGenie"
summary: "A flow matching video world model that learns to imagine game frames — built entirely from scratch in PyTorch with a 42M-parameter U-Net."
tags:
  - Deep Learning
  - Computer Vision
  - Generative AI
  - World Models
  - PyTorch
date: 2026-03-07
featured: true
image:
  filename: rollout_00.gif
  focal_point: Smart
links:
  - icon: github
    icon_pack: fab
    name: Source Code
    url: https://github.com/BrutalCaeser/minigenie
---

MiniGenie is a **flow matching video world model** that predicts the next frame of a game given past frames and a player action. Feed it 4 context frames and an action (e.g., "move right"), and it generates what the next frame should look like — learning the physics, dynamics, and visual style of the game purely from observation.

**Built entirely from scratch.** No pretrained models, no diffusion libraries, no external frameworks beyond PyTorch. Every component — VQ-VAE tokenizer, flow matching U-Net, training loops, evaluation pipeline — is implemented from first principles.

<div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/BrutalCaeser/minigenie/main/assets/rollout_00.gif" alt="MiniGenie rollout — CoinRun level 1" style="width: 256px; height: 256px; image-rendering: pixelated; border-radius: 8px;">
    <p><em>CoinRun Level 1</em></p>
  </div>
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/BrutalCaeser/minigenie/main/assets/rollout_01.gif" alt="MiniGenie rollout — CoinRun level 2" style="width: 256px; height: 256px; image-rendering: pixelated; border-radius: 8px;">
    <p><em>CoinRun Level 2</em></p>
  </div>
</div>
<p style="text-align: center;"><em>Autoregressive rollouts generated entirely by MiniGenie — given 4 real context frames and a sequence of actions, the model imagines what happens next.</em></p>

<!--more-->

## 🎮 Game: CoinRun

🪙 **CoinRun** — A procedurally generated platformer from [Procgen](https://openai.com/research/procgen-benchmark). The agent navigates randomized terrain, avoids enemies, and collects a coin. Every level has a different layout and visual theme — making it a challenging test for a generative model.

---

## Architecture

```text
Context frames (4×3 channels) ──┐
                                 ├─ concat ─→ U-Net ─→ predicted velocity ─→ ODE integrate ─→ next frame
Noisy target (3 channels) ──────┘              ↑
                                          action + time
                                        (AdaGN conditioning)
```

| Component | Details |
|---|---|
| **Dynamics model** | 42M-param pixel-space flow matching U-Net |
| **Conditioning** | Adaptive Group Normalization (AdaGN) — injects action + flow time into every ResBlock |
| **Generation** | 15-step Euler ODE integration with classifier-free guidance (scale 2.0) |
| **Training** | Flow matching loss + noise augmentation (GameNGen technique) + 10% CFG dropout |
| **VQ-VAE** | Standalone tokenizer — 512-entry codebook, EMA updates, L2 normalization |

---

## Results

Trained on **CoinRun** (5,000 episodes, 80K steps):

| Metric | Value | Target |
|---|---|---|
| Single-step PSNR | **26.75 dB** | ≥ 22 dB ✅ |
| Single-step SSIM | **0.840** | — |
| VQ-VAE PSNR | **31.12 dB** | ≥ 28 dB ✅ |
| Codebook utilization | **100%** | ≥ 80% ✅ |

### Rollout Quality Over Time

Prediction quality is strong for the first few steps, then drops as errors accumulate — the classic **compounding-error problem** in autoregressive generation. PSNR (blue) and SSIM (green) degrade over 50 rollout steps, with the model producing recognizable frames for ~3–5 steps before significant quality degradation.

### ⚠️ Honest Limitations

- **Rollout quality:** Degrades after ~5 autoregressive steps (PSNR drops from 27 → 14 dB by step 10). Errors compound when fed back as context.
- **Action conditioning:** Weak (L2 distance 0.064 between different-action predictions). The model predicts plausible next frames but doesn't strongly differentiate between actions at 80K steps (53% of planned budget).

---

## Interactive Demo

A **Gradio demo** lets you step through CoinRun one frame at a time — pick an action, and the model imagines the next frame.

---

## Technical Pipeline

The project is structured in 3 training phases:

1. **VQ-VAE Tokenizer** (50K steps, ~2h on T4) — Learns a 512-entry codebook for frame compression
2. **Flow Matching Dynamics Model** (80K steps, ~8h on A100) — Learns to predict next-frame velocity fields
3. **Evaluation Suite** — PSNR, SSIM, action differentiation metrics, autoregressive rollout analysis

### Test Coverage

**144 tests passing** — covering every component from ResBlocks to the full evaluation pipeline.

---

## References

Draws ideas from:
- **VQ-VAE** (van den Oord et al., 2017)
- **Flow Matching** (Lipman et al., 2023)
- **DIAMOND** (Alonso et al., 2024) — conditioning design
- **GameNGen** (Valevski et al., 2024) — noise augmentation

The architecture and analysis are original — not a reimplementation of any specific paper.

---

*Built from scratch in PyTorch · 42M parameters · 144 tests passing · MIT License*
