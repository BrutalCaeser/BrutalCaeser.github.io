---
title: "Teaching a Diffusion Language Model to Reason with RL"
date: 2026-06-15
description: "Reproducing the d1 recipe on a single GPU, and the simple trick that makes a 'bad' estimator work."
---
*Autoregressive LLMs get reinforcement learning almost for free. Diffusion language models, the
kind behind LLaDA, Dream, and Mercury, do not, because they have no easy way to say how likely
their own answer was. This is the story of reproducing the d1 recipe on one GPU, and of the one
subtle idea that makes the whole thing train.*

---

Most of the chatbots you have used are **autoregressive**: they write one word after another,
left to right, like a person typing a sentence. A newer family, **diffusion language models**,
writes differently. They start with the whole answer blanked out and fill in the blanks in
parallel, in no fixed order, refining over a few passes. That is why they can be much faster.

The interesting question: can we use reinforcement learning (RL), the technique behind RLHF and
modern reasoning models, to make a diffusion model *reason better*? A 2025 paper called **d1**
(arXiv:2504.12216) showed the answer is yes. This post reproduces that result on a single GPU,
explains the one genuinely tricky part in plain language, and ends with the honest limits of what
one GPU can show.

Two pieces of vocabulary first.

> **What is GRPO?**
> GRPO (Group Relative Policy Optimization) is the RL method we use. The idea is simpler than the
> name. For a given question, the model writes several answers. We score each one, then reward the
> answers that beat the group's average and discourage the ones below it.
>
> Think of **grading on a curve**: instead of an absolute score, each answer is judged against the
> other answers to the *same* question. The group is its own yardstick, so there is no need for a
> separate "judge" model. GRPO comes from the DeepSeekMath paper
> ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)); for a visual walkthrough, this
> [GRPO explainer video](https://youtu.be/xT4jxQUl0X8) is a good 10-minute primer.

> **What is the task? (Countdown)**
> We train and test on **Countdown**, borrowed from the old British game show. You are given a few
> numbers and a target, and you must combine the numbers with `+ - * /` (using each one once) to
> hit the target. Our version uses three numbers.
>
> Example: from **{3, 7, 5}**, reach **26**. Answer: `3 * 7 + 5 = 26`.
>
> Countdown is a perfect RL testbed for one reason: a correct answer is trivial to **check** (just
> do the arithmetic) but takes real searching to **find**. That clean, automatic right-or-wrong
> signal is exactly the "reward" RL needs, with no human and no second model in the loop.

---

## 1. Why RL on a diffusion model is hard

RL methods like GRPO need one number: how likely the model was to produce the answer it produced,
written `p(answer | question)`. For an autoregressive model you get it for free. Since it writes
left to right, the probability of the whole answer is just the chance of each word given the words
before it, multiplied together.

A diffusion model gives you no such thing, and here is the intuition for why.

> An autoregressive model writes a sentence the way you **type** it: one word after another.
> A diffusion model fills in the answer the way you solve a **crossword**: many blank squares at
> once, in whatever order the answers come to you.

For the typed sentence, "how likely was this sentence?" is easy. For the crossword, there are a
staggering number of *orders* in which you could have filled the squares, and to ask "how likely
was this exact finished grid?" you would have to account for all of them. That sum is far too
large to compute.

So `p(answer | question)` is **intractable** for a diffusion model. Every RL method for diffusion
models is, at heart, a different way to *estimate* this number cheaply. How good that estimate is,
how biased and how noisy, decides everything.

---

## 2. d1's estimate: the one-pass shortcut

d1 takes the cheapest possible shortcut:

> Blank out the entire answer, do **one** pass of the model, and add up how surprised it is by each
> correct word.

One forward pass, done. But this shortcut looks broken in two ways:

- **It is pessimistic (biased).** Blanking the whole answer is the hardest possible case: the model
  has to guess every word with no help from the others. So the score it produces is gloomier than
  the truth.
- **It is jumpy (high variance).** The score depends on the random pattern of blanks you happened
  to draw that time.

We checked this shortcut against a slow, careful reference calculation on the real LLaDA-8B model.
The pessimism turned out to be small and roughly constant. And crucially, the cheap shortcut
**ranks** answers in the same order as the careful version:

```
answer                cheap shortcut    careful reference
"Paris" (correct)         -0.41             -0.29
"Berlin" (wrong)          -8.97            -12.16
"pizza" (off-topic)      -14.88            -21.79
"qwx zzf" (gibberish)    -75.69            -45.33

            ranking agreement: perfect
```

Why does ranking matter more than the exact numbers? Because GRPO grades on a curve. It only needs
to know *which* answers in the group are better than the others. Getting the order right is enough.

> It is like judging students with the hardest possible closed-book exam. Everyone scores low, so
> the absolute marks look harsh, but the *ranking* of who understands the material best is still
> correct. That is all the curve needs.

---

## 3. The trick that makes it work: shared noise cancels

Now the real puzzle. That shortcut is genuinely jumpy: redraw the random blanks and the score
swings by a lot. How can training be stable on such a noisy signal?

The key insight is that GRPO never uses the raw score on its own. It compares the **same answer**
before and after a small training step, as a *ratio*. And if you score both with the **same
pattern of blanks**, the noise is identical in both, so it cancels out when you compare them.

> Imagine measuring two people's heights with the same slightly stretchy tape measure. Each
> measurement is a bit off. But the *difference* in their heights is almost exactly right, because
> the very same error is baked into both numbers.

That is precisely what happens here. The raw score is noisy, but the *comparison* GRPO actually
uses is rock-steady:

```
how noisy the raw score is (across random blanks)         very noisy
the before/after comparison, using the same blanks        essentially identical (it cancels)
how much quieter the comparison is than the raw score     about 200 times quieter
```

**This is the heart of it: a biased, noisy estimator is perfectly safe to train with, as long as
you reuse the same blanks when you compare.** It is a textbook variance-reduction idea (reusing the
same randomness for two measurements), and the same principle quietly shows up in other diffusion-RL
methods like VRPO (arXiv:2505.19223) and ESPO (arXiv:2512.03759).

---

## 4. Does it actually learn?

First, the starting point. LLaDA-8B on Countdown, with no RL at all:

```
our baseline    21.5 %  correct
d1's baseline   20.7 %  correct   (the small gap is just luck of the draw)
```

A clean match. The number to beat is about 21%.

Now RL. We train only small "adapter" weights on top of the frozen model (a technique called LoRA),
which is the only reason an 8-billion-parameter model fits on one GPU. Two runs tell the whole story:

```
show it ~250 problems, each once          reward stays flat       (it does not learn)
drill 32 problems over and over           reward climbs 0.25 -> 0.48   (it learns)
```

![Mean reward climbing over 11 passes on 32 fixed Countdown prompts](/blog/reward_climb.png)

*Mean reward per pass over 32 fixed Countdown prompts (LLaDA-8B + LoRA, diffu-GRPO). Flat for about nine passes, then it starts solving them: roughly 16% to 42% correct. This is the training-reward curve on the memorized set; held-out generalization is the next section.*

On the small drilled set, the share of correct answers rose from about 16% to about 42%. So RL
genuinely moves the needle, but only through **repetition**.

> It is the difference between flipping through 250 flashcards once and actually *drilling* a deck
> of 32 until you know them cold. The drilling is what sticks.

This matches what d1 did at full scale: roughly 240,000 problems, seen about 10 times each. On one
GPU you cannot match that breadth, so you trade breadth for depth to see the mechanism clearly.

---

## 5. The honest part: learning vs. truly learning

Here is the honest accounting. That 0.25 to 0.48 climb is on the *same* 32 problems the model
practiced on, so on its own it only proves the machinery works: GRPO can push a diffusion model to
get better at a reward.

So I also tested the trained model on **held-out** problems it had never seen. It scored **25.78%,
up from the 21.48% baseline, a +4.3 point lift.** Encouraging, but I want to be straight about the
statistics: on 256 test problems that gap is only about 1.6 standard errors. It is *suggestive, not
significant*. A real claim needs the same run repeated across several random seeds.

So the fair summary: the mechanism clearly works, and there is a genuine *hint* of generalization
beyond memorization. Pinning it down needs more runs (seeds), and matching d1's headline numbers
needs its scale, 8 GPUs for three days, well beyond a single-GPU project.

---

## 6. What is happening at the frontier (mid-2026)

If there is one sentence to take away about the research field: **it is a race to estimate that
impossible probability well.**

The shortcut d1 uses is the simplest entry. A dozen newer methods trade more compute for a better
estimate: coupled-GRPO, VRPO, GDPO, AGRPO, wd1, SPG, d2, and ESPO, among others. An open framework
called **DARE** (arXiv:2604.04215) put many of them head to head and reached a humbling verdict: no
single method wins everywhere, and the more elaborate estimators can actually become *unstable*
during training.

That instability is the live question right now, and the field is moving monthly. Just in the last
few weeks (read at the summary level):

- **GDSD** (arXiv:2605.29398) throws out the elaborate probability estimate entirely in favor of a
  "copy your own best behavior" approach, reporting double-digit accuracy gains.
- **RLDF** (arXiv:2605.25638) estimates the training signal from the denoising process itself.
- **CAPR** (arXiv:2606.04396) squeezes richer feedback out of the model's own fill-in trace.
- **PAPO** (arXiv:2606.08501) hands out credit step by step instead of only at the end.
- **AGDO** (arXiv:2606.12273) uses the model's attention to decide what to fill in first.

The common thread is that everyone is fighting the same missing-probability problem. The
shared-noise trick from Section 3 explains why d1's *simplest* shortcut is stable. Why the *fancier*
estimates can wobble is exactly what this newest wave is trying to fix.

---

## Takeaways

1. RL works on diffusion language models, and the whole difficulty is that they cannot easily say
   how likely their own answer was.
2. d1's one-pass estimate looks broken (pessimistic and noisy) but ranks answers correctly, and the
   shared-noise trick cancels the noise in the comparison GRPO actually uses. That is why a "bad"
   estimate trains just fine.
3. On a single GPU you can reproduce and validate the mechanism through repetition. True
   generalization is a scale story.
4. The research frontier is the estimation problem itself, and it is moving fast: five or more new
   methods in a single month.

---

## Try it yourself

Every number above is reproducible from the scripts in the repository: the baseline, the estimator
check, the reward tests, and the RL runs.

*A note on sourcing: claims are linked to their arXiv IDs. The newest frontier papers in Section 6
are read at the summary level, so verify any specific detail against the source before relying on
it. The numbers in Sections 2 through 5 are from this project's own runs.*


---

*This post also appears on [Substack](https://yashvardhangupta.substack.com/p/teaching-a-diffusion-language-model). The code and every number are reproducible in the [reinforcing_dLLMs](https://github.com/BrutalCaeser/reinforcing_dLLMs) repo.*
