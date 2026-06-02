// Single source of truth for the page content. Edit here, not in markup.

export const profile = {
  name: 'Yashvardhan Gupta',
  role: 'AI Researcher & Engineer',
  tagline: ['Diffusion models', 'World models', 'Physical intelligence'],
  email: 'gupta.yashv@northeastern.edu',
  socials: [
    { label: 'GitHub', url: 'https://github.com/BrutalCaeser' },
    { label: 'LinkedIn', url: 'https://www.linkedin.com/in/yashvardhangupta37/' },
    { label: 'Scholar', url: 'https://scholar.google.com/citations?user=yashvardhangupta' },
    { label: 'X', url: 'https://x.com/BrutalCaeser' },
  ],
};

// The journey, animated as a trail.
export const arc = [
  'adaptive optics', 'stereo vision', 'ML engineering',
  'diffusion', 'world models', 'language', 'physical AI',
];

export const now = [
  { k: 'Building', v: 'Deterministic agentic AI on Claude + MCP — AI/GenAI co-op at NovasIQ.' },
  { k: 'Researching', v: 'Flow-map language model inversion & the block-size Pareto frontier.' },
  { k: 'Reading', v: 'To Kill a Mockingbird — then Russell’s The Problems of Philosophy.' },
  { k: 'Away from the desk', v: 'Tennis, chess, and arguing with books.' },
];

// Selected work. `link` omitted = no outbound link (shows as a read-only entry).
export const projects = [
  {
    title: 'Flow Map Language Models',
    year: '2026',
    tags: ['flow matching', 'one-step gen'],
    blurb: 'Reproduced and extended FMLM — a continuous-flow approach to one-step text generation. Found a novel non-monotonic step-quality curve the original paper never reported.',
    link: 'https://github.com/BrutalCaeser/Flow-Language-Model',
  },
  {
    title: 'SIGReg World Model',
    year: '2026',
    tags: ['SSL', 'proof', 'collapse'],
    blurb: 'A rigorous test of whether distributional regularization can stop representational collapse in self-supervised video features — 77 commits, every approach failed, and a mathematical proof of why it can’t. That was the result.',
    link: 'https://github.com/BrutalCaeser/spatial-jepa-sigreg',
  },
  {
    title: 'MicroDiffusion LM',
    year: '2026',
    tags: ['discrete diffusion', '124M study'],
    blurb: 'From-scratch GPT-vs-diffusion on Tiny Shakespeare, then a 124M-parameter scaling study on FineWeb-Edu showing uniform masking beats entropy-weighted across every noise schedule.',
    link: 'https://github.com/BrutalCaeser/microDLM',
  },
  {
    title: 'MiniGenie',
    year: '2026',
    tags: ['video', 'flow matching'],
    blurb: 'A flow-matching video world model that learns to imagine game frames — built entirely from scratch in PyTorch around a 42M-parameter U-Net.',
    link: 'https://github.com/BrutalCaeser/minigenie',
  },
  {
    title: 'Jarvis — macOS AI Agent',
    year: '2026',
    tags: ['agents', 'A2A', 'safety'],
    blurb: 'An autonomous agent that orchestrates terminal, browser, and filesystem through LLMs, with a 4-tier safety system and Agent-to-Agent protocol support.',
    link: 'https://github.com/BrutalCaeser/jarvis-ai',
  },
  {
    title: 'Hunter — Job Application Agent',
    year: '2026',
    tags: ['browser automation', '24/7'],
    blurb: 'An LLM-driven browser agent that discovers internship roles, fills applications deterministically, and runs around the clock on macOS.',
    link: 'https://github.com/BrutalCaeser/Job_Hunter',
  },
  {
    title: 'Read My Lips',
    year: '2025',
    tags: ['multimodal', 'assistive'],
    blurb: 'A privacy-preserving, end-to-end visual-speech-to-avatar interface — lip-read in real time, projected through a synthesized talking avatar.',
    link: 'https://github.com/BrutalCaeser/read_my_lips',
  },
  {
    title: 'AI-Enabled Vehicle Headlamps',
    year: '2023',
    tags: ['CV', 'embedded', 'B.Tech'],
    blurb: 'An adaptive LED-matrix system using real-time object detection to light the road while sparing oncoming drivers from glare. Where the build-from-scratch habit started.',
  },
];

export const writing = [
  {
    slug: 'from-noise-to-shakespeare',
    title: 'From Noise to Shakespeare',
    sub: 'Building a diffusion language model from scratch',
    date: 'Mar 2026',
  },
  {
    slug: 'gmm-mle-em',
    title: 'From K-Means to Gaussian Mixtures',
    sub: 'The math, the intuition, and the EM algorithm',
    date: 'Feb 2026',
  },
  {
    slug: 'lagrangian-kkt-svm',
    title: 'Lagrangians, KKT & SVMs',
    sub: 'A mathematical journey to the margin',
    date: 'Jan 2025',
  },
];

export const closingQuote =
  'Knowledge is abundant now because of AI. As humans, more than ever, we need to find and focus our time on problems that really matter — simply because we can, and we owe it to ourselves.';
