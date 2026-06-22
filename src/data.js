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
  'Python', 'DSA', 'Data Science', 'Machine Learning', 'Deep Learning',
  'CV', 'NLP', 'LLMs', 'Diffusion', 'World Models', 'Physical AI',
];

// Education — shown in the short-version section.
export const education = [
  {
    degree: 'M.S. Artificial Intelligence',
    org: 'Northeastern University',
    loc: 'Silicon Valley',
    time: '2025 — 2027',
    note: '4.0 / 4.0 GPA',
  },
  {
    degree: 'B.Tech, Mechanical Engineering',
    org: 'Delhi Technological University',
    loc: 'New Delhi',
    time: '2019 — 2023',
    note: 'IEEE-published',
  },
];

export const now = [
  { k: 'Building', v: 'Deterministic agentic AI on Claude + MCP — AI/GenAI co-op at NovasIQ.' },
  { k: 'Researching', v: 'Flow-map language model inversion & the block-size Pareto frontier.' },
  { k: 'Reading', v: 'To Kill a Mockingbird — then Russell’s The Problems of Philosophy.' },
  { k: 'Away from the desk', v: 'Tennis, chess, and arguing with books.' },
];

// Toolkit — grouped by layer, research first. (No skill bars; tools are means, not scores.)
export const skills = [
  { group: 'Research & methods', items: ['Diffusion models', 'Flow matching', 'Discrete diffusion', 'World models', 'Reinforcement learning', 'Self-supervised learning'] },
  { group: 'ML engineering', items: ['PyTorch', 'JAX / Flax', 'Hugging Face', 'CUDA', 'ONNX', 'multi-GPU / SLURM'] },
  { group: 'Systems & deployment', items: ['Docker', 'FastAPI', 'CI/CD', 'AWS', 'GCP', 'Linux'] },
  { group: 'Languages', items: ['Python', 'SQL', 'Bash', 'Java'] },
];

// Selected work. `link` omitted = no outbound link (shows as a read-only entry).
export const projects = [
  {
    title: 'Reinforced Diffusion LLMs',
    year: '2026',
    tags: ['reinforcement learning', 'diffu-GRPO'],
    blurb: 'A single-GPU study of diffu-GRPO, the reinforcement-learning recipe behind reasoning in diffusion LLMs. The catch is that these models can’t compute their own sequence probability like a normal LLM, so I validated the cheap estimator that makes RL possible, then showed it lifts held-out reasoning by a modest ~4 points, where a faithful run would cost 24 GPU-days.',
    link: 'https://github.com/BrutalCaeser/reinforcing_dLLMs',
  },
  {
    title: 'Block-Size Pareto Frontier',
    year: '2026',
    tags: ['block diffusion', 'throughput'],
    blurb: 'Why does a leading diffusion language model run at “block size 32”? I mapped the full speed-vs-quality trade-off and found generation throughput peaks exactly at 32 — reproducing, on the real code path, a sweet spot that was never published.',
    link: 'https://github.com/BrutalCaeser/block-diffusion-pareto',
  },
  {
    title: 'Phantom Gradients',
    year: '2026',
    tags: ['self-supervised', 'representation collapse'],
    blurb: 'When a model’s useful features live in far fewer dimensions than it has room for, training fights noise in the empty ones. I built a testbed showing a simple coherence-guided trick recovers the real structure — without being told how many dimensions to look for — beating even a method that is.',
    link: 'https://github.com/BrutalCaeser/phantom-gradients',
  },
  {
    title: 'Flow Map Language Models',
    year: '2026',
    tags: ['flow matching', 'one-step gen'],
    blurb: 'A new way to generate text in a single step instead of dozens. I reproduced the method and uncovered a quality pattern the original authors had missed.',
    link: 'https://github.com/BrutalCaeser/Flow-Language-Model',
  },
  {
    title: 'SIGReg World Model',
    year: '2026',
    tags: ['self-supervised', 'proof'],
    blurb: 'Can a popular trick stop AI video models from “collapsing” into useless features? After 77 experiments, the answer was no — and I proved mathematically why.',
    link: 'https://github.com/BrutalCaeser/spatial-jepa-sigreg',
  },
  {
    title: 'MicroDiffusion LM',
    year: '2026',
    tags: ['diffusion', '124M study'],
    blurb: 'A diffusion-based language model built from scratch, plus a 124M-parameter study showing the simplest training recipe quietly beats the fancier one.',
    link: 'https://github.com/BrutalCaeser/microDLM',
  },
  {
    title: 'MiniGenie',
    year: '2026',
    tags: ['video', 'world model'],
    blurb: 'A small video “world model” that learns to imagine the next frames of a game — built entirely from scratch in PyTorch.',
    link: 'https://github.com/BrutalCaeser/minigenie',
  },
  {
    title: 'Once Upon Earth',
    year: '2026',
    tags: ['full-stack', 'live product'],
    blurb: 'A 3D globe of the world’s history: spin it, tap a country, and read a source-grounded story. Stories are either hand-curated or generated on demand from real Wikipedia sources, with a gate that refuses to make things up. Shipped and live.',
    link: 'https://onceuponearth.co',
    image: '/img/oue-landing.webp',
    imageAlt: 'Once Upon Earth landing page — a night-lit globe under the title',
    imageLabel: 'onceuponearth.co',
  },
  {
    title: 'OpsGraph',
    year: '2026',
    tags: ['agentic-RAG', 'knowledge graph'],
    blurb: 'An enterprise three-way-match system: ingest purchase orders, receipts, and invoices into a Neo4j knowledge graph, then let an agentic-RAG layer answer multi-hop questions and surface the financial discrepancies. The arithmetic is deterministic Python; the model only routes and explains.',
  },
  {
    title: 'Jarvis — macOS AI Agent',
    year: '2026',
    tags: ['agents', 'safety'],
    blurb: 'A personal AI assistant for macOS that can safely use my terminal, browser, and files to actually get things done — with guardrails at every step.',
    link: 'https://github.com/BrutalCaeser/jarvis-ai',
  },
  {
    title: 'Hunter — Job Application Agent',
    year: '2026',
    tags: ['automation', '24/7'],
    blurb: 'An AI agent that finds internship listings and fills out applications on its own, running around the clock.',
    link: 'https://github.com/BrutalCaeser/Job_Hunter',
  },
  {
    title: 'Read My Lips',
    year: '2025',
    tags: ['multimodal', 'assistive'],
    blurb: 'A tool that reads lips from video and speaks for you through a lifelike avatar — built for people who can’t speak, with privacy kept on-device.',
    link: 'https://github.com/BrutalCaeser/read_my_lips',
  },
  {
    title: 'AI-Enabled Vehicle Headlamps',
    year: '2023',
    tags: ['computer vision', 'B.Tech'],
    blurb: 'Smart headlights that light up the road but dim automatically to avoid blinding oncoming drivers. My first real build-from-scratch project.',
  },
];

export const writing = [
  {
    slug: 'teaching-a-diffusion-language-model',
    title: 'Teaching a Diffusion Language Model to Reason with RL',
    sub: 'diffu-GRPO from scratch on one GPU, and the trick that makes it train',
    date: 'Jun 2026',
  },
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
  'Knowledge is abundant now because of AI. As humans, now more than ever, we need to find the problems that truly matter — simply because we can, and because we owe it to ourselves, and to the human race, to use AI for the betterment of human civilisation.';
