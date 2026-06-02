# brutalcaeser.github.io

Personal site of **Yashvardhan Gupta** — AI researcher & engineer.
A hand-built, editorial-minimal site with a diffusion-denoising hero.

## Stack
- **Vite** (vanilla, no framework) · **GSAP** + ScrollTrigger · **Lenis** smooth scroll
- Diffusion hero: custom canvas particle system (`src/diffusion.js`)
- Writing pages: markdown in `posts/` rendered at build time with **KaTeX** math + highlight.js (`scripts/build-posts.mjs`)

## Develop
```bash
npm install
npm run dev      # builds writing pages, then starts Vite
npm run build    # writing pages + production build → dist/
```

## Editing content
All site copy lives in **`src/data.js`** (profile, projects, writing, "now", quote).
Blog posts are markdown in **`posts/`** — add a `.md` file and it becomes `/writing/<slug>/`.

## Deploy
Pushing to `main` triggers `.github/workflows/deploy.yml` → GitHub Pages.
