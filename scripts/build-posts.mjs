// Renders the markdown posts in /posts into self-contained reading pages under
// /public/writing/<slug>/index.html — math (KaTeX) and code (highlight.js) via CDN.
// Math is protected from the markdown parser, then restored, so KaTeX sees raw LaTeX.

import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dir, '..');
const SRC = join(ROOT, 'posts');
const OUT = join(ROOT, 'public', 'writing');

function frontmatter(raw) {
  const m = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  const meta = {};
  let body = raw;
  if (m) {
    body = raw.slice(m[0].length);
    for (const line of m[1].split('\n')) {
      const mm = line.match(/^([A-Za-z_]+):\s*(.*)$/);
      if (mm) meta[mm[1]] = mm[2].replace(/^["']|["']$/g, '');
    }
  }
  return { meta, body };
}

function protectMath(md) {
  const store = [];
  // display $$...$$ first, then inline $...$
  md = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, x) => { store.push(`\\[${x}\\]`); return `@@MATH${store.length - 1}@@`; });
  md = md.replace(/(?<!\\)\$([^\n$]+?)\$/g, (_, x) => { store.push(`\\(${x}\\)`); return `@@MATH${store.length - 1}@@`; });
  return { md, store };
}
const restoreMath = (html, store) => html.replace(/@@MATH(\d+)@@/g, (_, i) => store[+i]);

function fmtDate(d) {
  if (!d) return '';
  const dt = new Date(d);
  return isNaN(dt) ? d : dt.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
}

const page = ({ title, desc, date, content }) => `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>${title} — Yashvardhan Gupta</title>
<meta name="description" content="${(desc || '').replace(/"/g, '&quot;')}"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css"/>
<style>
:root{--bg:#0b0b0f;--ink:#f3f1ea;--muted:#9a9aa6;--faint:#56565f;--line:rgba(243,241,234,.10);--accent:#8b7cf6;}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:'Inter',system-ui,sans-serif;line-height:1.75;font-size:1.06rem;-webkit-font-smoothing:antialiased}
::selection{background:var(--accent);color:#0b0b0f}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.top{position:sticky;top:0;display:flex;justify-content:space-between;padding:1.2rem clamp(1.2rem,5vw,3rem);font-family:'JetBrains Mono',monospace;font-size:.8rem;background:linear-gradient(var(--bg),rgba(11,11,15,.6) 70%,transparent);backdrop-filter:blur(6px);z-index:10}
.top a{color:var(--ink)}
.wrap{max-width:720px;margin:0 auto;padding:clamp(2rem,7vw,5rem) clamp(1.2rem,5vw,2rem) 7rem}
.kicker{font-family:'JetBrains Mono',monospace;font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
h1{font-family:'Fraunces',serif;font-weight:400;font-size:clamp(2rem,6vw,3.4rem);line-height:1.08;letter-spacing:-.02em;margin:1rem 0 .8rem}
.lead{color:var(--muted);font-size:1.15rem;margin-bottom:1.4rem}
.meta{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--faint);padding-bottom:2rem;border-bottom:1px solid var(--line);margin-bottom:2.5rem}
article h2{font-family:'Fraunces',serif;font-weight:500;font-size:1.9rem;letter-spacing:-.01em;margin:2.8rem 0 1rem;line-height:1.15}
article h3{font-family:'Fraunces',serif;font-weight:500;font-size:1.4rem;margin:2rem 0 .8rem}
article p{margin:1.1rem 0;color:#dcdad3}
article ul,article ol{margin:1.1rem 0 1.1rem 1.4rem;color:#dcdad3}
article li{margin:.4rem 0}
article strong{color:var(--ink)}
article blockquote{border-left:2px solid var(--accent);padding-left:1.2rem;margin:1.6rem 0;color:var(--muted);font-style:italic;font-family:'Fraunces',serif;font-size:1.15rem}
article img{max-width:100%;border-radius:10px;margin:1.5rem 0}
article hr{border:none;border-top:1px solid var(--line);margin:2.5rem 0}
code{font-family:'JetBrains Mono',monospace;font-size:.86em;background:rgba(243,241,234,.07);padding:.15em .4em;border-radius:5px}
pre{background:#13131a;border:1px solid var(--line);border-radius:12px;padding:1.2rem;overflow-x:auto;margin:1.5rem 0}
pre code{background:none;padding:0;font-size:.85rem;line-height:1.6}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:.4rem 0}
table{width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:.95rem}
th,td{border:1px solid var(--line);padding:.6rem .9rem;text-align:left}
th{background:rgba(243,241,234,.04);font-family:'JetBrains Mono',monospace;font-size:.8rem}
.backhome{display:inline-block;margin-top:3rem;font-family:'JetBrains Mono',monospace;font-size:.85rem;color:var(--ink);border:1px solid var(--line);padding:.7em 1.3em;border-radius:999px}
.backhome:hover{border-color:var(--accent);color:var(--accent);text-decoration:none}
</style>
</head>
<body>
<div class="top"><a href="/">◉ Yashvardhan Gupta</a><a href="/#writing">all writing ↗</a></div>
<div class="wrap">
<p class="kicker">Essay</p>
<h1>${title}</h1>
${desc ? `<p class="lead">${desc}</p>` : ''}
<p class="meta">${fmtDate(date)} · Yashvardhan Gupta</p>
<article>${content}</article>
<a class="backhome" href="/">← back home</a>
</div>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded',function(){
  if(window.hljs){document.querySelectorAll('pre code').forEach(b=>hljs.highlightElement(b));}
});
window.addEventListener('load',function(){
  if(window.renderMathInElement){renderMathInElement(document.body,{delimiters:[
    {left:'\\\\[',right:'\\\\]',display:true},{left:'\\\\(',right:'\\\\)',display:false}
  ]});}
});
</script>
</body>
</html>`;

function build() {
  const files = readdirSync(SRC).filter((f) => f.endsWith('.md'));
  marked.setOptions({ gfm: true, breaks: false });
  for (const f of files) {
    const slug = f.replace(/\.md$/, '');
    const raw = readFileSync(join(SRC, f), 'utf8');
    const { meta, body } = frontmatter(raw);
    const { md, store } = protectMath(body);
    let html = marked.parse(md);
    html = restoreMath(html, store);
    const out = page({
      title: meta.title || slug,
      desc: meta.description || meta.summary || '',
      date: meta.date || '',
      content: html,
    });
    const dir = join(OUT, slug);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'index.html'), out);
    console.log('  ✓ /writing/' + slug + '/');
  }
  console.log('Built ' + files.length + ' writing pages.');
}

build();
