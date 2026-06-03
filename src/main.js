import './style.css';
import Lenis from 'lenis';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { initDiffusion } from './diffusion.js';
import { initPortrait } from './portrait.js';
import { profile, arc, education, now, projects, writing, closingQuote } from './data.js';

gsap.registerPlugin(ScrollTrigger);

const $ = (s, r = document) => r.querySelector(s);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------------------------------------------------------------
   1. Render content from data
--------------------------------------------------------------- */
function render() {
  // hero tagline words
  $('#heroTag').innerHTML = profile.tagline.map((t) => `<span>${t}</span>`).join('');

  // education
  $('#education').innerHTML = education.map((e) => `
    <div class="edurow">
      <div class="edurow__deg">${e.degree}</div>
      <div class="edurow__org">${e.org} <span class="edurow__loc">· ${e.loc}</span></div>
      <div class="edurow__time">${e.time}</div>
      <div class="edurow__note">${e.note}</div>
    </div>`).join('');

  // arc trail
  $('#arc').innerHTML = arc
    .map((a, i) => `<span class="arc__node">${a}</span>${i < arc.length - 1 ? '<span class="arc__sep">→</span>' : ''}`)
    .join('');

  // work list
  $('#worklist').innerHTML = projects.map((p, i) => {
    const idx = String(i + 1).padStart(2, '0');
    const tags = p.tags.map((t) => `<span class="tag">${t}</span>`).join('');
    const isLink = !!p.link;
    const tag = isLink ? 'a' : 'div';
    const href = isLink ? ` href="${p.link}" target="_blank" rel="noopener"` : '';
    return `<li class="workitem" data-i="${i}">
      <${tag}${href} class="workitem__row" data-row>
        <span class="workitem__idx">${idx}</span>
        <span class="workitem__title">${p.title}</span>
        <span class="workitem__meta">
          <span class="workitem__year">${p.year}</span>
          <span class="workitem__arrow">↗</span>
        </span>
      </${tag}>
      <div class="workitem__panel">
        <div class="workitem__inner">
          <span></span>
          <div>
            <p class="workitem__blurb">${p.blurb}</p>
            <div class="workitem__tags">${tags}</div>
          </div>
        </div>
      </div>
    </li>`;
  }).join('');

  // writing list
  $('#writinglist').innerHTML = writing.map((w) => `
    <a class="writeitem" href="/writing/${w.slug}/">
      <div>
        <div class="writeitem__t">${w.title}</div>
        <div class="writeitem__s">${w.sub}</div>
      </div>
      <div class="writeitem__d">${w.date} ↗</div>
    </a>`).join('');

  // now list
  $('#nowlist').innerHTML = now.map((n) => `
    <li class="nowitem">
      <span class="nowitem__k">${n.k}</span>
      <span class="nowitem__v">${n.v}</span>
    </li>`).join('');

  // contact
  $('#contactBig').href = `mailto:${profile.email}`;
  $('#socials').innerHTML = profile.socials
    .map((s) => `<a class="social" href="${s.url}" target="_blank" rel="noopener">${s.label} ↗</a>`)
    .join('') + `<a class="social" href="mailto:${profile.email}">Email ↗</a>`;
  $('#closing').textContent = closingQuote;
  $('#year').textContent = new Date().getFullYear();
}

/* ---------------------------------------------------------------
   2. Smooth scroll (Lenis) wired into GSAP
--------------------------------------------------------------- */
function smoothScroll() {
  if (reduced) return null;
  const lenis = new Lenis({ duration: 1.15, easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)) });
  lenis.on('scroll', ScrollTrigger.update);
  gsap.ticker.add((time) => lenis.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);

  document.querySelectorAll('[data-link]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href');
      if (id && id.startsWith('#')) { e.preventDefault(); lenis.scrollTo(id, { offset: 0 }); }
    });
  });
  return lenis;
}

/* ---------------------------------------------------------------
   3. Hero choreography
--------------------------------------------------------------- */
function hero() {
  const canvas = $('#diffusion');
  const ctrl = initDiffusion(canvas, { name: 'YASHVARDHAN\nGUPTA' });
  const hints = {
    converge: 'denoising → signal', hold: 'Yashvardhan Gupta',
    disperse: 'returning to noise', scatter: 'sampling…',
  };
  const hint = $('#heroHint');
  ctrl.onPhase((ph) => {
    if (!hint) return;
    gsap.to(hint, { opacity: 0, duration: 0.3, onComplete: () => {
      hint.textContent = hints[ph] || '';
      gsap.to(hint, { opacity: 1, duration: 0.4 });
    }});
  });

  // tagline words rise in
  gsap.to('.hero__tag span', {
    opacity: 1, y: 0, duration: 1, stagger: 0.12, delay: 1.1, ease: 'power3.out',
  });
}

/* ---------------------------------------------------------------
   4. Scroll reveals
--------------------------------------------------------------- */
function reveals() {
  if (reduced) return;

  // split the big statement into lines for a staggered wipe
  const stmt = $('.reveal-lines');
  if (stmt) {
    const words = stmt.innerHTML;
    stmt.innerHTML = `<span class="line-mask"><span class="line-inner">${words}</span></span>`;
    gsap.set('.line-inner', { yPercent: 110 });
    gsap.to('.line-inner', {
      yPercent: 0, duration: 1.1, ease: 'power4.out',
      scrollTrigger: { trigger: stmt, start: 'top 80%' },
    });
  }

  gsap.utils.toArray('.reveal').forEach((el) => {
    gsap.from(el, {
      y: 40, opacity: 0, duration: 1, ease: 'power3.out',
      scrollTrigger: { trigger: el, start: 'top 88%' },
    });
  });

  // arc nodes pop in sequence
  gsap.from('.arc__node, .arc__sep', {
    y: 16, opacity: 0, duration: 0.5, stagger: 0.06, ease: 'back.out(1.6)',
    scrollTrigger: { trigger: '#arc', start: 'top 85%' },
  });

  // work + writing items rise
  ['.workitem', '.writeitem', '.nowitem'].forEach((sel) => {
    gsap.utils.toArray(sel).forEach((el, i) => {
      gsap.from(el, {
        y: 50, opacity: 0, duration: 0.9, ease: 'power3.out', delay: (i % 4) * 0.05,
        scrollTrigger: { trigger: el, start: 'top 90%' },
      });
    });
  });

  // section titles
  gsap.utils.toArray('.section-title, .contact__big, .closing').forEach((el) => {
    gsap.from(el, {
      y: 30, opacity: 0, duration: 1, ease: 'power3.out',
      scrollTrigger: { trigger: el, start: 'top 90%' },
    });
  });
}

/* ---------------------------------------------------------------
   5. Work list expand-on-tap (mobile-friendly accordion)
--------------------------------------------------------------- */
function workInteractions() {
  document.querySelectorAll('.workitem').forEach((item) => {
    const row = item.querySelector('[data-row]');
    // hover opens panel on desktop
    if (matchMedia('(hover: hover)').matches) {
      item.addEventListener('mouseenter', () => item.classList.add('open'));
      item.addEventListener('mouseleave', () => item.classList.remove('open'));
    } else {
      row.addEventListener('click', (e) => {
        if (item.classList.contains('open')) return; // let the link through second tap
        item.classList.add('open');
      });
    }
  });
}

/* ---------------------------------------------------------------
   6. Custom cursor + magnetic interactives
--------------------------------------------------------------- */
function cursor() {
  if (!matchMedia('(hover: hover)').matches) return;
  const dot = $('.cursor');
  let x = innerWidth / 2, y = innerHeight / 2, cx = x, cy = y;
  window.addEventListener('pointermove', (e) => { x = e.clientX; y = e.clientY; });
  gsap.ticker.add(() => {
    cx += (x - cx) * 0.18; cy += (y - cy) * 0.18;
    dot.style.transform = `translate(${cx}px, ${cy}px) translate(-50%, -50%)`;
  });
  const hot = 'a, button, .workitem__row, .writeitem, .social';
  document.querySelectorAll(hot).forEach((el) => {
    el.addEventListener('mouseenter', () => dot.classList.add('is-hover'));
    el.addEventListener('mouseleave', () => dot.classList.remove('is-hover'));
  });

  // magnetic
  document.querySelectorAll('[data-magnetic]').forEach((el) => {
    el.addEventListener('mousemove', (e) => {
      const r = el.getBoundingClientRect();
      gsap.to(el, { x: (e.clientX - (r.left + r.width / 2)) * 0.3, y: (e.clientY - (r.top + r.height / 2)) * 0.3, duration: 0.4 });
    });
    el.addEventListener('mouseleave', () => gsap.to(el, { x: 0, y: 0, duration: 0.5, ease: 'elastic.out(1, 0.4)' }));
  });
}

/* ---------------------------------------------------------------
   7. Live local clock in the nav
--------------------------------------------------------------- */
function clock() {
  const el = $('#clock'), d = $('#nowdate');
  const tick = () => {
    const t = new Date();
    if (el) el.textContent = t.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    if (d) d.textContent = t.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };
  tick(); setInterval(tick, 1000);
}

/* ---------------------------------------------------------------
   8. Film-grain texture
--------------------------------------------------------------- */
function grain() {
  const c = $('#grain'), ctx = c.getContext('2d');
  const size = 220;
  c.width = size; c.height = size;
  function draw() {
    const img = ctx.createImageData(size, size);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = Math.random() * 255;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  draw();
  // refresh occasionally so it shimmers, but cheaply
  if (!reduced) setInterval(draw, 130);
}

/* ---------------------------------------------------------------
   boot
--------------------------------------------------------------- */
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
render();
hero();
initPortrait(document.getElementById('portrait'), { src: '/portrait.webp', focusX: 0.27, focusY: 0.64, zoom: 2.0 });
smoothScroll();
reveals();
workInteractions();
cursor();
clock();
grain();
