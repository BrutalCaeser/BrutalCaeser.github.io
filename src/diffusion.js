// Diffusion hero: a field of noise particles that *denoises* into the name,
// holds, then re-diffuses — the research as the centerpiece.

export function initDiffusion(canvas, { name = 'YASHVARDHAN\nGUPTA' } = {}) {
  const ctx = canvas.getContext('2d', { alpha: true });
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W, H, dpr, particles = [], targets = [], pointer = { x: -9999, y: -9999 };

  const ACCENT = [139, 124, 246];   // violet
  const ACCENT2 = [232, 164, 77];   // amber
  const lerp = (a, b, t) => a + (b - a) * t;

  // Sample target points from the rendered name.
  function buildTargets() {
    const off = document.createElement('canvas');
    off.width = W; off.height = H;
    const o = off.getContext('2d');
    o.fillStyle = '#fff';
    o.textAlign = 'center';
    o.textBaseline = 'middle';
    const lines = name.split('\n');
    const base = Math.min(W, H * 1.7);
    let size = base / 9;
    size = Math.min(size, W / 7.2);
    o.font = `600 ${size}px Fraunces, Georgia, serif`;
    const lh = size * 1.0;
    const cy = H / 2 - ((lines.length - 1) * lh) / 2;
    lines.forEach((ln, i) => o.fillText(ln, W / 2, cy + i * lh));

    const img = o.getImageData(0, 0, W, H).data;
    const pts = [];
    const gap = Math.max(4, Math.round(W / 360)); // density
    for (let y = 0; y < H; y += gap) {
      for (let x = 0; x < W; x += gap) {
        if (img[(y * W + x) * 4 + 3] > 128) {
          pts.push({ x: x + (Math.random() - 0.5) * gap, y: y + (Math.random() - 0.5) * gap });
        }
      }
    }
    return pts;
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    targets = buildTargets();
    seed();
  }

  function seed() {
    const n = targets.length;
    particles = new Array(n);
    for (let i = 0; i < n; i++) {
      const t = targets[i];
      particles[i] = {
        x: Math.random() * W, y: Math.random() * H,
        tx: t.x, ty: t.y,
        vx: 0, vy: 0,
        s: Math.random() * 1.4 + 0.5,
        mix: Math.random(),
      };
    }
  }

  // Phase machine: diffuse(noise) -> converge -> hold -> disperse -> repeat
  let phase = 'converge', phaseStart = performance.now(), coherence = 0;
  const DUR = { converge: 3200, hold: 2600, disperse: 1500, scatter: 1200 };

  function step(now) {
    const elapsed = now - phaseStart;
    if (phase === 'converge') { coherence = ease(Math.min(1, elapsed / DUR.converge)); if (elapsed > DUR.converge) setPhase('hold', now); }
    else if (phase === 'hold') { coherence = 1; if (elapsed > DUR.hold) setPhase('disperse', now); }
    else if (phase === 'disperse') { coherence = 1 - ease(Math.min(1, elapsed / DUR.disperse)); if (elapsed > DUR.disperse) setPhase('scatter', now); }
    else if (phase === 'scatter') { coherence = 0; if (elapsed > DUR.scatter) setPhase('converge', now); }

    ctx.clearRect(0, 0, W, H);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      // target = name point; noise = drift. Blend by coherence.
      const noiseX = p.x + Math.sin((now / 1400) + i) * 0.6 + (Math.random() - 0.5) * (1 - coherence) * 6;
      const noiseY = p.y + Math.cos((now / 1600) + i * 0.7) * 0.6 + (Math.random() - 0.5) * (1 - coherence) * 6;
      const goalX = lerp(noiseX, p.tx, coherence);
      const goalY = lerp(noiseY, p.ty, coherence);

      // pointer repulsion — feels alive
      const dx = p.x - pointer.x, dy = p.y - pointer.y;
      const d2 = dx * dx + dy * dy;
      let px = goalX, py = goalY;
      if (d2 < 14000) { const f = (14000 - d2) / 14000; px += dx * f * 0.25; py += dy * f * 0.25; }

      p.vx = lerp(p.vx, (px - p.x) * 0.12, 0.4);
      p.vy = lerp(p.vy, (py - p.y) * 0.12, 0.4);
      p.x += p.vx; p.y += p.vy;

      const a = 0.25 + coherence * 0.75;
      const c = coherence > 0.5 ? ACCENT : ACCENT2;
      const cc = [
        Math.round(lerp(c[0], 243, coherence * 0.6)),
        Math.round(lerp(c[1], 241, coherence * 0.6)),
        Math.round(lerp(c[2], 234, coherence * 0.6)),
      ];
      ctx.fillStyle = `rgba(${cc[0]},${cc[1]},${cc[2]},${a})`;
      const r = p.s * (0.7 + coherence * 0.6);
      ctx.fillRect(p.x, p.y, r, r);
    }
    raf = requestAnimationFrame(step);
  }

  let onHint = null;
  function setPhase(next, now) {
    phase = next; phaseStart = now;
    if (onHint) onHint(next);
  }
  const ease = (t) => 1 - Math.pow(1 - t, 3);

  let raf;
  function start() {
    if (reduced) { drawStatic(); return; }
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(step);
  }
  function drawStatic() {
    coherence = 1;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(243,241,234,0.9)';
    for (const t of targets) ctx.fillRect(t.x, t.y, 1.6, 1.6);
  }

  window.addEventListener('resize', debounce(resize, 200));
  window.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    pointer.x = e.clientX - r.left; pointer.y = e.clientY - r.top;
  });
  window.addEventListener('pointerleave', () => { pointer.x = -9999; pointer.y = -9999; });

  resize();
  start();
  return { onPhase(fn) { onHint = fn; } };
}

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
