// Diffusion hero: a field of noise particles that *denoises* into the name,
// holds, then gently re-diffuses — the research as the centerpiece.

export function initDiffusion(canvas, { name = 'YASHVARDHAN\nGUPTA' } = {}) {
  const ctx = canvas.getContext('2d', { alpha: true });
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W, H, dpr, lastW = 0, particles = [], targets = [], pointer = { x: -9999, y: -9999 };

  const ACCENT = [149, 134, 255];   // violet (a touch brighter for legibility)
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

    // Fit the longest line to ~78% of width, but cap by height too.
    const longest = lines.reduce((a, b) => (a.length > b.length ? a : b));
    let size = 220;
    o.font = `700 ${size}px Fraunces, Georgia, serif`;
    const target = W * (W < 640 ? 0.85 : 0.74);
    size = size * (target / o.measureText(longest).width);
    size = Math.min(size, H / (lines.length + 1.4));
    o.font = `700 ${size}px Fraunces, Georgia, serif`;

    const lh = size * 0.96;
    const cy = H / 2 - ((lines.length - 1) * lh) / 2;
    lines.forEach((ln, i) => o.fillText(ln, W / 2, cy + i * lh));

    const img = o.getImageData(0, 0, W, H).data;
    const pts = [];
    // Sampling step: dense enough for crisp edges, sparse enough that the
    // glyphs read as particles forming letters — not a solid slab.
    const gap = Math.max(2, Math.round(W / 420));
    for (let y = 0; y < H; y += gap) {
      for (let x = 0; x < W; x += gap) {
        if (img[(y * W + x) * 4 + 3] > 90) {
          pts.push({ x: x + (Math.random() - 0.5) * gap, y: y + (Math.random() - 0.5) * gap });
        }
      }
    }
    return pts;
  }

  function measure() {
    dpr = Math.min(window.devicePixelRatio || 1, 3);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function rebuild() {
    measure();
    targets = buildTargets();
    seed();
    lastW = W;
  }

  function seed() {
    const n = targets.length;
    particles = new Array(n);
    for (let i = 0; i < n; i++) {
      const t = targets[i];
      particles[i] = {
        x: Math.random() * W, y: Math.random() * H,
        tx: t.x, ty: t.y, vx: 0, vy: 0,
        s: Math.random() * 1.1 + 0.7,
        ph: Math.random() * Math.PI * 2,
      };
    }
  }

  // Phase machine: converge -> hold -> disperse -> scatter -> repeat
  let phase = 'converge', phaseStart = performance.now(), coherence = 0;
  const DUR = { converge: 3000, hold: 7000, disperse: 2200, scatter: 1400 };
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  function step(now) {
    const elapsed = now - phaseStart;
    if (phase === 'converge') { coherence = easeInOut(Math.min(1, elapsed / DUR.converge)); if (elapsed > DUR.converge) setPhase('hold', now); }
    else if (phase === 'hold') { coherence = 1; if (elapsed > DUR.hold) setPhase('disperse', now); }
    else if (phase === 'disperse') { coherence = 1 - easeInOut(Math.min(1, elapsed / DUR.disperse)); if (elapsed > DUR.disperse) setPhase('scatter', now); }
    else if (phase === 'scatter') { coherence = 0; if (elapsed > DUR.scatter) setPhase('converge', now); }

    ctx.clearRect(0, 0, W, H);
    const drift = (1 - coherence);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const noiseX = p.x + Math.sin(now / 1500 + p.ph) * 0.5 + (Math.random() - 0.5) * drift * 5;
      const noiseY = p.y + Math.cos(now / 1700 + p.ph) * 0.5 + (Math.random() - 0.5) * drift * 5;
      let goalX = lerp(noiseX, p.tx, coherence);
      let goalY = lerp(noiseY, p.ty, coherence);

      // pointer repulsion — feels alive
      const dx = p.x - pointer.x, dy = p.y - pointer.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < 13000) { const f = (13000 - d2) / 13000; goalX += dx * f * 0.22; goalY += dy * f * 0.22; }

      p.vx = lerp(p.vx, (goalX - p.x) * 0.14, 0.45);
      p.vy = lerp(p.vy, (goalY - p.y) * 0.14, 0.45);
      p.x += p.vx; p.y += p.vy;

      const a = 0.22 + coherence * 0.78;
      const c = coherence > 0.55 ? ACCENT : ACCENT2;
      const cc = [
        Math.round(lerp(c[0], 243, coherence * 0.55)),
        Math.round(lerp(c[1], 241, coherence * 0.55)),
        Math.round(lerp(c[2], 234, coherence * 0.55)),
      ];
      ctx.fillStyle = `rgba(${cc[0]},${cc[1]},${cc[2]},${a})`;
      // crisper, fuller dots when resolved; pixel-snapped to avoid sub-pixel fuzz
      const r = p.s * (0.95 + coherence * 0.85);
      ctx.fillRect(Math.round(p.x), Math.round(p.y), r, r);
    }
    raf = requestAnimationFrame(step);
  }

  let onHint = null;
  function setPhase(next, now) { phase = next; phaseStart = now; if (onHint) onHint(next); }

  let raf;
  function start() {
    if (reduced) { drawStatic(); return; }
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(step);
  }
  function drawStatic() {
    coherence = 1;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(243,241,234,0.92)';
    for (const t of targets) ctx.fillRect(t.x, t.y, 1.7, 1.7);
  }

  // Only rebuild when WIDTH changes — mobile address-bar show/hide changes
  // height constantly and must NOT trigger a jarring re-seed.
  window.addEventListener('resize', debounce(() => {
    if (Math.abs(canvas.clientWidth - lastW) < 2) { measure(); return; }
    rebuild();
  }, 220));

  window.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    pointer.x = e.clientX - r.left; pointer.y = e.clientY - r.top;
  });
  window.addEventListener('pointerleave', () => { pointer.x = -9999; pointer.y = -9999; });

  // Wait for Fraunces before sampling glyphs, else we trace a fallback font
  // (the cause of the "missing edges / unclear" letters).
  measure();
  const ready = (document.fonts && document.fonts.load)
    ? Promise.all([
        document.fonts.load('700 200px Fraunces'),
        document.fonts.load('600 200px Fraunces'),
      ]).catch(() => {})
    : Promise.resolve();
  ready.then(() => { rebuild(); start(); });

  return { onPhase(fn) { onHint = fn; } };
}

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
