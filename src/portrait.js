// Portrait with a "diffusion" reveal: the photo resolves from a coarse, noisy
// field into a crisp image — coarse→fine, like reverse diffusion sampling.
// Desktop only (the element is hidden on small screens via CSS).

export function initPortrait(figure, { src, focusX = 0.27, focusY = 0.64, zoom = 2.0 } = {}) {
  const canvas = figure.querySelector('.portrait__canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let W = 0, H = 0, dpr = 1, img = null, played = false;

  // reusable noise tile
  const noise = document.createElement('canvas');
  noise.width = noise.height = 160;
  const nctx = noise.getContext('2d');
  function reseedNoise() {
    const im = nctx.createImageData(160, 160);
    for (let i = 0; i < im.data.length; i += 4) {
      const v = Math.random() * 255;
      im.data[i] = im.data[i + 1] = im.data[i + 2] = v; im.data[i + 3] = 255;
    }
    nctx.putImageData(im, 0, 0);
  }

  function size() {
    const r = canvas.getBoundingClientRect();
    if (!r.width) return false;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.round(r.width); H = Math.round(r.height);
    canvas.width = W * dpr; canvas.height = H * dpr;
    return true;
  }

  // cover-fit source rect, biased to the focus point
  function coverRect() {
    const iw = img.naturalWidth, ih = img.naturalHeight;
    const boxAR = W / H, imgAR = iw / ih;
    let sw, sh;
    if (imgAR > boxAR) { sh = ih / zoom; sw = sh * boxAR; }
    else { sw = iw / zoom; sh = sw / boxAR; }
    const sx = (iw - sw) * focusX;
    const sy = (ih - sh) * focusY;
    return { sx, sy, sw, sh };
  }

  // one frame at progress p in [0,1]
  function frame(p) {
    const tctx = ctx;
    tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    tctx.clearRect(0, 0, W, H);
    const { sx, sy, sw, sh } = coverRect();

    // coarse -> fine: low-res buffer whose resolution grows with p
    const minBlocks = 7;
    const lr = Math.max(minBlocks, Math.round(minBlocks + Math.pow(p, 1.7) * (W - minBlocks)));
    const tw = lr, th = Math.max(minBlocks, Math.round(lr * H / W));
    tmp.width = tw; tmp.height = th;
    tmpCtx.imageSmoothingEnabled = true;
    tmpCtx.drawImage(img, sx, sy, sw, sh, 0, 0, tw, th);

    tctx.save();
    tctx.globalAlpha = Math.min(1, p * 3);                 // fade in early
    tctx.filter = `blur(${(1 - p) * 7}px) saturate(${0.7 + p * 0.3})`;
    tctx.imageSmoothingEnabled = p > 0.5;                  // blocky early, smooth late
    tctx.drawImage(tmp, 0, 0, tw, th, 0, 0, W, H);
    tctx.restore();

    // fading noise veil
    const na = (1 - p) * 0.5;
    if (na > 0.01) {
      reseedNoise();
      tctx.save();
      tctx.globalAlpha = na; tctx.globalCompositeOperation = 'overlay';
      tctx.drawImage(noise, 0, 0, W, H);
      tctx.restore();
    }

    // thin scan of accent light sweeping down as it resolves
    if (p < 1) {
      const y = p * H;
      const g = tctx.createLinearGradient(0, y - 40, 0, y + 8);
      g.addColorStop(0, 'rgba(139,124,246,0)');
      g.addColorStop(1, 'rgba(139,124,246,0.35)');
      tctx.fillStyle = g; tctx.fillRect(0, y - 40, W, 48);
    }
  }

  const tmp = document.createElement('canvas');
  const tmpCtx = tmp.getContext('2d');

  const ease = (t) => 1 - Math.pow(1 - t, 3);

  function play() {
    if (played || !img || !size()) return;
    played = true;
    if (reduced) { frame(1); return; }
    const dur = 1900; const t0 = performance.now();
    const tick = (now) => {
      const p = ease(Math.min(1, (now - t0) / dur));
      frame(p);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // keep it sharp on resize after it has played
  let ro = new ResizeObserver(() => { if (played && img && size()) frame(1); });
  ro.observe(canvas);

  img = new Image();
  img.onload = () => {
    if (size()) frame(0);
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { play(); io.disconnect(); } });
    }, { threshold: 0.25 });
    io.observe(figure);
    // Safety: never leave the portrait stuck on the noise frame.
    setTimeout(() => play(), 4000);
  };
  img.onerror = () => { figure.style.display = 'none'; };
  img.src = src;
}
