/**
 * Seeded "signal" previews.
 *
 * Every entry gets a stable, abstract trace derived from its slug — texture for
 * the preview pane, not a chart: no axes, no numbers, nothing to misread as data.
 */

/** mulberry32, seeded from a string, so a slug always draws the same figure */
export function rng(seed: string): () => number {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  let a = h >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function cssVar(name: string, fallback = '#35d492'): string {
  if (typeof window === 'undefined') return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

const CATEGORY_HUE: Record<string, string> = {
  paper: '--blue',
  'deep-dive': '--orange',
  idea: '--purple',
  concept: '--aqua',
  tutorial: '--aqua',
};

export function categoryVar(category: string): string {
  return CATEGORY_HUE[category] || '--green';
}

/** Draws the trace. `phase` shifts the waveform so the featured pane can drift. */
export function drawThumb(canvas: HTMLCanvasElement, seed: string, colorVar: string, phase = 0) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;

  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = cssVar('--bg0h', '#080a0d');
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = cssVar('--bg1', '#161a20');
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 16) {
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 16) {
    ctx.beginPath();
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(w, y + 0.5);
    ctx.stroke();
  }

  const color = cssVar(colorVar);
  const rnd = rng(seed);
  const traces = 2 + Math.floor(rnd() * 2);

  for (let t = 0; t < traces; t++) {
    const f1 = 1 + rnd() * 3;
    const f2 = 4 + rnd() * 9;
    const ph = rnd() * 6.283 + phase * (0.4 + t * 0.25);
    const amp = 0.18 + rnd() * 0.22;
    const decay = rnd() < 0.5 ? 0 : rnd() * 1.5;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.globalAlpha = t === 0 ? 0.95 : 0.4;
    ctx.lineWidth = t === 0 ? 1.6 : 1;
    for (let x = 0; x <= w; x += 2) {
      const u = x / w;
      const env = Math.exp(-decay * u);
      const y =
        h / 2 +
        h * amp * env * (Math.sin(u * f1 * 6.283 + ph) * 0.7 + Math.sin(u * f2 * 6.283 + ph * 0.5) * 0.3);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.fillRect(w - 6, h / 2 - 2, 4, 4);
}
