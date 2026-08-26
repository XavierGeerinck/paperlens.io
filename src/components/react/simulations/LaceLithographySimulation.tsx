// Lace Lithography — EUV vs He*: side-by-side chip cross-section
// Key visual: explicit diffraction CONE below each mask opening for EUV
// vs a tight COLUMN for He* — the difference is obvious before any simulation runs.

import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, FlaskConical, Cpu } from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

// ─── Canvas ───────────────────────────────────────────────────────────────────
const W = 700;
const H = 340;
const HALF = W / 2; // 350 — left = EUV, right = He*
const NC = 35; // dose-columns per half
const MX = 8; // margin inside each half
const CW = (HALF - MX * 2) / NC; // ≈ 9.5 px / cell

// ─── Layer Y Positions ────────────────────────────────────────────────────────
const SRC_Y = 10;
const MASK_Y = 52;
const MASK_H = 14;
const MASK_BOT = MASK_Y + MASK_H; // 66

// Diffraction-visualization zone — the key educational layer
const DIFF_Y = MASK_BOT; // 66
const DIFF_H = 36; // 36 px tall "what happens after the mask"
const DIFF_BOT = DIFF_Y + DIFF_H; // 102

const RES_Y = DIFF_BOT; // 102
const RES_H = 44;
const RES_BOT = RES_Y + RES_H; // 146

const OX_Y = RES_BOT + 3; // 149
const OX_H = 20;
const OX_BOT = OX_Y + OX_H; // 169

const SUB_Y = OX_BOT; // 169
const SUB_H = 34;
const SUB_BOT = SUB_Y + SUB_H; // 203

// Dose profile chart
const CHART_Y = SUB_BOT + 12; // 215
const CHART_H = 62;
const CHART_BOT = CHART_Y + CHART_H; // 277

// EUV diffraction spread: how many px the beam fans out per px of depth
// 38 px deep zone, spread = 1.3 × depth → each side fans ≈ 47 px at RES level
const EUV_SPREAD_PX = DIFF_H * 1.4; // ≈ 50 px per side at DIFF_BOT
const HE_SPREAD_PX = 1.5; // He* barely diverges

// ─── Dose Physics ─────────────────────────────────────────────────────────────
const SIGMA_EUV = 7.5; // columns — blurry dose hump
const SIGMA_HE = 1.0; // columns — tight dose spike
const DOSE_THRESH = 0.55;

// ─── Mask: 3 gate openings, 11-col pitch (tight enough to stress EUV) ─────────
const GAPS: Array<[number, number]> = [
  [3, 3],   // cols 3–5
  [14, 3],  // cols 14–16
  [25, 3],  // cols 25–27
];

const inGap = (c: number) => GAPS.some(([s, w]) => c >= s && c < s + w);
const lX = (c: number) => MX + c * CW;              // left (EUV) col → x
const rX = (c: number) => HALF + MX + c * CW;       // right (He*) col → x
const xToC = (x: number, side: "L" | "R") =>
  Math.max(0, Math.min(NC - 1, Math.floor((x - (side === "L" ? MX : HALF + MX)) / CW)));

// ─── Types ────────────────────────────────────────────────────────────────────
type Phase = "expose" | "develop" | "etch" | "done";

interface Particle {
  id: number;
  x: number; y: number;
  vx: number; vy: number;
  alpha: number;
  passed: boolean;
  side: "L" | "R";
}

interface SimState extends Record<string, unknown> {
  tick: number;
  particles: Particle[];
  dose_euv: number[];
  dose_he: number[];
  devProg_euv: number[];
  devProg_he: number[];
  etchProg_euv: number[];
  etchProg_he: number[];
}

const mkState = (): SimState => ({
  tick: 0, particles: [],
  dose_euv: new Array(NC).fill(0), dose_he: new Array(NC).fill(0),
  devProg_euv: new Array(NC).fill(0), devProg_he: new Array(NC).fill(0),
  etchProg_euv: new Array(NC).fill(0), etchProg_he: new Array(NC).fill(0),
});

let _pid = 0;

// ─── Component ────────────────────────────────────────────────────────────────
const LaceLithographySimulation: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef<Phase>("expose");
  const [phase, setPhaseState] = useState<Phase>("expose");

  const setPhase = useCallback((p: Phase) => {
    phaseRef.current = p;
    setPhaseState(p);
  }, []);

  const { isRunning, state, start, stop, reset: simReset } = useSimulation<SimState>({
    initialState: mkState(),
    tickRate: 35,
    onTick: (prev, tick) => {
      const ph = phaseRef.current;

      // ── EXPOSE ──────────────────────────────────────────────────────────────
      if (ph === "expose") {
        const parts: Particle[] = [];

        for (const p of prev.particles) {
          const np: Particle = { ...p, x: p.x + p.vx, y: p.y + p.vy, alpha: p.alpha - 0.018 };
          if (!np.passed && np.y >= MASK_Y && np.y < MASK_BOT + 4) {
            const c = xToC(np.x, np.side);
            if (!inGap(c)) {
              np.alpha = 0;
            } else {
              np.passed = true;
              // EUV photons diffract strongly at aperture edges; He* atoms do not
              if (np.side === "L") np.vx += (Math.random() - 0.5) * 8.0;
            }
          }
          if (np.alpha > 0 && np.y < H + 6) parts.push(np);
        }

        // Spawn one EUV + one He* per pair of ticks
        if (tick % 2 === 0) {
          const g = GAPS[Math.floor(Math.random() * GAPS.length)];
          const cx = g[0] + Math.random() * g[1];
          for (const side of ["L", "R"] as const) {
            parts.push({
              id: _pid++,
              x: (side === "L" ? lX(cx) : rX(cx)),
              y: SRC_Y + Math.random() * 10,
              vx: 0, vy: 3.4 + Math.random() * 0.8,
              alpha: 0.75 + Math.random() * 0.25,
              passed: false, side,
            });
          }
        }

        // Accumulate dose with Gaussian lateral spread per column
        const dose_euv = prev.dose_euv.slice();
        const dose_he  = prev.dose_he.slice();
        for (const p of parts) {
          if (!p.passed || p.y <= RES_Y || p.y >= RES_BOT + 4) continue;
          const pc = xToC(p.x, p.side);
          const sigma = p.side === "L" ? SIGMA_EUV : SIGMA_HE;
          const target = p.side === "L" ? dose_euv : dose_he;
          for (let c = 0; c < NC; c++) {
            const d = c - pc;
            target[c] = Math.min(1, target[c] + 0.011 * Math.exp(-d * d / (2 * sigma * sigma)));
          }
        }
        return { ...prev, tick, particles: parts, dose_euv, dose_he };
      }

      // ── DEVELOP ─────────────────────────────────────────────────────────────
      if (ph === "develop") {
        const devProg_euv = prev.devProg_euv.slice();
        const devProg_he  = prev.devProg_he.slice();
        for (let c = 0; c < NC; c++) {
          if (prev.dose_euv[c] >= DOSE_THRESH) devProg_euv[c] = Math.min(1, devProg_euv[c] + 0.055);
          if (prev.dose_he[c]  >= DOSE_THRESH) devProg_he[c]  = Math.min(1, devProg_he[c]  + 0.055);
        }
        return { ...prev, tick, particles: [], devProg_euv, devProg_he };
      }

      // ── ETCH ─────────────────────────────────────────────────────────────────
      if (ph === "etch") {
        const etchProg_euv = prev.etchProg_euv.slice();
        const etchProg_he  = prev.etchProg_he.slice();
        for (let c = 0; c < NC; c++) {
          if (prev.devProg_euv[c] >= 0.98) etchProg_euv[c] = Math.min(1, etchProg_euv[c] + 0.036);
          if (prev.devProg_he[c]  >= 0.98) etchProg_he[c]  = Math.min(1, etchProg_he[c]  + 0.036);
        }
        return { ...prev, tick, etchProg_euv, etchProg_he };
      }

      return { ...prev, tick };
    },
  });

  const dose_euv    = state.dose_euv    as number[];
  const dose_he     = state.dose_he     as number[];
  const devProg_euv = state.devProg_euv as number[];
  const devProg_he  = state.devProg_he  as number[];
  const etchProg_euv = state.etchProg_euv as number[];
  const etchProg_he  = state.etchProg_he  as number[];
  const particles   = state.particles   as Particle[];

  const devComplete = useMemo(() =>
    phase === "develop" &&
    (dose_euv.some(d => d >= DOSE_THRESH) || dose_he.some(d => d >= DOSE_THRESH)) &&
    devProg_euv.every((v, i) => dose_euv[i] < DOSE_THRESH || v >= 0.98) &&
    devProg_he.every( (v, i) => dose_he[i]  < DOSE_THRESH || v >= 0.98),
    [phase, dose_euv, dose_he, devProg_euv, devProg_he]);

  const etchComplete = useMemo(() =>
    phase === "etch" &&
    (etchProg_euv.some(v => v >= 0.98) || etchProg_he.some(v => v >= 0.98)) &&
    etchProg_euv.every((v, i) => devProg_euv[i] < 0.98 || v >= 0.98) &&
    etchProg_he.every( (v, i) => devProg_he[i]  < 0.98 || v >= 0.98),
    [phase, etchProg_euv, etchProg_he, devProg_euv, devProg_he]);

  useEffect(() => { if (devComplete) stop(); }, [devComplete, stop]);
  useEffect(() => {
    if (etchComplete) { setPhase("done"); stop(); }
  }, [etchComplete, setPhase, stop]);

  const handleDevelop = useCallback(() => { setPhase("develop"); start(); }, [setPhase, start]);
  const handleEtch    = useCallback(() => { setPhase("etch");    start(); }, [setPhase, start]);
  const handleReset   = useCallback(() => { simReset(); setPhase("expose"); _pid = 0; }, [simReset, setPhase]);

  // ─── Canvas Rendering ──────────────────────────────────────────────────────
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "rgb(2,6,23)";
    ctx.fillRect(0, 0, W, H);

    // ── render one side's chip cross-section ─────────────────────────────────
    const drawSide = (
      side: "L" | "R",
      dose: number[], devProg: number[], etchProg: number[],
      ar: number, ag: number, ab: number,
    ) => {
      const xOrig = side === "L" ? 0 : HALF;
      const cx    = side === "L" ? lX : rX;
      const sW    = HALF;
      const acc   = (a: number) => `rgba(${ar},${ag},${ab},${a})`;
      const isEUV = side === "L";
      const spread = isEUV ? EUV_SPREAD_PX : HE_SPREAD_PX;

      // ── Si substrate ─────────────────────────────────────────────────────
      const sg = ctx.createLinearGradient(0, SUB_Y, 0, SUB_BOT);
      sg.addColorStop(0, "#1a2a4a"); sg.addColorStop(1, "#0f1729");
      ctx.fillStyle = sg;
      ctx.fillRect(xOrig + MX, SUB_Y, sW - MX * 2, SUB_H);
      ctx.strokeStyle = "rgba(99,102,241,0.06)";
      ctx.lineWidth = 0.5;
      for (let x = xOrig + MX; x < xOrig + sW - MX; x += 10) {
        ctx.beginPath(); ctx.moveTo(x, SUB_Y); ctx.lineTo(x, SUB_BOT); ctx.stroke();
      }

      // ── SiO₂ oxide layer ─────────────────────────────────────────────────
      for (let c = 0; c < NC; c++) {
        const ep = etchProg[c];
        const x  = cx(c);
        if (ep >= 1) {
          ctx.fillStyle = "rgba(26,42,74,0.45)";
          ctx.fillRect(x, OX_Y, CW - 0.5, OX_H);
        } else if (ep > 0) {
          const etcH    = OX_H * (1 - ep);
          const etcTopY = OX_BOT - etcH;
          ctx.fillStyle = "rgba(100,116,139,0.82)";
          ctx.fillRect(x, OX_Y, CW - 0.5, OX_H - etcH);
          const gl = ctx.createLinearGradient(0, etcTopY - 4, 0, etcTopY + 4);
          gl.addColorStop(0, "rgba(56,189,248,0)");
          gl.addColorStop(0.5, "rgba(56,189,248,0.75)");
          gl.addColorStop(1, "rgba(56,189,248,0)");
          ctx.fillStyle = gl;
          ctx.fillRect(x - 1, etcTopY - 4, CW + 1, 8);
        } else {
          ctx.fillStyle = devProg[c] >= 0.98 ? "rgba(148,163,184,0.5)" : "rgba(100,116,139,0.82)";
          ctx.fillRect(x, OX_Y, CW - 0.5, OX_H);
        }
      }
      ctx.strokeStyle = "rgba(148,163,184,0.12)"; ctx.lineWidth = 0.5;
      ctx.strokeRect(xOrig + MX, OX_Y, sW - MX * 2, OX_H);

      // ── Photoresist ──────────────────────────────────────────────────────
      for (let c = 0; c < NC; c++) {
        const d  = dose[c];
        const dp = devProg[c];
        if (dp >= 1) continue;
        const x       = cx(c);
        const remainH = RES_H * (1 - dp);
        const resTop  = RES_BOT - remainH;
        let rr = 38, gg = 82, bb = 20;
        if (d >= DOSE_THRESH) {
          rr = 245; gg = 55; bb = 10;
        } else if (d > DOSE_THRESH * 0.4) {
          const t = (d - DOSE_THRESH * 0.4) / (DOSE_THRESH * 0.6);
          rr = 38 + Math.round(t * 207);
          gg = 82 - Math.round(t * 27);
          bb = 20 - Math.round(t * 10);
        }
        ctx.fillStyle = `rgba(${rr},${gg},${bb},${(1 - dp) * 0.9})`;
        ctx.fillRect(x, resTop, CW - 0.5, remainH);
      }
      ctx.strokeStyle = "rgba(74,222,128,0.08)"; ctx.lineWidth = 0.5;
      ctx.strokeRect(xOrig + MX, RES_Y, sW - MX * 2, RES_H);

      // ── h-BN Mask ────────────────────────────────────────────────────────
      ctx.fillStyle = "#242e3d";
      ctx.fillRect(xOrig + MX, MASK_Y, sW - MX * 2, MASK_H);
      for (const [s, w] of GAPS) {
        ctx.fillStyle = "rgba(2,6,23,0.92)";
        ctx.fillRect(cx(s), MASK_Y, w * CW, MASK_H);
        ctx.strokeStyle = "rgba(148,163,184,0.3)"; ctx.lineWidth = 0.5;
        ctx.strokeRect(cx(s), MASK_Y, w * CW, MASK_H);
      }
      ctx.strokeStyle = "rgba(148,163,184,0.18)"; ctx.lineWidth = 0.5;
      ctx.strokeRect(xOrig + MX, MASK_Y, sW - MX * 2, MASK_H);

      // ── DIFFRACTION VISUALIZATION ZONE ───────────────────────────────────
      // This is the key educational element:
      // EUV: beam fans OUT as a wide cone (overlapping cones → dose bleeds between features)
      // He*: beam travels STRAIGHT DOWN (tight column, no overlap)
      ctx.fillStyle = "rgba(2,6,23,0.5)";
      ctx.fillRect(xOrig + MX, DIFF_Y, sW - MX * 2, DIFF_H);

      for (const [s, w] of GAPS) {
        const x1 = cx(s);
        const x2 = cx(s + w);

        if (isEUV) {
          // Filled trapezoid = diffraction fan
          ctx.beginPath();
          ctx.moveTo(x1, DIFF_Y);            // mask opening left
          ctx.lineTo(x2, DIFF_Y);            // mask opening right
          ctx.lineTo(x2 + spread, DIFF_BOT); // fan spread right at resist top
          ctx.lineTo(x1 - spread, DIFF_BOT); // fan spread left at resist top
          ctx.closePath();
          const grad = ctx.createLinearGradient(0, DIFF_Y, 0, DIFF_BOT);
          grad.addColorStop(0, acc(0.55));
          grad.addColorStop(1, acc(0.15));
          ctx.fillStyle = grad;
          ctx.fill();
          // Diverging edge lines
          ctx.strokeStyle = acc(0.65);
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(x1, DIFF_Y); ctx.lineTo(x1 - spread, DIFF_BOT);
          ctx.moveTo(x2, DIFF_Y); ctx.lineTo(x2 + spread, DIFF_BOT);
          ctx.stroke();
          ctx.setLineDash([]);
        } else {
          // Tight collimated column
          const colGrad = ctx.createLinearGradient(0, DIFF_Y, 0, DIFF_BOT);
          colGrad.addColorStop(0, acc(0.55));
          colGrad.addColorStop(1, acc(0.35));
          ctx.fillStyle = colGrad;
          ctx.fillRect(x1 - spread, DIFF_Y, (x2 - x1) + spread * 2, DIFF_H);
          ctx.strokeStyle = acc(0.65);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x1, DIFF_Y); ctx.lineTo(x1, DIFF_BOT);
          ctx.moveTo(x2, DIFF_Y); ctx.lineTo(x2, DIFF_BOT);
          ctx.stroke();
        }
      }

      // Zone label
      ctx.fillStyle = isEUV ? "rgba(249,115,22,0.5)" : "rgba(6,182,212,0.5)";
      ctx.font = "bold 7px 'Courier New',monospace";
      ctx.textAlign = isEUV ? "right" : "left";
      ctx.fillText(
        isEUV ? "↔ diffracts out wide" : "↕ stays collimated",
        isEUV ? xOrig + sW - MX - 2 : xOrig + MX + 2,
        DIFF_Y + DIFF_H / 2 + 3,
      );

      // ── Beam source glow ─────────────────────────────────────────────────
      const bg = ctx.createLinearGradient(0, SRC_Y, 0, MASK_Y);
      bg.addColorStop(0, acc(0.09)); bg.addColorStop(1, acc(0));
      ctx.fillStyle = bg;
      ctx.fillRect(xOrig + MX, SRC_Y, sW - MX * 2, MASK_Y - SRC_Y);
      // Incoming parallel-ray hint
      ctx.strokeStyle = acc(0.2); ctx.lineWidth = 0.5;
      for (const [s, w] of GAPS) {
        for (let i = 0; i <= w; i++) {
          const rx = cx(s) + i * CW;
          ctx.beginPath(); ctx.moveTo(rx, SRC_Y + 4); ctx.lineTo(rx, MASK_Y - 2); ctx.stroke();
        }
      }

      // ── Feature-width brackets (done phase) ──────────────────────────────
      if (phase === "done") {
        const BY = OX_BOT + 4;
        ctx.strokeStyle = acc(0.9); ctx.lineWidth = 1;
        ctx.fillStyle   = acc(0.9);
        ctx.font = "8px 'Courier New',monospace";
        for (const [s, w] of GAPS) {
          let lo = s, hi = s + w - 1;
          while (lo > 0 && etchProg[lo - 1] >= 0.9) lo--;
          while (hi < NC - 1 && etchProg[hi + 1] >= 0.9) hi++;
          const bx1 = cx(lo), bx2 = cx(hi + 1);
          ctx.beginPath();
          ctx.moveTo(bx1, BY); ctx.lineTo(bx1, BY + 5);
          ctx.moveTo(bx1, BY + 2.5); ctx.lineTo(bx2, BY + 2.5);
          ctx.moveTo(bx2, BY); ctx.lineTo(bx2, BY + 5);
          ctx.stroke();
          ctx.textAlign = "center";
          ctx.fillText(`${hi - lo + 1} cols`, (bx1 + bx2) / 2, BY + 14);
        }
      }
    };

    // Draw both halves
    drawSide("L", dose_euv, devProg_euv, etchProg_euv, 249, 115, 22);
    drawSide("R", dose_he,  devProg_he,  etchProg_he,  6, 182, 212);

    // ── Particles ──────────────────────────────────────────────────────────
    for (const p of particles) {
      if (p.alpha <= 0) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.passed ? 1.5 : 1.9, 0, Math.PI * 2);
      ctx.fillStyle = p.side === "L"
        ? `rgba(251,191,36,${p.alpha})`
        : `rgba(103,232,249,${p.alpha})`;
      ctx.fill();
    }

    // ── Centre divider ─────────────────────────────────────────────────────
    ctx.strokeStyle = "rgba(71,85,105,0.4)";
    ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(HALF, SRC_Y); ctx.lineTo(HALF, CHART_BOT + 6); ctx.stroke();
    ctx.setLineDash([]);

    // ── Side header labels ─────────────────────────────────────────────────
    ctx.font = "bold 8px 'Courier New',monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(249,115,22,0.9)";
    ctx.fillText("► EUV PHOTON  —  λ = 13.5 nm  (ASML)", HALF / 2, SRC_Y + 9);
    ctx.fillStyle = "rgba(6,182,212,0.9)";
    ctx.fillText("► He* ATOM  —  λ = 0.056 nm  (LACE)", HALF + HALF / 2, SRC_Y + 9);

    // ── Right-edge layer labels ────────────────────────────────────────────
    ctx.textAlign = "right";
    const lbl = (t: string, y: number, clr: string, fnt?: string) => {
      ctx.fillStyle = clr; ctx.font = fnt ?? "bold 7px 'Courier New',monospace";
      ctx.fillText(t, W - 1, y);
    };
    lbl("h-BN mask",    MASK_Y + 9,  "#323944");
    lbl("↕ beam zone",  DIFF_Y + 20, "#6b7583");
    lbl("photoresist",  RES_Y + 11,  "#4ade80");
    lbl("SiO₂ oxide",   OX_Y + 13,   "#98a2ae");
    lbl("Si substrate", SUB_Y + 13,  "#a992f6");
    ctx.textAlign = "left";

    // ── DOSE PROFILE CHART ────────────────────────────────────────────────
    // Shows lateral dose per column — the quantitative proof
    ctx.fillStyle = "rgba(8,12,32,0.7)";
    ctx.fillRect(0, CHART_Y - 16, W, CHART_H + 20);

    ctx.fillStyle = "#22272f"; ctx.font = "bold 7px 'Courier New',monospace";
    ctx.textAlign = "left";
    ctx.fillText("DOSE PROFILE PER COLUMN  ─  red = expose threshold", MX + 2, CHART_Y - 5);

    for (let c = 0; c < NC; c++) {
      const hE = Math.round(dose_euv[c] * CHART_H);
      const hH = Math.round(dose_he[c]  * CHART_H);

      // Shade masked columns subtly
      if (!inGap(c)) {
        ctx.fillStyle = "rgba(100,116,139,0.08)";
        ctx.fillRect(lX(c), CHART_Y, CW - 0.5, CHART_H);
        ctx.fillRect(rX(c), CHART_Y, CW - 0.5, CHART_H);
      }

      if (hE > 0) {
        ctx.fillStyle = dose_euv[c] >= DOSE_THRESH ? "rgba(249,115,22,0.9)" : "rgba(249,115,22,0.38)";
        ctx.fillRect(lX(c), CHART_Y + CHART_H - hE, CW - 0.5, hE);
      }
      if (hH > 0) {
        ctx.fillStyle = dose_he[c] >= DOSE_THRESH ? "rgba(6,182,212,0.9)" : "rgba(6,182,212,0.38)";
        ctx.fillRect(rX(c), CHART_Y + CHART_H - hH, CW - 0.5, hH);
      }
    }

    // Gap zone highlights
    for (const [s, w] of GAPS) {
      ctx.fillStyle = "rgba(255,255,255,0.03)";
      ctx.fillRect(lX(s), CHART_Y, w * CW, CHART_H);
      ctx.fillRect(rX(s), CHART_Y, w * CW, CHART_H);
    }

    // Threshold line
    const thY = CHART_Y + CHART_H - Math.round(DOSE_THRESH * CHART_H);
    ctx.strokeStyle = "rgba(248,113,113,0.7)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(MX, thY); ctx.lineTo(HALF - MX, thY);
    ctx.moveTo(HALF + MX, thY); ctx.lineTo(W - MX, thY);
    ctx.stroke(); ctx.setLineDash([]);

    ctx.fillStyle = "rgba(248,113,113,0.7)"; ctx.font = "7px monospace";
    ctx.textAlign = "right";
    ctx.fillText("expose thr.", HALF - MX - 2, thY - 2);
    ctx.fillText("expose thr.", W - MX - 2, thY - 2);

    // EUV "dose spill" annotation — only visible once dose starts accumulating
    const spillVisible = dose_euv.some((d, i) => !inGap(i) && d > 0.15);
    if (spillVisible) {
      // Find the gap between features 1 and 2 (cols 6-13)
      const midC = Math.round((GAPS[0][0] + GAPS[0][1] + GAPS[1][0]) / 2);
      const midX = lX(midC);
      ctx.fillStyle = "rgba(249,115,22,0.65)";
      ctx.font = "bold 7px monospace"; ctx.textAlign = "center";
      ctx.fillText("spill!", midX, thY - 8);
    }

    // Chart centre divider
    ctx.strokeStyle = "rgba(71,85,105,0.25)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(HALF, CHART_Y - 2); ctx.lineTo(HALF, CHART_BOT + 2); ctx.stroke();
    ctx.setLineDash([]);

    // ── Done: result strip ─────────────────────────────────────────────────
    if (phase === "done") {
      const euvEtched = etchProg_euv.filter(v => v >= 0.9).length;
      const heEtched  = etchProg_he.filter( v => v >= 0.9).length;
      ctx.fillStyle = "rgba(2,6,23,0.9)";
      ctx.fillRect(0, CHART_BOT + 3, W, 22);
      ctx.font = "bold 9px 'Courier New',monospace"; ctx.textAlign = "center";
      ctx.fillStyle = "rgba(249,115,22,0.95)";
      ctx.fillText(`EUV result: ${euvEtched} cols etched — gaps merged, features unresolved`, HALF / 2, CHART_BOT + 15);
      ctx.fillStyle = "rgba(6,182,212,0.95)";
      ctx.fillText(`He* result: ${heEtched} cols etched — 3 isolated trenches, perfect pitch`, HALF + HALF / 2, CHART_BOT + 15);
      ctx.textAlign = "left";
    }

    ctx.textAlign = "left";
  }, [dose_euv, dose_he, devProg_euv, devProg_he, etchProg_euv, etchProg_he, particles, phase]);

  // ─── UI ────────────────────────────────────────────────────────────────────
  const phaseLabels: Record<Phase, string> = {
    expose: "1 · EXPOSE", develop: "2 · DEVELOP", etch: "3 · ETCH", done: "4 · DONE",
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Canvas */}
      <SchematicCard title="EUV vs He* — SAME MASK, RUNNING SIMULTANEOUSLY">
        <div className="flex flex-col items-center gap-3">
          <canvas
            ref={canvasRef} width={W} height={H}
            className="w-full max-w-2xl border border-slate-700/50 rounded-sm"
            style={{ background: "rgb(2,6,23)" }}
          />

          {/* Phase tabs */}
          <div className="flex gap-1.5 text-[9px] font-mono flex-wrap justify-center">
            {(["expose", "develop", "etch", "done"] as Phase[]).map(p => (
              <span key={p} className={`px-2 py-0.5 border transition-colors ${
                phase === p
                  ? "border-cyan-500 text-cyan-300 bg-cyan-950/50"
                  : "border-slate-800 text-slate-600"
              }`}>{phaseLabels[p]}</span>
            ))}
          </div>

          {/* Controls */}
          <div className="flex flex-wrap gap-2 justify-center">
            {phase === "expose" && (
              <>
                <SchematicButton
                  onClick={isRunning ? stop : start}
                  icon={isRunning ? Pause : Play}
                  label={isRunning ? "PAUSE" : "START EXPOSURE"}
                  active={isRunning}
                />
                <SchematicButton onClick={handleDevelop} icon={FlaskConical} label="DEVELOP →" active={false} />
              </>
            )}
            {phase === "develop" && !devComplete && (
              <span className="text-[10px] font-mono text-amber-400 py-1 px-2 border border-amber-800/50">DEVELOPING…</span>
            )}
            {phase === "develop" && devComplete && (
              <SchematicButton onClick={handleEtch} icon={Cpu} label="PLASMA ETCH →" active={false} />
            )}
            {phase === "etch" && !etchComplete && (
              <span className="text-[10px] font-mono text-sky-400 py-1 px-2 border border-sky-800/50">PLASMA ETCHING…</span>
            )}
            {phase === "done" && (
              <span className="text-[10px] font-mono text-emerald-400 py-1 px-2 border border-emerald-700/50">
                ✓ DONE — see trench widths above
              </span>
            )}
            <SchematicButton onClick={handleReset} icon={RotateCcw} label="RESET" />
          </div>
        </div>
      </SchematicCard>

      {/* Physics explanation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SchematicCard title="EUV — ASML STATE OF THE ART">
          <div className="flex flex-col gap-2 text-xs font-mono text-slate-400">
            <div className="flex justify-between">
              <span>Wavelength</span>
              <span className="text-orange-400 font-bold">λ = 13.5 nm</span>
            </div>
            <div className="flex justify-between">
              <span>Particle type</span>
              <span className="text-orange-300">photon (electromagnetic)</span>
            </div>
            <div className="flex justify-between">
              <span>Diffraction after mask</span>
              <span className="text-red-400 font-bold">WIDE  (~50 px fan)</span>
            </div>
            <div className="flex justify-between">
              <span>Dose in gaps between features</span>
              <span className="text-red-400 font-bold">YES — features merge</span>
            </div>
            <div className="border border-orange-900/40 bg-orange-950/20 rounded p-2 text-[10px] text-orange-300/80 mt-1">
              Photons obey wave optics. Any aperture smaller than ~λ causes strong
              diffraction — the beam fans into a wide cone <em>regardless</em> of how
              precise the mask is. At sub-10 nm pitch the cones from adjacent openings
              overlap completely.
            </div>
          </div>
        </SchematicCard>

        <SchematicCard title="He* — LACE NEUTRAL ATOM BEAM">
          <div className="flex flex-col gap-2 text-xs font-mono text-slate-400">
            <div className="flex justify-between">
              <span>de Broglie wavelength</span>
              <span className="text-cyan-400 font-bold">λ ≈ 0.056 nm</span>
            </div>
            <div className="flex justify-between">
              <span>Particle type</span>
              <span className="text-cyan-300">neutral He atom (matter wave)</span>
            </div>
            <div className="flex justify-between">
              <span>Diffraction after mask</span>
              <span className="text-emerald-400 font-bold">NEGLIGIBLE (&lt; 2 px)</span>
            </div>
            <div className="flex justify-between">
              <span>Dose in gaps between features</span>
              <span className="text-emerald-400 font-bold">ZERO — features isolated</span>
            </div>
            <div className="border border-cyan-900/40 bg-cyan-950/20 rounded p-2 text-[10px] text-cyan-300/80 mt-1">
              He* atoms have a de Broglie wavelength 240× shorter than EUV. Diffraction
              is proportional to λ — atoms pass through the mask opening like billiard
              balls, travelling in a tight vertical column all the way to the resist.
            </div>
          </div>
        </SchematicCard>
      </div>

      {/* How to use */}
      <SchematicCard title="WHAT TO WATCH">
        <div className="flex flex-col gap-2 text-xs font-mono text-slate-400">
          <div className="flex gap-2 items-start">
            <span className="text-slate-300 font-bold shrink-0">BEAM ZONE</span>
            <span>
              The orange/cyan shaded zone between the mask and resist shows the beam shape <em>immediately
              after the mask</em>. EUV fans into a wide trapezoid — you can already
              see adjacent cones overlap. He* stays a tight vertical column.
            </span>
          </div>
          <div className="flex gap-2 items-start">
            <span className="text-slate-300 font-bold shrink-0">RESIST</span>
            <span>
              Click <span className="text-slate-200">START EXPOSURE</span>. Watch the resist colour
              change: dark olive = low dose, bright red = exposed (above threshold). EUV turns whole
              rows red; He* only lights up the 3 columns directly below each opening.
            </span>
          </div>
          <div className="flex gap-2 items-start">
            <span className="text-slate-300 font-bold shrink-0">CHART</span>
            <span>
              The dose chart grows in real time. EUV bars exceed the red threshold line
              <em> between</em> the mask openings — the features cannot be resolved. He* bars only
              exceed it <em>at</em> the mask openings. Click DEVELOP → PLASMA ETCH to see the
              final trench widths.
            </span>
          </div>
        </div>
      </SchematicCard>
    </div>
  );
};

export default LaceLithographySimulation;
