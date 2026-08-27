import { test, expect } from "bun:test";
import { planProblems, simProblems } from "./build-entry";

/** The exact body the pipeline produced for EchoWM and called verified. */
const REAL_STUB =
  "# Executive Summary EchoWM introduces an omnimodal world model that can be entered and " +
  "navigated, generating synchronized 720p video, environmental sound, music, and speech from a " +
  "unified camera‑intent signal. By mapping discrete commands and continuouspos";

const base = {
  slug: "x", title: "X", subtitle: "s", impact: "i", readTime: "5m",
  category: "paper" as const, status: "RESEARCH" as const, tags: [],
  simulationName: "EchoWM", simulationBrief: "b", relatedSlugs: [], body: "",
};

test("catches the stub that passed build and browser checks", () => {
  const found = planProblems({ ...base, body: REAL_STUB }, false);
  expect(found.some((p) => p.includes("words"))).toBe(true);
  expect(found.some((p) => p.includes("'##' headings"))).toBe(true);
  expect(found.some((p) => p.includes("mermaid"))).toBe(true);
  expect(found.some((p) => p.includes("heading runs into body text"))).toBe(true);
  expect(found.some((p) => p.includes("ends mid-sentence"))).toBe(true);
});

test("flags a truncated reply even when the text looks complete", () => {
  const good = "## A\n\ntext. " + "word ".repeat(700) + "end.\n\n```mermaid\ngraph TD\nA-->B\n```\n\n```python\nx=1\n```\n\n## B\n\n## C\n\n## D\n\nDone.";
  expect(planProblems({ ...base, body: good }, true))
    .toEqual(["the reply hit the token cap, so the entry is cut off mid-generation"]);
});

test("passes a well-formed entry", () => {
  const good =
    "# Executive Summary\n\n" + "word ".repeat(650) + "end.\n\n" +
    "## Why it matters\n\ntext.\n\n## The mechanism\n\n```mermaid\ngraph TD\nA-->B\n```\n\n" +
    "## Cost\n\n```python\nx = 1\n```\n\n## Open questions\n\nMore text here.";
  expect(planProblems({ ...base, body: good }, false)).toEqual([]);
});

test("catches the broken mulberry32 the model invented", () => {
  const bad = `function mulberry32(seed) {
  let t = seed >>> 0;
  return function() { t = Math.imul(t *= 0x9e3779b9, 0xffffffff); return t; };
}
export default function S() { return null; }`;
  expect(simProblems(bad).some((p) => p.includes("not mulberry32"))).toBe(true);
  expect(simProblems(bad).some((p) => p.includes("lines"))).toBe(true);
});

test("accepts a real seeded simulation", () => {
  const ok = "const x=1;\n".repeat(200) +
    "function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6d2b79f5)>>>0;return s/2**32;};}\n" +
    "export default function S(){return null;}";
  expect(simProblems(ok)).toEqual([]);
});

import { unsupportedNumbers } from "./build-entry";

const ABSTRACT =
  "We present EchoWM, an omnimodal world model for enterable generative media that responds to " +
  "continuous navigation while jointly generating 720p video, environmental sound, music and speech. " +
  "Discrete commands and continuous poses are mapped to a shared metric-scale relative 6-DoF trajectory.";

test("accepts numbers that appear in the abstract", () => {
  const body = "It generates 720p video and maps commands to a 6-DoF trajectory.";
  expect(unsupportedNumbers(body, ABSTRACT)).toEqual([]);
});

test("flags an invented benchmark figure", () => {
  const body = "EchoWM reaches 81.7 on WBench and improves quality by 23%.";
  const found = unsupportedNumbers(body, ABSTRACT);
  expect(found.some((n) => n.includes("81.7"))).toBe(true);
  expect(found.some((n) => n.includes("23"))).toBe(true);
});

test("ignores numbers inside code and math", () => {
  const body = "Text.\n\n```python\nlr = 0.0003\nsteps = 50000\n```\n\nAnd $\\alpha = 0.15$ inline.";
  expect(unsupportedNumbers(body, ABSTRACT)).toEqual([]);
});

test("ignores years", () => {
  expect(unsupportedNumbers("Published in 2026, following 2024 work.", ABSTRACT)).toEqual([]);
});

test("catches a canvas colour set to an unresolved CSS variable", () => {
  const src = "const x=1;\n".repeat(200) +
    "const c = el.getContext('2d');\nc.strokeStyle = 'var(--purple)';\n" +
    "const dpr = window.devicePixelRatio; const w = el.clientWidth;\n" +
    "export default function S(){return null;}";
  expect(simProblems(src).some((p) => p.includes("cannot resolve CSS variables"))).toBe(true);
});

test("catches a canvas that ignores DPR and container width", () => {
  const src = "const x=1;\n".repeat(200) +
    "const c = el.getContext('2d');\nc.strokeStyle = resolved;\n" +
    "export default function S(){return null;}";
  const found = simProblems(src);
  expect(found.some((p) => p.includes("devicePixelRatio"))).toBe(true);
  expect(found.some((p) => p.includes("sized from its container"))).toBe(true);
});

test("accepts a correctly written canvas simulation", () => {
  const src = "const x=1;\n".repeat(200) +
    "function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6d2b79f5)>>>0;return s/2**32;};}\n" +
    "const c = el.getContext('2d');\nconst dpr = window.devicePixelRatio;\n" +
    "const w = wrap.clientWidth; new ResizeObserver(m).observe(wrap);\n" +
    "const css = getComputedStyle(document.documentElement);\n" +
    "const resolve = (t) => css.getPropertyValue(t.slice(6, -1));\n" +
    "c.strokeStyle = resolve('var(--purple)');\n" +
    "export default function S(){return null;}";
  expect(simProblems(src)).toEqual([]);
});
