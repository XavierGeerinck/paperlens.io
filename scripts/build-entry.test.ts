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
