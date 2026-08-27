import { test, expect } from "bun:test";
import { isoWeek, clampTitle } from "./radar";

test("isoWeek matches the ISO-8601 week the HuggingFace pages use", () => {
  expect(isoWeek(new Date("2026-08-27T00:00:00Z"))).toBe("2026-W35");
  expect(isoWeek(new Date("2026-08-17T00:00:00Z"))).toBe("2026-W34");
  // 1 Jan 2027 is a Friday, so it belongs to the final week of 2026
  expect(isoWeek(new Date("2027-01-01T00:00:00Z"))).toBe("2026-W53");
  // 4 Jan 2027 is the Monday of W01
  expect(isoWeek(new Date("2027-01-04T00:00:00Z"))).toBe("2027-W01");
});

test("clampTitle keeps the suffix when it fits", () => {
  const t = clampTitle("Radar 2026-W35 · EchoWM — enterable omnimodal world models", " · 6 of 204");
  expect(t).toBe("Radar 2026-W35 · EchoWM — enterable omnimodal world models · 6 of 204");
  expect(t.length).toBeLessThanOrEqual(90);
});

test("clampTitle drops the suffix before it truncates the headline", () => {
  const head = "Radar 2026-W35 · " + "x".repeat(70);
  expect(clampTitle(head, " · 6 of 204")).toBe(head);
});

test("clampTitle truncates a headline that cannot fit", () => {
  const t = clampTitle("Radar 2026-W35 · " + "y".repeat(200), " · 6 of 204");
  expect(t.length).toBeLessThanOrEqual(90);
  expect(t.endsWith("…")).toBe(true);
});
