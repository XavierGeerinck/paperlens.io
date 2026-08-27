import { test, expect } from "bun:test";
import { extractJson } from "./llm";

test("plain json", () => {
  expect(extractJson('{"a":1}')).toEqual({ a: 1 });
});

test("fenced json", () => {
  expect(extractJson('```json\n{"a":[1,2]}\n```')).toEqual({ a: [1, 2] });
});

test("json wrapped in prose", () => {
  expect(extractJson('Sure!\n{"a":1}\nHope that helps')).toEqual({ a: 1 });
});

test("truncated mid-object recovers the complete elements", () => {
  const truncated = '{"shortlist":[{"id":"2608.1","total":16},{"id":"2608.2","total":15},{"id":"2608.3","tot';
  const out = extractJson(truncated) as any;
  expect(out.shortlist).toHaveLength(2);
  expect(out.shortlist[1].id).toBe("2608.2");
});

test("truncated mid-string is rejected rather than guessed at", () => {
  expect(() => extractJson('{"a":[{"b":"unterminated')).toThrow();
});

test("truncated with nested arrays", () => {
  const t = '{"pick":{"id":"x"},"shortlist":[{"id":"a","tags":["p","q"]},{"id":"b","tags":["r"]},{"id":"c"';
  const out = extractJson(t) as any;
  expect(out.shortlist).toHaveLength(2);
  expect(out.shortlist[0].tags).toEqual(["p", "q"]);
  expect(out.pick.id).toBe("x");
});
