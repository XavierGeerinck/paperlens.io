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

test("fenced json containing inner fences survives (the mermaid case)", () => {
  // Exactly the shape that broke a live build: the reply is fenced, and the
  // body it carries contains its own mermaid and python fences.
  const body = "# Summary\\n\\n```mermaid\\ngraph TD\\nA-->B\\n```\\n\\n```python\\nx = 1\\n```\\n\\nDone.";
  const reply = "```json\n" + JSON.stringify({ slug: "x", body }) + "\n```";
  const out = extractJson(reply) as any;
  expect(out.slug).toBe("x");
  expect(out.body).toContain("mermaid");
  expect(out.body).toContain("python");
});

test("bare json containing fences still parses", () => {
  const reply = JSON.stringify({ body: "```mermaid\\ngraph TD\\n```" });
  expect((extractJson(reply) as any).body).toContain("mermaid");
});
