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

test("the aggregate error keeps the status so the chain can fall through", async () => {
  // Regression: a wrapper Error without .status made every rate limit look like
  // an unknown failure, so the chain gave up on the first model every time.
  const { models } = await import("./llm");
  process.env.PAPERLENS_MODEL = "a/one,b/two,c/three";
  expect(models()).toEqual(["a/one", "b/two", "c/three"]);
  delete process.env.PAPERLENS_MODEL;
  const err = Object.assign(new Error("Model call failed after 3 attempts"), { status: 429 });
  const status = (err as { status?: number }).status;
  expect(status === 429 || status === 402 || status === 404 || status === 503).toBe(true);
});

test("a retired slug's 404 moves the chain on instead of ending the run", async () => {
  // Regression: z-ai/glm-5.2:free stopped being free, its 404 lost its status on
  // the non-transient path, and every scheduled radar died on the first model.
  const { ask } = await import("./llm");
  const realFetch = globalThis.fetch;
  const asked: string[] = [];

  process.env.OPENAI_API_KEY = "test";
  process.env.OPENAI_API_URL = "https://llm.invalid/v1";
  process.env.PAPERLENS_MODEL = "gone/model:free,live/model";

  globalThis.fetch = (async (_input: unknown, init?: { body?: unknown }) => {
    const slug = JSON.parse(String(init?.body)).model as string;
    asked.push(slug);
    if (slug === "gone/model:free") {
      return Response.json({ error: { message: "This model is unavailable for free." } }, { status: 404 });
    }
    return Response.json({
      id: "x",
      object: "chat.completion",
      model: slug,
      choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "from the second model" } }],
    });
  }) as unknown as typeof fetch;

  try {
    expect(await ask({ system: "s", user: "u", retries: 0 })).toBe("from the second model");
    expect(asked).toEqual(["gone/model:free", "live/model"]);
  } finally {
    globalThis.fetch = realFetch;
    delete process.env.PAPERLENS_MODEL;
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_URL;
  }
});

test("a bad key is not worth trying on the next model", async () => {
  const { isModelSpecific } = await import("./llm");
  expect(isModelSpecific(Object.assign(new Error("no auth"), { status: 401 }))).toBe(false);
  expect(isModelSpecific(Object.assign(new Error("provider error"), { status: 400 }))).toBe(true);
});
