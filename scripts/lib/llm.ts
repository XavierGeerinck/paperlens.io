/**
 * Model access for the paper pipeline.
 *
 * The endpoint is OpenAI-compatible (OpenRouter), so the official `openai` SDK
 * talks to it directly — only the base URL changes:
 *
 *   OPENAI_API_KEY   secret   the OpenRouter key
 *   OPENAI_API_URL   var      https://openrouter.ai/api/v1
 *   PAPERLENS_MODEL  var      optional model slug override
 *
 * Structured output is requested through `response_format` where the model
 * supports it, and falls back to prompt-enforced JSON where it does not, so
 * changing PAPERLENS_MODEL cannot silently break the pipeline.
 */

import OpenAI from "openai";

export const DEFAULT_MODEL = "~anthropic/claude-opus-latest";

export function model(): string {
	return process.env.PAPERLENS_MODEL?.trim() || DEFAULT_MODEL;
}

export function client(): OpenAI {
	const apiKey = process.env.OPENAI_API_KEY;
	const baseURL = process.env.OPENAI_API_URL?.trim() || "https://openrouter.ai/api/v1";

	if (!apiKey) {
		throw new Error(
			"OPENAI_API_KEY is not set. In CI it comes from repository secrets; " +
				"locally, export it before running this script.",
		);
	}

	return new OpenAI({
		apiKey,
		baseURL,
		// Attribution headers OpenRouter uses for its dashboard; harmless elsewhere.
		defaultHeaders: {
			"HTTP-Referer": "https://paperlens.io",
			"X-Title": "PaperLens paper pipeline",
		},
	});
}

/**
 * Fail early and loudly on a bad model slug rather than after a long harvest.
 * Suggests near matches, since OpenRouter slugs are easy to mistype.
 */
export async function assertModelAvailable(slug = model()): Promise<void> {
	const baseURL = process.env.OPENAI_API_URL?.trim() || "https://openrouter.ai/api/v1";
	let ids: string[];
	try {
		const res = await fetch(`${baseURL}/models`);
		if (!res.ok) return; // not fatal — let the completion call be the real test
		ids = ((await res.json()) as { data: { id: string }[] }).data.map((m) => m.id);
	} catch {
		return;
	}

	if (ids.includes(slug)) return;

	const stem = slug.replace(/^~/, "").split("/")[0];
	const near = ids.filter((id) => id.includes(stem)).slice(0, 12);
	throw new Error(
		`Model "${slug}" is not available at this endpoint.\n` +
			(near.length ? `Close matches:\n  ${near.join("\n  ")}` : `No slugs matched "${stem}".`),
	);
}

interface AskOptions {
	system: string;
	user: string;
	/** JSON Schema; when given, the reply is parsed and returned as an object */
	schema?: { name: string; schema: Record<string, unknown> };
	maxTokens?: number;
	temperature?: number;
	/** retries for transient failures and unparseable JSON */
	retries?: number;
}

/** Pull JSON out of a reply that may be fenced or prefaced with prose. */
export function extractJson(text: string): unknown {
	const trimmed = text.trim();
	const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
	const candidate = fenced ? fenced[1] : trimmed;

	try {
		return JSON.parse(candidate);
	} catch {}

	// Outermost balanced braces, for a reply wrapped in prose.
	const start = candidate.indexOf("{");
	const end = candidate.lastIndexOf("}");
	if (start !== -1 && end > start) {
		try {
			return JSON.parse(candidate.slice(start, end + 1));
		} catch {}
	}

	// Last resort: the reply was cut off mid-structure. Drop the incomplete
	// trailing element and close whatever is still open. Recovering four of five
	// ranked papers beats discarding the whole run.
	if (start !== -1) {
		const repaired = repairTruncatedJson(candidate.slice(start));
		if (repaired) {
			try {
				return JSON.parse(repaired);
			} catch {}
		}
	}

	throw new Error(`Reply was not JSON. Tail of what came back:\n…${text.slice(-500)}`);
}

/** Close the open brackets of a truncated JSON document, discarding a partial tail. */
function repairTruncatedJson(s: string): string | null {
	const stack: string[] = [];
	let inString = false;
	let escaped = false;
	/** index of the last `}`/`]` that closed a complete element near the top level */
	let lastSafe = -1;

	for (let i = 0; i < s.length; i++) {
		const ch = s[i];
		if (escaped) { escaped = false; continue; }
		if (ch === "\\") { escaped = true; continue; }
		if (ch === '"') { inString = !inString; continue; }
		if (inString) continue;

		if (ch === "{" || ch === "[") stack.push(ch);
		else if (ch === "}" || ch === "]") {
			stack.pop();
			// A completed element inside an array is a safe place to cut.
			if (stack.length <= 2) lastSafe = i;
		}
	}

	// A truncation landing inside a string is fine: the incomplete trailing
	// element is discarded at lastSafe anyway. Only the absence of any completed
	// element makes the reply unsalvageable.
	if (lastSafe === -1) return null;

	// Re-walk to the cut point to learn what is still open there.
	const head = s.slice(0, lastSafe + 1);
	const open: string[] = [];
	let str = false, esc = false;
	for (const ch of head) {
		if (esc) { esc = false; continue; }
		if (ch === "\\") { esc = true; continue; }
		if (ch === '"') { str = !str; continue; }
		if (str) continue;
		if (ch === "{" || ch === "[") open.push(ch);
		else if (ch === "}" || ch === "]") open.pop();
	}

	return head + open.reverse().map((c) => (c === "{" ? "}" : "]")).join("");
}

export async function ask(opts: AskOptions): Promise<string> {
	const { text } = await askRaw(opts);
	return text;
}

/** Like `ask`, but tells you whether the reply ran into the token cap. */
export async function askChecked(opts: AskOptions): Promise<{ text: string; truncated: boolean }> {
	return askRaw(opts);
}

export async function askJson<T = unknown>(opts: AskOptions): Promise<T> {
	const { text } = await askRaw(opts);
	return extractJson(text) as T;
}

/** Parsed result plus whether the reply was cut off mid-generation. */
export async function askJsonChecked<T = unknown>(opts: AskOptions): Promise<{ value: T; truncated: boolean }> {
	const { text, truncated } = await askRaw(opts);
	return { value: extractJson(text) as T, truncated };
}

async function askRaw(opts: AskOptions): Promise<{ text: string; truncated: boolean }> {
	const { system, user, schema, maxTokens = 16000, temperature = 0.3, retries = 2 } = opts;
	const api = client();
	const slug = model();

	// Some slugs reject response_format outright; drop it and lean on the prompt.
	let useResponseFormat = Boolean(schema);
	let lastError: unknown;

	for (let attempt = 0; attempt <= retries; attempt++) {
		try {
			const completion = await api.chat.completions.create({
				model: slug,
				temperature,
				max_tokens: maxTokens,
				messages: [
					{ role: "system", content: system },
					{
						role: "user",
						content: schema
							? `${user}\n\nReply with JSON only, matching this schema:\n${JSON.stringify(schema.schema, null, 2)}`
							: user,
					},
				],
				...(useResponseFormat && schema
					? {
							response_format: {
								type: "json_schema" as const,
								json_schema: { name: schema.name, schema: schema.schema, strict: false },
							},
						}
					: {}),
			});

			const choice = completion.choices[0];
			const text = choice?.message?.content ?? "";

			// `slug` may be a router (openrouter/free, openrouter/auto), in which case
			// the model that actually answered is the only thing that explains the
			// output quality. Always say which one it was.
			const served = (completion as { model?: string }).model;
			if (served && served !== slug) {
				console.error(`  · ${slug} routed to ${served}`);
			}
			const usage = (completion as { usage?: { completion_tokens?: number } }).usage;
			if (usage?.completion_tokens) {
				console.error(`  · ${usage.completion_tokens} completion tokens`);
			}

			if (!text.trim()) throw new Error("Model returned an empty reply.");

			if (choice?.finish_reason === "length") {
				console.error(
					`  ! reply hit the ${maxTokens}-token cap and was truncated; attempting to salvage it`,
				);
			}
			if (schema) extractJson(text); // validate parseability before returning
			return { text, truncated: choice?.finish_reason === "length" };
		} catch (err) {
			lastError = err;
			const message = err instanceof Error ? err.message : String(err);

			if (useResponseFormat && /response_format|json_schema|not support/i.test(message)) {
				useResponseFormat = false;
				attempt--; // the retry budget is for real failures, not this downgrade
				continue;
			}

			// Only transient failures are worth repeating. A 402 for credits, a 401
			// for a bad key or a 404 for a bad slug will fail identically every time,
			// and retrying only obscures the real message.
			const status = (err as { status?: number })?.status;
			const transient = status === undefined || status === 408 || status === 429 || status >= 500;
			if (!transient) {
				throw new Error(`Model call failed (${slug}, HTTP ${status}): ${message}`);
			}
			if (attempt < retries) {
				await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
				continue;
			}
		}
	}

	throw new Error(
		`Model call failed after ${retries + 1} attempts (${slug}): ` +
			(lastError instanceof Error ? lastError.message : String(lastError)),
	);
}
