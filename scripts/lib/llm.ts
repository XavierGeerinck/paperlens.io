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

/**
 * Ordered fallback chain.
 *
 * Free models are individually capable but individually rate-limited: a single
 * pinned slug returns 429 whenever that provider is saturated, and the router
 * (openrouter/free) can hand back anything down to a 2.6B model with an 8k
 * output cap — which produces a truncated stub rather than an entry. Trying
 * capable models in order gets both capability and availability.
 *
 * Ordered by output budget and structured-output support, router last.
 */
export const DEFAULT_MODELS = [
	"z-ai/glm-5.2:free",
	"nvidia/nemotron-3-super-120b-a12b:free",
	"dots-studio/dots-3-note-preview:free",
	"minimax/minimax-m3:free",
	"openrouter/free",
];

/** PAPERLENS_MODEL may be a single slug or a comma-separated chain. */
export function models(): string[] {
	const configured = process.env.PAPERLENS_MODEL?.trim();
	if (!configured) return DEFAULT_MODELS;
	return configured.split(",").map((m) => m.trim()).filter(Boolean);
}

export function model(): string {
	return models()[0];
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
export async function assertModelAvailable(): Promise<void> {
	const baseURL = process.env.OPENAI_API_URL?.trim() || "https://openrouter.ai/api/v1";
	let ids: string[];
	try {
		const res = await fetch(`${baseURL}/models`);
		if (!res.ok) return; // not fatal — let the completion call be the real test
		ids = ((await res.json()) as { data: { id: string }[] }).data.map((m) => m.id);
	} catch {
		return;
	}

	const chain = models();
	const missing = chain.filter((slug) => !ids.includes(slug));

	// Only a chain with nothing usable left is fatal; a stale entry just gets skipped.
	if (missing.length === chain.length) {
		const stem = chain[0].replace(/^~/, "").split("/")[0];
		const near = ids.filter((id) => id.includes(stem)).slice(0, 12);
		throw new Error(
			`None of the configured models exist at this endpoint: ${chain.join(", ")}\n` +
				(near.length ? `Close matches:\n  ${near.join("\n  ")}` : `No slugs matched "${stem}".`),
		);
	}
	for (const slug of missing) console.error(`  ! ${slug} is not offered here; it will be skipped`);
	console.error(`  · model chain: ${chain.filter((s) => ids.includes(s)).join(" → ")}`);
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

	// Strip an outer code fence by taking everything between the first newline
	// and the LAST fence. A non-greedy regex stops at the first inner ``` —
	// and the entries we ask for contain mermaid and python fences, so that
	// silently shredded otherwise-valid replies.
	let candidate = trimmed;
	if (trimmed.startsWith("```")) {
		const firstNewline = trimmed.indexOf("\n");
		const lastFence = trimmed.lastIndexOf("```");
		if (firstNewline !== -1 && lastFence > firstNewline) {
			candidate = trimmed.slice(firstNewline + 1, lastFence).trim();
		}
	}

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
	const chain = models();
	let lastFailure: unknown;

	for (let i = 0; i < chain.length; i++) {
		const slug = chain[i];
		try {
			return await askOne(opts, slug);
		} catch (err) {
			lastFailure = err;
			const status = (err as { status?: number })?.status;
			const message = err instanceof Error ? err.message : String(err);
			const exhausted =
				status === 429 ||
				status === 402 ||
				status === 404 ||
				status === 503 ||
				// Repeated unparseable replies mean this model cannot hold the schema.
				/Reply was not JSON/.test(message);
			if (exhausted && i < chain.length - 1) {
				console.error(`  · ${slug} unavailable (HTTP ${status}); falling back to ${chain[i + 1]}`);
				continue;
			}
			throw err;
		}
	}
	throw lastFailure instanceof Error ? lastFailure : new Error(String(lastFailure));
}

async function askOne(opts: AskOptions, slug: string): Promise<{ text: string; truncated: boolean }> {
	const { system, user, schema, maxTokens = 16000, temperature = 0.3, retries = 2 } = opts;
	const api = client();

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
			// Report every attempt as it fails. Keeping only the last one meant a
			// rate limit on retry masked whatever actually went wrong first.
			const status = (err as { status?: number })?.status;
			console.error(`  ! attempt ${attempt + 1}/${retries + 1} failed: ${message.slice(0, 200)}`);

			if (useResponseFormat && /response_format|json_schema|not support/i.test(message)) {
				useResponseFormat = false;
				attempt--; // the retry budget is for real failures, not this downgrade
				continue;
			}

			// Only transient failures are worth repeating. A 402 for credits, a 401
			// for a bad key or a 404 for a bad slug will fail identically every time,
			// and retrying only obscures the real message.
			const transient = status === undefined || status === 408 || status === 429 || status >= 500;
			if (!transient) {
				throw new Error(`Model call failed (${slug}, HTTP ${status}): ${message}`);
			}
			if (attempt < retries) {
				// A rate limit needs a real pause; two seconds just burns an attempt.
				// Honour Retry-After when the provider sends one.
				const retryAfter = Number(
					(err as { headers?: Record<string, string> })?.headers?.["retry-after"] ?? NaN,
				);
				// A 429 means this provider is saturated; the chain moves on rather
				// than waiting minutes here, so keep the in-model pause short.
				const wait = Number.isFinite(retryAfter)
					? Math.min(retryAfter * 1000, 20_000)
					: status === 429
						? 5000 * (attempt + 1)
						: 2000 * (attempt + 1);
				console.error(`    waiting ${Math.round(wait / 1000)}s before retrying`);
				await new Promise((r) => setTimeout(r, wait));
				continue;
			}
		}
	}

	throw new Error(
		`Model call failed after ${retries + 1} attempts (${slug}): ` +
			(lastError instanceof Error ? lastError.message : String(lastError)),
	);
}
