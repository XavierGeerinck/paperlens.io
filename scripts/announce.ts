#!/usr/bin/env bun
/**
 * Announce a published entry on Threads.
 *
 *   bun scripts/announce.ts <slug>                 # dry run, prints only
 *   bun scripts/announce.ts <slug> --post          # actually publish
 *   bun scripts/announce.ts --delete <media-id>    # remove a published post
 *
 * Publishing is opt-in: without --post this only composes and prints, so the
 * text can be reviewed before anything leaves the machine.
 *
 *   THREADS_ACCESS_TOKEN  secret   long-lived token, threads_basic +
 *                                  threads_content_publish, expires after 60 days
 *   THREADS_USER_ID       var      optional; resolved from /me when absent
 *
 * Threads publishes in two steps: create a media container, then publish it.
 * The 500-character cap counts UTF-8 bytes, not code points, so emoji cost more
 * than they look — the composer measures bytes.
 */

import { catalogue } from "./catalogue";
import { ask } from "./lib/llm";

const GRAPH = "https://graph.threads.net/v1.0";
const SITE = "https://paperlens.io";
const LIMIT = 500;

function bytes(s: string): number {
	return new TextEncoder().encode(s).length;
}

/** Trim to a byte budget without splitting a character or a word. */
function clampBytes(s: string, max: number): string {
	if (bytes(s) <= max) return s;
	let out = s;
	while (bytes(out) > max - 1 && out.length > 0) {
		out = out.slice(0, -1);
	}
	const cut = out.lastIndexOf(" ");
	return (cut > max / 2 ? out.slice(0, cut) : out).trimEnd() + "…";
}

async function threads(path: string, params: Record<string, string>): Promise<any> {
	const res = await fetch(`${GRAPH}${path}`, {
		method: "POST",
		headers: { "Content-Type": "application/x-www-form-urlencoded" },
		body: new URLSearchParams(params),
	});
	const body = await res.json().catch(() => ({}));
	if (!res.ok) {
		const err = (body as any)?.error;
		throw new Error(
			`Threads API ${res.status} on ${path}: ${err?.message ?? JSON.stringify(body)}` +
				(err?.code ? ` (code ${err.code})` : ""),
		);
	}
	return body;
}

async function resolveUserId(token: string): Promise<string> {
	const fromEnv = process.env.THREADS_USER_ID?.trim();
	if (fromEnv) return fromEnv;

	const res = await fetch(`${GRAPH}/me?fields=id,username&access_token=${encodeURIComponent(token)}`);
	const body = (await res.json().catch(() => ({}))) as any;
	if (!res.ok || !body?.id) {
		throw new Error(
			`Could not resolve the Threads user ID: ${body?.error?.message ?? res.status}. ` +
				"Set THREADS_USER_ID explicitly if /me is unavailable.",
		);
	}
	console.error(`  posting as @${body.username ?? body.id}`);
	return body.id as string;
}

/**
 * Signs that the model is thinking out loud rather than writing the post.
 *
 * A reasoning model can spend its whole token budget on preamble; if that text
 * is published it reads as a leaked prompt. This has happened once, so the
 * check is deliberately broad — a false positive costs a fallback template, a
 * false negative costs a public post of the instructions.
 */
const REASONING_SIGNS = [
	/^(we|i|let'?s|okay|ok|first,|the user)\b/i,
	/\b\d{2,4}\s*characters?\b/i,
	/\bannouncement text\b/i,
	/\bnew post:/i,
	/\bmust (open|mention|not)\b/i,
	/\bno (hashtags?|url)\b/i,
	/\b(the )?(prompt|instruction|rule)s?\b/i,
	/\bprovide the\b/i,
];

export function looksLikeReasoning(text: string): string | null {
	for (const re of REASONING_SIGNS) {
		if (re.test(text)) return re.source;
	}
	return null;
}

/** Drop explicit reasoning blocks some models wrap around their answer. */
export function stripReasoning(text: string): string {
	return text
		.replace(/<(think|thinking|reasoning)>[\s\S]*?<\/\1>/gi, "")
		.replace(/^\s*(?:<\/?(?:think|thinking|reasoning)>)\s*/gi, "")
		.trim();
}

/**
 * Compose the post. The model writes the hook when it is reachable; the
 * deterministic fallback keeps this working when it is not, because a failed
 * announcement should never be the reason a release is held up.
 */
async function compose(entry: { title: string; subtitle: string; tags: string[] }): Promise<string> {
	const fallback = `${entry.title}\n\n${entry.subtitle}`;

	try {
		const written = await ask({
			system:
				"You write short posts for Threads announcing technical explainers. " +
				"Plain, concrete, no hype, no hashtag spam, at most one emoji and only if it earns its place. " +
				"You never claim a result the material you were given does not state.",
			user: [
				"Announce this new entry on paperlens.io.",
				"",
				`Title: ${entry.title}`,
				`Summary: ${entry.subtitle}`,
				`Topics: ${entry.tags.join(", ")}`,
				"",
				"Rules:",
				"- Under 320 characters. Going over risks being cut mid-sentence.",
				"- Open with the concrete surprising thing, not 'New post:'.",
				"- Mention that the page has an interactive simulation you can play with.",
				"- Do NOT include a URL; the link is attached separately.",
				"- Return the post text only, no quotes around it, no commentary.",
			].join("\n"),
			// Generous, because a reasoning model that runs out mid-thought returns
			// its preamble as the answer. The length limit is enforced below, not
			// by starving the model.
			maxTokens: 4000,
			temperature: 0.6,
			retries: 1,
		});

		const text = stripReasoning(written).replace(/^["']|["']$/g, "").trim();

		if (!text) {
			console.error("  ! nothing left after stripping reasoning; using the fallback");
		} else if (looksLikeReasoning(text)) {
			console.error(`  ! the reply reads as reasoning, not a post (matched /${looksLikeReasoning(text)}/)`);
			console.error(`    "${text.slice(0, 120)}…"`);
			console.error("    using the fallback rather than publishing it");
		} else if (bytes(text) > LIMIT) {
			// Never publish clamped output: truncation is exactly how a half-finished
			// thought reaches the timeline.
			console.error(`  ! composed ${bytes(text)} bytes, over the ${LIMIT} limit — using the fallback`);
		} else {
			return text;
		}
	} catch (err) {
		console.error(`  ! could not compose with the model (${(err as Error).message}); using the fallback`);
	}

	return clampBytes(fallback, LIMIT);
}

async function main() {
	// Deleting needs the threads_delete scope, which a publish-only token lacks.
	const deleteAt = Bun.argv.indexOf("--delete");
	if (deleteAt !== -1) {
		const id = Bun.argv[deleteAt + 1];
		const token = process.env.THREADS_ACCESS_TOKEN?.trim();
		if (!id || !token) {
			console.error("usage: bun scripts/announce.ts --delete <media-id>   (needs THREADS_ACCESS_TOKEN)");
			process.exit(2);
		}
		const res = await fetch(`${GRAPH}/${id}?access_token=${encodeURIComponent(token)}`, { method: "DELETE" });
		const body = (await res.json().catch(() => ({}))) as any;
		if (!res.ok) {
			console.error(`Delete failed (${res.status}): ${body?.error?.message ?? JSON.stringify(body)}`);
			if (/permission|scope/i.test(body?.error?.message ?? "")) {
				console.error(
					"\nThe token needs the threads_delete scope. Re-authorize with\n" +
						"  scope=threads_basic,threads_content_publish,threads_delete\n" +
						"or delete the post in the Threads app, which is faster for a one-off.",
				);
			}
			process.exit(1);
		}
		console.log(`deleted ${id}`);
		return;
	}

	const slug = Bun.argv[2];
	const post = Bun.argv.includes("--post");

	if (!slug || slug.startsWith("--")) {
		console.error("usage: bun scripts/announce.ts <slug> [--post]");
		process.exit(2);
	}

	const entry = (await catalogue()).find((e) => e.slug === slug);
	if (!entry) {
		console.error(`No entry with slug "${slug}" in src/content/ideas.`);
		process.exit(1);
	}

	const url = `${SITE}/idea/${entry.slug}/`;

	// Never announce a page that is not actually live.
	const live = await fetch(url, { method: "HEAD" }).catch(() => null);
	if (!live?.ok) {
		console.error(`${url} returned ${live?.status ?? "no response"} — refusing to announce a page that is not live.`);
		process.exit(1);
	}

	const text = await compose(entry);

	// Last line of defence before anything leaves the machine.
	const suspicious = looksLikeReasoning(text);
	if (suspicious) {
		console.error(`Refusing to publish: the composed text still reads as reasoning (/${suspicious}/).`);
		console.error(text);
		process.exit(1);
	}

	const rule = "─".repeat(56);
	console.error(`\n${rule}\n${text}\n${rule}`);
	console.error(`${bytes(text)}/${LIMIT} bytes · link: ${url}\n`);

	// stdout stays machine-readable for the workflow.
	console.log(JSON.stringify({ slug: entry.slug, url, text, bytes: bytes(text) }));

	if (!post) {
		console.error("Dry run. Pass --post to publish.");
		return;
	}

	const token = process.env.THREADS_ACCESS_TOKEN?.trim();
	if (!token) {
		console.error("THREADS_ACCESS_TOKEN is not set; nothing published.");
		return;
	}

	const userId = await resolveUserId(token);

	const container = await threads(`/${userId}/threads`, {
		media_type: "TEXT",
		text,
		link_attachment: url,
		access_token: token,
	});
	if (!container?.id) throw new Error(`No container id in the response: ${JSON.stringify(container)}`);

	// Meta recommends a short pause between creating and publishing a container.
	await new Promise((r) => setTimeout(r, 3000));

	const published = await threads(`/${userId}/threads_publish`, {
		creation_id: container.id,
		access_token: token,
	});

	console.error(`published: ${published?.id ?? "(no id returned)"}`);
	console.log(JSON.stringify({ ok: true, slug, url, threadsId: published?.id }));
}

if (import.meta.main) {
	main().catch((err) => {
		console.error(err instanceof Error ? err.message : String(err));
		process.exit(1);
	});
}
