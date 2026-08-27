#!/usr/bin/env bun
/**
 * Announce a published entry on Threads.
 *
 *   bun scripts/announce.ts recirculation-inference-time-recurrence   # dry run
 *   bun scripts/announce.ts <slug> --post                             # actually publish
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
				"- Under 400 characters.",
				"- Open with the concrete surprising thing, not 'New post:'.",
				"- Mention that the page has an interactive simulation you can play with.",
				"- Do NOT include a URL; the link is attached separately.",
				"- Return the post text only, no quotes around it, no commentary.",
			].join("\n"),
			maxTokens: 600,
			temperature: 0.6,
			retries: 1,
		});

		const text = written.trim().replace(/^["']|["']$/g, "");
		if (text && bytes(text) <= LIMIT) return text;
		if (text) return clampBytes(text, LIMIT);
	} catch (err) {
		console.error(`  ! could not compose with the model (${(err as Error).message}); using the fallback`);
	}

	return clampBytes(fallback, LIMIT);
}

async function main() {
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

	console.error(`\n--- post (${bytes(text)}/${LIMIT} bytes) ---`);
	console.log(text);
	console.error(`--- link attachment ---`);
	console.log(url);
	console.error("");

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

main().catch((err) => {
	console.error(err instanceof Error ? err.message : String(err));
	process.exit(1);
});
