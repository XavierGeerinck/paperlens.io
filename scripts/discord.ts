#!/usr/bin/env bun
/**
 * Announce a published entry in a Discord channel.
 *
 *   bun scripts/discord.ts <slug>           # dry run, prints the message
 *   bun scripts/discord.ts <slug> --post    # actually send it
 *
 * Sending is opt-in, the same as announce.ts.
 *
 *   DISCORD_WEBHOOK_URL  secret   a channel webhook, from the channel's
 *                                 Integrations settings
 *
 * A webhook needs no bot and no gateway connection — it is one POST. The embed
 * carries the entry's own title and subtitle instead of model-written copy, so
 * nothing here can put a model's reasoning in front of people.
 */

import { catalogue } from "./catalogue";

const SITE = "https://paperlens.io";
/** primary-400 in tailwind.config.mjs */
const MINT = 0x35d492;

/** Discord rejects the whole message when a field runs over its cap. */
function clamp(s: string, max: number): string {
	return s.length <= max ? s : s.slice(0, max - 1).trimEnd() + "…";
}

export function message(entry: { slug: string; title: string; subtitle: string; tags: string[] }) {
	return {
		username: "PaperLens",
		// Titles come from a model; none of them should be able to ping a channel.
		allowed_mentions: { parse: [] as string[] },
		embeds: [
			{
				title: clamp(entry.title, 256),
				url: `${SITE}/idea/${entry.slug}/`,
				description: clamp(entry.subtitle, 4096),
				color: MINT,
				// The social card the build draws for every entry.
				image: { url: `${SITE}/og/${entry.slug}.png` },
				...(entry.tags.length ? { footer: { text: clamp(entry.tags.slice(0, 5).join(" · "), 2048) } } : {}),
			},
		],
	};
}

async function main() {
	const slug = Bun.argv[2];
	const post = Bun.argv.includes("--post");

	if (!slug || slug.startsWith("--")) {
		console.error("usage: bun scripts/discord.ts <slug> [--post]");
		process.exit(2);
	}

	const entry = (await catalogue()).find((e) => e.slug === slug);
	if (!entry) {
		console.error(`No entry with slug "${slug}" in src/content/ideas.`);
		process.exit(1);
	}

	const payload = message(entry);
	const url = payload.embeds[0].url;

	// Never announce a page that is not actually live.
	const live = await fetch(url, { method: "HEAD" }).catch(() => null);
	if (!live?.ok) {
		console.error(`${url} returned ${live?.status ?? "no response"} — refusing to announce a page that is not live.`);
		process.exit(1);
	}

	console.log(JSON.stringify(payload));

	if (!post) {
		console.error("Dry run. Pass --post to send.");
		return;
	}

	const webhook = process.env.DISCORD_WEBHOOK_URL?.trim();
	if (!webhook) {
		console.error("DISCORD_WEBHOOK_URL is not set; nothing sent.");
		return;
	}

	// wait=true makes Discord answer with the created message rather than an
	// empty 204, so a success here means the message exists.
	const target = new URL(webhook);
	target.searchParams.set("wait", "true");
	const res = await fetch(target, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	});
	const body = (await res.json().catch(() => ({}))) as { id?: string; message?: string };
	// The webhook URL is the credential, so it never goes in an error message.
	if (!res.ok || !body.id) {
		throw new Error(`Discord webhook returned ${res.status}: ${body.message ?? "no message id"}`);
	}
	console.error(`sent to Discord: message ${body.id}`);
}

if (import.meta.main) {
	main().catch((err) => {
		console.error(err instanceof Error ? err.message : String(err));
		process.exit(1);
	});
}
