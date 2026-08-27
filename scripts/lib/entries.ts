/**
 * Reads the entry collection straight off disk.
 *
 * Build-time scripts run before Astro does, so they cannot use
 * `getCollection`. The frontmatter this repo writes is a small, fixed shape —
 * scalars and one list — so a hand parser beats pulling in a YAML dependency.
 */

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const ENTRY_DIR = "src/content/ideas";

export interface Entry {
	slug: string;
	file: string;
	title: string;
	subtitle: string;
	date: string;
	status: string;
	category: string;
	impact: string;
	readTime: string;
	tags: string[];
	simulation?: string;
	coverImage?: string;
	featured: boolean;
}

/** strips the quoting `frontmatter()` in build-entry.ts applies */
function unquote(value: string): string {
	const trimmed = value.trim();
	if (
		(trimmed.startsWith('"') && trimmed.endsWith('"')) ||
		(trimmed.startsWith("'") && trimmed.endsWith("'"))
	) {
		return trimmed.slice(1, -1).replace(/\\"/g, '"');
	}
	return trimmed;
}

export function parseFrontmatter(source: string): Record<string, string | string[]> {
	const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/);
	if (!match) return {};

	const out: Record<string, string | string[]> = {};
	let listKey: string | null = null;

	for (const line of match[1].split(/\r?\n/)) {
		const item = line.match(/^\s+-\s+(.*)$/);
		if (item && listKey) {
			(out[listKey] as string[]).push(unquote(item[1]));
			continue;
		}

		const pair = line.match(/^([A-Za-z0-9_]+):\s*(.*)$/);
		if (!pair) continue;

		const [, key, rest] = pair;
		if (rest.trim() === "") {
			listKey = key;
			out[key] = [];
		} else {
			listKey = null;
			out[key] = unquote(rest);
		}
	}
	return out;
}

export function readEntries(dir = ENTRY_DIR): Entry[] {
	const files = readdirSync(dir).filter((f) => f.endsWith(".md") || f.endsWith(".mdx"));

	const entries = files.map((file) => {
		const data = parseFrontmatter(readFileSync(join(dir, file), "utf-8"));
		const str = (key: string) => (typeof data[key] === "string" ? (data[key] as string) : undefined);

		return {
			slug: file.replace(/\.mdx?$/, ""),
			file: join(dir, file),
			title: str("title") ?? file,
			subtitle: str("subtitle") ?? "",
			date: (str("date") ?? "").slice(0, 10),
			status: str("status") ?? "RESEARCH",
			category: str("category") ?? "idea",
			impact: str("impact") ?? "",
			readTime: str("readTime") ?? "",
			tags: Array.isArray(data.tags) ? (data.tags as string[]) : [],
			simulation: str("simulation"),
			coverImage: str("coverImage"),
			featured: str("featured") === "true",
		} satisfies Entry;
	});

	return entries.sort((a, b) => b.date.localeCompare(a.date));
}
