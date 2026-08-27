#!/usr/bin/env bun
/**
 * Everything the static build needs in `public/` before Astro runs.
 *
 *   1. KaTeX, copied out of node_modules. The page used to pull 0.16.9 off
 *      jsdelivr while rehype-katex generated markup with whatever version is
 *      installed — a third-party origin on the critical path *and* a version
 *      skew. Serving the installed copy fixes both.
 *   2. The social cards (scripts/og.ts).
 *
 * Both outputs are generated, so they are gitignored: this runs as part of
 * `bun run build`.
 */

import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const KATEX_SRC = "node_modules/katex/dist";
const KATEX_OUT = "public/katex";

async function vendorKatex() {
	if (!existsSync(KATEX_SRC)) throw new Error(`missing ${KATEX_SRC} — run bun install`);

	mkdirSync(join(KATEX_OUT, "fonts"), { recursive: true });

	const copy = (from: string, to: string) => {
		if (existsSync(to) && statSync(to).mtimeMs >= statSync(from).mtimeMs) return 0;
		copyFileSync(from, to);
		return 1;
	};

	let copied = copy(join(KATEX_SRC, "katex.min.css"), join(KATEX_OUT, "katex.min.css"));

	// katex.min.css lists woff2 first, then woff and ttf; every browser that can
	// run the site takes the woff2, so the other two formats are left behind.
	for (const font of readdirSync(join(KATEX_SRC, "fonts")).filter((f) => f.endsWith(".woff2"))) {
		copied += copy(join(KATEX_SRC, "fonts", font), join(KATEX_OUT, "fonts", font));
	}

	const version = JSON.parse(
		await Bun.file("node_modules/katex/package.json").text(),
	).version as string;
	console.error(`# katex ${version}: ${copied ? `vendored ${copied} file(s)` : "up to date"}`);
}

await vendorKatex();
await import("./og.ts");
