import type { APIRoute } from "astro";
import fs from "node:fs";
import path from "node:path";

export const GET: APIRoute = async () => {
	// Read the auto-generated sitemap-index.xml from dist
	const sitemapIndexPath = path.join(process.cwd(), "dist", "sitemap-index.xml");
	
	let sitemap: string;
	try {
		sitemap = fs.readFileSync(sitemapIndexPath, "utf-8");
	} catch {
		// Fallback if file doesn't exist yet
		sitemap = `<?xml version="1.0" encoding="UTF-8"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://paperlens.io/sitemap-0.xml</loc></sitemap></sitemapindex>`;
	}

	return new Response(sitemap, {
		headers: {
			"Content-Type": "application/xml; charset=utf-8",
		},
	});
};
