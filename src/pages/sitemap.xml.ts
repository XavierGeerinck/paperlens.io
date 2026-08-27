import type { APIRoute } from "astro";

/**
 * /sitemap.xml — an alias for the index @astrojs/sitemap writes.
 *
 * robots.txt points at /sitemap-index.xml, but /sitemap.xml is where crawlers
 * and humans look first, and Search Console may already have it on file. This
 * used to read dist/sitemap-index.xml off disk while the build was still
 * writing it; the index is two lines, so it is simply emitted here.
 */
export const GET: APIRoute = ({ site }) => {
	const base = (site?.href ?? "https://paperlens.io/").replace(/\/$/, "");

	const body = `<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>${base}/sitemap-0.xml</loc></sitemap>
</sitemapindex>
`;

	return new Response(body, {
		headers: { "Content-Type": "application/xml; charset=utf-8" },
	});
};
