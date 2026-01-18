import { cp, rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { IDEAS } from "../generated-ideas";
import { USER_CONFIG } from "../config";

const OUT_DIR = "dist";
const BASE_URL = "https://paperlens.io"; // Updated based on user request

// 1. Clean output directory
console.log("Cleaning output directory...");
await rm(OUT_DIR, { recursive: true, force: true });

// 2. Build the application
console.log("Building application...");
const result = await Bun.build({
	entrypoints: ["./index.tsx"],
	outdir: OUT_DIR,
	minify: true,
	target: "browser",
	// We don't need to split chunks manually as Bun does a good job,
	// but for a single page app, a single bundle is often fine or default splitting.
	naming: "index.js",
});

if (!result.success) {
	console.error("Build failed");
	for (const message of result.logs) {
		console.error(message);
	}
	process.exit(1);
}

// 3. Copy static assets
console.log("Copying static assets...");

// Copy public folder contents if it exists
try {
	await cp("public", OUT_DIR, { recursive: true });
} catch (e) {
	// It's okay if public folder doesn't exist or is empty
	console.log("No public folder found or error copying, skipping...");
}

// 4. Process and copy index.html
console.log("Processing index.html...");
const indexHtml = await Bun.file("index.html").text();

// Create new File with updated script tag
// We replace the .tsx script with the bundled .js script
// Use absolute path for script to support nested routes (SSG)
let updatedHtml = indexHtml.replace(
	/<script type="module" src=".*index\.tsx"><\/script>/,
	'<script type="module" src="/index.js"></script>',
);

// Remove importmap if it exists
updatedHtml = updatedHtml.replace(
	/<script type="importmap">[\s\S]*?<\/script>/,
	"",
);

// Handle index.css
const cssFile = Bun.file("index.css");
if (await cssFile.exists()) {
	try {
		await cp("index.css", join(OUT_DIR, "index.css"));
	} catch (e) {
		console.error("Failed to copy index.css", e);
	}
} else {
	console.log("index.css not found, removing link...");
	updatedHtml = updatedHtml.replace(
		/<link rel="stylesheet" href="\/index.css">/,
		"",
	);
}

// 5. Generate Static Routes for SEO (SSG)
console.log("Generating static routes for SEO...");

function injectMeta(html: string, { title, description, image, url, noindex = false }: { title: string; description: string; image: string; url: string; noindex?: boolean }) {
	let newHtml = html;
	// Replace title
	newHtml = newHtml.replace(/<title>.*<\/title>/, `<title>${title}</title>`);

	const replaceMeta = (
		attr: "name" | "property",
		key: string,
		content: string,
	) => {
		const escapedContent = content.replace(/"/g, "&quot;");
		const regex = new RegExp(`<meta ${attr}="${key}" content=".*?" />`);
		if (regex.test(newHtml)) {
			newHtml = newHtml.replace(
				regex,
				`<meta ${attr}="${key}" content="${escapedContent}" />`,
			);
		} else {
			newHtml = newHtml.replace(
				"</head>",
				`    <meta ${attr}="${key}" content="${escapedContent}" />\n  </head>`,
			);
		}
	};

	// Add canonical link
	const canonicalLink = `<link rel="canonical" href="${url}" />`;
	newHtml = newHtml.replace("</head>", `    ${canonicalLink}\n  </head>`);

	// Add robots meta tag
	const robotsContent = noindex ? "noindex, nofollow" : "index, follow";
	replaceMeta("name", "robots", robotsContent);

	replaceMeta("name", "description", description);
	replaceMeta("property", "og:title", title);
	replaceMeta("property", "og:description", description);
	replaceMeta("property", "og:image", image);
	replaceMeta("property", "og:type", "article");
	replaceMeta("property", "og:url", url);
	replaceMeta("name", "twitter:card", "summary_large_image");
	replaceMeta("name", "twitter:title", title);
	replaceMeta("name", "twitter:description", description);
	replaceMeta("name", "twitter:image", image);

	return newHtml;
}

// Generate idea pages
for (const idea of IDEAS) {
	const ideaDir = join(OUT_DIR, "idea", idea.id);
	await mkdir(ideaDir, { recursive: true });

	const ideaHtml = injectMeta(updatedHtml, {
		title: `${idea.title} | ${USER_CONFIG.lab}`,
		description: idea.subtitle,
		image: idea.coverImage,
		url: `${BASE_URL}/idea/${idea.id}/`,
	});

	await Bun.write(join(ideaDir, "index.html"), ideaHtml);
}

// Create 404.html for GitHub Pages SPA fallback (with noindex to prevent duplicate content)
const notFoundHtml = injectMeta(updatedHtml, {
	title: `Page Not Found | ${USER_CONFIG.lab}`,
	description: "The page you're looking for doesn't exist.",
	image: USER_CONFIG.avatar,
	url: BASE_URL,
	noindex: true,
});
await Bun.write(join(OUT_DIR, "404.html"), notFoundHtml);

// Create index.html with proper homepage SEO
const indexHtmlWithMeta = injectMeta(updatedHtml, {
	title: `${USER_CONFIG.name} | ${USER_CONFIG.lab}`,
	description: USER_CONFIG.bio.replace("%NAME%", USER_CONFIG.name),
	image: USER_CONFIG.avatar,
	url: `${BASE_URL}/`,
});
await Bun.write(join(OUT_DIR, "index.html"), indexHtmlWithMeta);

// 6. Generate Sitemap & Robots.txt
console.log("Generating sitemap.xml and robots.txt...");

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${BASE_URL}/</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
${IDEAS.map(
	(idea) => `  <url>
    <loc>${BASE_URL}/idea/${idea.id}/</loc>
    <lastmod>${idea.date}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>`,
).join("\n")}
</urlset>`;

await Bun.write(join(OUT_DIR, "sitemap.xml"), sitemap);

const robotsTxt = `User-agent: *
Allow: /
Sitemap: ${BASE_URL}/sitemap.xml
`;

await Bun.write(join(OUT_DIR, "robots.txt"), robotsTxt);

// 7. Generate RSS Feed
console.log("Generating RSS feed...");

const rss = `<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
  <title>${USER_CONFIG.lab}</title>
  <link>${BASE_URL}</link>
  <description>${USER_CONFIG.bio.replace("%NAME%", USER_CONFIG.name)}</description>
  <language>en-us</language>
  <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
  <atom:link href="${BASE_URL}/rss.xml" rel="self" type="application/rss+xml" />
  ${IDEAS.map(
		(idea) => `
  <item>
    <title><![CDATA[${idea.title}]]></title>
    <link>${BASE_URL}/idea/${idea.id}/</link>
    <guid>${BASE_URL}/idea/${idea.id}/</guid>
    <pubDate>${new Date(idea.date).toUTCString()}</pubDate>
    <description><![CDATA[${idea.subtitle}]]></description>
  </item>`,
	).join("")}
</channel>
</rss>`;

await Bun.write(join(OUT_DIR, "rss.xml"), rss);

console.log("Build complete!");
