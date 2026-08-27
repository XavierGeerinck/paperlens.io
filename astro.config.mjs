import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkMermaid from './src/lib/remark-mermaid.mjs';

import mdx from '@astrojs/mdx';

/**
 * `lastmod` per entry, read straight off the frontmatter.
 *
 * The integration has no view of the content collection, so the dates are
 * picked up from disk here. Without them the sitemap tells a crawler nothing
 * about what changed, which on a weekly-publishing site is the one signal worth
 * sending.
 */
const ENTRY_DIR = 'src/content/ideas';

const entryDates = new Map(
  readdirSync(ENTRY_DIR)
    .filter((file) => file.endsWith('.md') || file.endsWith('.mdx'))
    .map((file) => {
      const source = readFileSync(join(ENTRY_DIR, file), 'utf-8');
      const date = source.match(/^date:\s*["']?(\d{4}-\d{2}-\d{2})/m)?.[1];
      return [`/idea/${file.replace(/\.mdx?$/, '')}/`, date];
    })
    .filter(([, date]) => date)
);

/** the archive as a whole is as fresh as its newest entry */
const newest = [...entryDates.values()].sort().pop();

export default defineConfig({
  site: 'https://paperlens.io',
  integrations: [
    react(),
    tailwind(),
    sitemap({
      serialize(item) {
        const path = new URL(item.url).pathname;
        const lastmod = entryDates.get(path) ?? newest;

        if (lastmod) item.lastmod = `${lastmod}T00:00:00+00:00`;
        // the home page and the topic hubs reshuffle whenever an entry lands;
        // an entry itself is written once and rarely touched again
        item.changefreq = entryDates.has(path) ? 'monthly' : 'weekly';
        item.priority = path === '/' ? 1.0 : entryDates.has(path) ? 0.8 : 0.5;

        return item;
      },
    }),
    mdx(),
  ],
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
    remarkPlugins: [remarkMath, remarkMermaid],
    rehypePlugins: [rehypeKatex],
  },
});
