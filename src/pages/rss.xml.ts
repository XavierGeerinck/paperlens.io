import { existsSync, statSync } from 'node:fs';
import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

/** readers show the social card next to the item; RSS wants a real byte length */
function enclosure(site: string, slug: string): string {
  const file = `public/og/${slug}.png`;
  if (!existsSync(file)) return '';
  return `<enclosure url="${site}/og/${slug}.png" type="image/png" length="${statSync(file).size}"/>`;
}

export async function GET(context: APIContext) {
  const site = context.site?.href.replace(/\/$/, '') ?? 'https://paperlens.io';

  const ideas = await getCollection('ideas');
  const sortedIdeas = ideas.sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );

  return rss({
    title: 'PaperLens — papers and research for AI engineers, visualized',
    description:
      "Every entry rebuilds a paper's core mechanism as an interactive simulation, next to the math and the code.",
    site,
    xmlns: { atom: 'http://www.w3.org/2005/Atom' },
    // a self link is what an aggregator uses to re-find the feed after a move
    customData: [
      '<language>en-us</language>',
      `<atom:link href="${site}/rss.xml" rel="self" type="application/rss+xml"/>`,
      '<managingEditor>xavier@m18x.com (Xavier Geerinck)</managingEditor>',
    ].join(''),
    items: sortedIdeas.map((idea) => ({
      title: idea.data.title,
      pubDate: idea.data.date,
      description: idea.data.subtitle,
      link: `/idea/${idea.slug}/`,
      categories: idea.data.tags,
      customData: enclosure(site, idea.slug),
    })),
  });
}
