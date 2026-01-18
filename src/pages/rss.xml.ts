import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const ideas = await getCollection('ideas');
  const sortedIdeas = ideas.sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );

  return rss({
    title: 'PaperLens - Xavier Geerinck',
    description: 'Research and ideas on AI, systems, and emerging technologies',
    site: context.site ?? 'https://paperlens.io',
    items: sortedIdeas.map((idea) => ({
      title: idea.data.title,
      pubDate: idea.data.date,
      description: idea.data.subtitle,
      link: `/idea/${idea.slug}/`,
    })),
  });
}
