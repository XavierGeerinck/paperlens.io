/**
 * Tags are written for humans in the frontmatter ("World Models") and used as
 * URLs everywhere else ("world-models"). One slugifier so the tag on an entry,
 * the tag in the sidebar, and /tag/world-models never drift apart.
 */

export function tagSlug(tag: string): string {
  return tag
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

export interface TagGroup<T> {
  slug: string;
  /** the spelling used most often across entries */
  label: string;
  entries: T[];
}

/** groups entries by tag slug; the input order decides the order within a group */
export function groupByTag<T extends { data: { tags: string[] } }>(entries: T[]): TagGroup<T>[] {
  const groups = new Map<string, { labels: Map<string, number>; entries: T[] }>();

  for (const entry of entries) {
    for (const tag of entry.data.tags) {
      const slug = tagSlug(tag);
      if (!slug) continue;

      const group = groups.get(slug) ?? { labels: new Map(), entries: [] };
      group.labels.set(tag, (group.labels.get(tag) ?? 0) + 1);
      if (!group.entries.includes(entry)) group.entries.push(entry);
      groups.set(slug, group);
    }
  }

  return [...groups.entries()]
    .map(([slug, group]) => ({
      slug,
      label: [...group.labels.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0],
      entries: group.entries,
    }))
    .sort((a, b) => b.entries.length - a.entries.length || a.slug.localeCompare(b.slug));
}
