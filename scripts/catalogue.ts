#!/usr/bin/env bun
/**
 * Catalogue index for PaperLens.
 *
 * Prints what the site already covers so a scan can dedupe against it without
 * re-reading 28 markdown files. arXiv IDs are pulled from `pdfUrl` and from any
 * `arXiv:NNNN.NNNNN` mention in the body, so a paper cited inside an entry
 * counts as covered even when it isn't the entry's headline source.
 *
 *   bun scripts/catalogue.ts              # human-readable table
 *   bun scripts/catalogue.ts --json       # machine-readable
 *   bun scripts/catalogue.ts --ids        # bare arXiv IDs, one per line
 *   bun scripts/catalogue.ts --check 2608.17981 2607.09225
 */

import { readdir } from 'node:fs/promises';
import { join } from 'node:path';

const CONTENT_DIR = 'src/content/ideas';
const SIM_DIR = 'src/components/react/simulations';

// arXiv IDs are NNNN.NNNNN (5 digits since 2015), optionally with a version suffix.
const ARXIV_RE = /(?:arxiv\.org\/(?:abs|pdf)\/|arXiv:\s*)(\d{4}\.\d{4,5})/gi;

export interface Entry {
  slug: string;
  file: string;
  title: string;
  subtitle: string;
  date: string;
  category: string;
  status: string;
  tags: string[];
  simulation?: string;
  /** every arXiv ID the entry references, headline source first */
  arxiv: string[];
}

/** Minimal frontmatter reader — enough for this schema, no YAML dependency. */
function parseFrontmatter(raw: string): Record<string, string | string[]> {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return {};

  const out: Record<string, string | string[]> = {};
  let listKey: string | null = null;

  for (const line of match[1].split(/\r?\n/)) {
    const item = line.match(/^\s*-\s+(.*)$/);
    if (item && listKey) {
      (out[listKey] as string[]).push(clean(item[1]));
      continue;
    }

    const kv = line.match(/^([A-Za-z][\w-]*):\s*(.*)$/);
    if (!kv) continue;

    const [, key, rest] = kv;
    if (rest.trim() === '') {
      listKey = key;
      out[key] = [];
    } else {
      listKey = null;
      out[key] = clean(rest);
    }
  }
  return out;
}

/** Strip quotes and trailing `# comments` that several entries carry on pdfUrl. */
function clean(value: string): string {
  return value
    .replace(/\s+#\s.*$/, '')
    .trim()
    .replace(/^["'](.*)["']$/, '$1')
    .trim();
}

function str(v: string | string[] | undefined, fallback = ''): string {
  return typeof v === 'string' ? v : fallback;
}

export async function catalogue(): Promise<Entry[]> {
  const files = (await readdir(CONTENT_DIR)).filter((f) => /\.mdx?$/.test(f)).sort();

  const entries = await Promise.all(
    files.map(async (file) => {
      const raw = await Bun.file(join(CONTENT_DIR, file)).text();
      const fm = parseFrontmatter(raw);

      // Headline sources first (pdfUrl may hold a comma-separated list), then body mentions.
      const ids: string[] = [];
      const add = (text: string) => {
        for (const [, id] of text.matchAll(ARXIV_RE)) if (!ids.includes(id)) ids.push(id);
      };
      add(str(fm.pdfUrl));
      add(raw);

      return {
        slug: file.replace(/\.mdx?$/, ''),
        file: join(CONTENT_DIR, file),
        title: str(fm.title),
        subtitle: str(fm.subtitle),
        date: str(fm.date),
        category: str(fm.category, 'idea'),
        status: str(fm.status),
        tags: Array.isArray(fm.tags) ? fm.tags : [],
        simulation: str(fm.simulation) || undefined,
        arxiv: ids,
      } satisfies Entry;
    }),
  );

  return entries.sort((a, b) => b.date.localeCompare(a.date));
}

/** Words that carry no signal when comparing a paper title to the catalogue. */
const STOP = new Set(
  ('a an the of for with and or to in on via using is are be beyond towards toward how why what ' +
    'model models learning neural network networks deep large language llm llms training train ' +
    'efficient scalable novel new framework approach method methods can does do we our').split(' '),
);

function keywords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, ' ')
      .split(/\s+/)
      .filter((w) => w.length > 2 && !STOP.has(w)),
  );
}

/**
 * Overlap between a candidate title and an entry, as a fraction of the
 * candidate's distinctive words. High overlap means "look closer", never
 * "definitely a duplicate" — the judgement stays with the caller.
 */
export function similarity(candidateTitle: string, entry: Entry): number {
  const a = keywords(candidateTitle);
  if (a.size === 0) return 0;
  const b = keywords(`${entry.title} ${entry.subtitle} ${entry.tags.join(' ')}`);
  let shared = 0;
  for (const w of a) if (b.has(w)) shared += 1;
  return shared / a.size;
}

// --- CLI -------------------------------------------------------------------

if (import.meta.main) {
  const argv = Bun.argv.slice(2);
  const entries = await catalogue();

  const checkAt = argv.indexOf('--check');
  if (checkAt !== -1) {
    const wanted = argv.slice(checkAt + 1).map((a) => a.replace(/v\d+$/, '').trim());
    if (wanted.length === 0) {
      console.error('usage: bun scripts/catalogue.ts --check <arxivId> [...]');
      process.exit(2);
    }
    let anyCovered = false;
    for (const id of wanted) {
      const hit = entries.find((e) => e.arxiv.includes(id));
      if (hit) {
        anyCovered = true;
        console.log(`COVERED  ${id}  ${hit.slug}  "${hit.title}"`);
      } else {
        console.log(`NEW      ${id}`);
      }
    }
    process.exit(anyCovered ? 1 : 0);
  }

  if (argv.includes('--json')) {
    console.log(JSON.stringify(entries, null, 2));
  } else if (argv.includes('--ids')) {
    console.log([...new Set(entries.flatMap((e) => e.arxiv))].sort().join('\n'));
  } else {
    const sims = new Set(
      (await readdir(SIM_DIR)).filter((f) => f.endsWith('.tsx')).map((f) => f.replace(/\.tsx$/, '')),
    );
    console.log(`# ${entries.length} entries · ${sims.size} simulations\n`);
    for (const e of entries) {
      const ids = e.arxiv.length ? e.arxiv.join(', ') : '—';
      console.log(`${e.date}  ${e.category.padEnd(9)}  ${e.slug}`);
      console.log(`            ${e.title}`);
      console.log(`            arxiv: ${ids}`);
      console.log(`            tags: ${e.tags.join(', ') || '—'}\n`);
    }
  }
}
