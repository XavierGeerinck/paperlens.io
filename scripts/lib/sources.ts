/**
 * Candidate harvesting.
 *
 * Two sources, because they surface different things: HuggingFace's daily
 * papers carry community attention (which is the search-demand signal the
 * rubric weights double), and the arXiv API carries the quiet architecture
 * papers that never trend. Both are real APIs — nothing here scrapes HTML.
 */

export interface Candidate {
	/** bare arXiv ID, no version suffix */
	id: string;
	title: string;
	abstract: string;
	/** HuggingFace upvotes, when the paper appeared there */
	upvotes?: number;
	published: string;
	sources: string[];
}

const ID_RE = /^\d{4}\.\d{4,5}$/;

/**
 * arXiv rejects unidentified callers with 429, so every request carries a
 * User-Agent naming the project. Transient 429/503 still get a backoff.
 */
const UA = "paperlens.io/1.0 (+https://paperlens.io)";

async function arxivFetch(url: string, attempts = 4): Promise<Response> {
	let last: Response | undefined;
	for (let i = 0; i < attempts; i++) {
		const res = await fetch(url, { headers: { "User-Agent": UA, Accept: "application/atom+xml" } });
		if (res.ok) return res;
		last = res;
		if (res.status !== 429 && res.status !== 503) return res;
		await new Promise((r) => setTimeout(r, 3000 * (i + 1)));
	}
	return last!;
}

function clean(s: string): string {
	return s.replace(/\s+/g, " ").trim();
}

/** Community attention over the last `days` days. */
export async function fromHuggingFace(days: number): Promise<Candidate[]> {
	const cutoff = Date.now() - days * 86_400_000;
	const out: Candidate[] = [];

	try {
		const res = await fetch("https://huggingface.co/api/daily_papers?limit=100");
		if (!res.ok) {
			console.warn(`  ! huggingface returned ${res.status}; continuing without it`);
			return out;
		}
		const rows = (await res.json()) as any[];

		for (const row of rows) {
			const p = row?.paper;
			if (!p?.id || !ID_RE.test(p.id)) continue;
			const when = Date.parse(p.publishedAt ?? row.publishedAt ?? "");
			if (Number.isFinite(when) && when < cutoff) continue;

			out.push({
				id: p.id,
				title: clean(p.title ?? row.title ?? ""),
				abstract: clean(p.summary ?? row.summary ?? ""),
				upvotes: typeof p.upvotes === "number" ? p.upvotes : 0,
				published: (p.publishedAt ?? "").slice(0, 10),
				sources: ["huggingface"],
			});
		}
	} catch (err) {
		console.warn(`  ! huggingface harvest failed: ${(err as Error).message}`);
	}

	return out;
}

/** The firehose, newest first, one query per category. */
export async function fromArxiv(categories: string[], perCategory = 60): Promise<Candidate[]> {
	const out: Candidate[] = [];

	for (const cat of categories) {
		const url =
			"https://export.arxiv.org/api/query?" +
			new URLSearchParams({
				search_query: `cat:${cat}`,
				sortBy: "submittedDate",
				sortOrder: "descending",
				max_results: String(perCategory),
			});

		try {
			const res = await arxivFetch(url);
			if (!res.ok) {
				console.warn(`  ! arxiv ${cat} returned ${res.status}; skipping`);
				continue;
			}
			const xml = await res.text();

			for (const entry of xml.split("<entry>").slice(1)) {
				const id = entry.match(/<id>https?:\/\/arxiv\.org\/abs\/([^<v]+)/)?.[1];
				const title = entry.match(/<title>([\s\S]*?)<\/title>/)?.[1];
				const abstract = entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1];
				const published = entry.match(/<published>([^<]+)<\/published>/)?.[1];
				if (!id || !ID_RE.test(id) || !title) continue;

				out.push({
					id,
					title: clean(title),
					abstract: clean(abstract ?? ""),
					published: (published ?? "").slice(0, 10),
					sources: [`arxiv:${cat}`],
				});
			}
			// The arXiv API asks callers to space out requests.
			await new Promise((r) => setTimeout(r, 3000));
		} catch (err) {
			console.warn(`  ! arxiv ${cat} harvest failed: ${(err as Error).message}`);
		}
	}

	return out;
}

/** Merge by arXiv ID, keeping the richest record and unioning the sources. */
export function merge(...lists: Candidate[][]): Candidate[] {
	const byId = new Map<string, Candidate>();

	for (const c of lists.flat()) {
		const existing = byId.get(c.id);
		if (!existing) {
			byId.set(c.id, { ...c, sources: [...c.sources] });
			continue;
		}
		existing.sources = [...new Set([...existing.sources, ...c.sources])];
		if ((c.upvotes ?? 0) > (existing.upvotes ?? 0)) existing.upvotes = c.upvotes;
		if (c.abstract.length > existing.abstract.length) existing.abstract = c.abstract;
		if (!existing.published && c.published) existing.published = c.published;
	}

	return [...byId.values()].sort((a, b) => (b.upvotes ?? 0) - (a.upvotes ?? 0));
}

/** Authoritative record for one paper, straight from the arXiv API. */
export async function fetchPaper(id: string): Promise<Candidate & { authors: string[] }> {
	const bare = id.replace(/v\d+$/, "").trim();
	const res = await arxivFetch(
		`https://export.arxiv.org/api/query?${new URLSearchParams({ id_list: bare, max_results: "1" })}`,
	);
	if (!res.ok) throw new Error(`arXiv API returned ${res.status} for ${bare}`);

	const xml = await res.text();
	const entry = xml.split("<entry>")[1];
	if (!entry) throw new Error(`arXiv has no record for ${bare}`);

	const title = entry.match(/<title>([\s\S]*?)<\/title>/)?.[1];
	const abstract = entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1];
	const published = entry.match(/<published>([^<]+)<\/published>/)?.[1];
	const authors = [...entry.matchAll(/<name>([^<]+)<\/name>/g)].map((m) => clean(m[1]));

	if (!title || !abstract) throw new Error(`arXiv record for ${bare} is missing title or abstract`);

	return {
		id: bare,
		title: clean(title),
		abstract: clean(abstract),
		published: (published ?? "").slice(0, 10),
		authors,
		sources: ["arxiv"],
	};
}
