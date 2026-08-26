---
description: Scan this week's AI research, drop what PaperLens already covers, and publish the best remaining paper as an entry with a working simulation
argument-hint: "[--shortlist] [--week 2026-W35] [--pr] [--dry-run] [arxiv-id] [topic]"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch
---

# scan-papers

Arguments: `$ARGUMENTS`

Today is !`date -u +%Y-%m-%d`, ISO week !`date +%G-W%V`.

The site already covers:

!`bun scripts/catalogue.ts`

## Modes

Read the arguments above and pick one. Default (no arguments) is a full run.

| Argument | Behaviour |
|---|---|
| *(none)* | Full run: harvest → dedupe → rank → build → verify → ship |
| `--shortlist` | Stop after ranking. Print the table, write nothing. Use this when the user wants to choose. |
| `--week 2026-W34` | Harvest that ISO week instead of the current one |
| `2608.17981` | Skip harvesting; build that arXiv ID directly |
| a topic, e.g. `world models` | Bias harvesting and ranking toward that topic |
| `--pr` | Ship on a branch and open a pull request instead of pushing to `main` |
| `--dry-run` | Build and verify, then stop. Nothing is committed. |

## 1 · Harvest

Pull candidates from all three sources — they surface different things, and one
source alone will miss the paper worth writing about:

1. **Trending** — `https://huggingface.co/papers/week/<ISO-week>` for what the
   community actually read, and `https://huggingface.co/papers` for today.
2. **Firehose** — `https://arxiv.org/list/cs.LG/recent`, `cs.AI/recent`,
   `cs.CL/recent`. Trending misses quiet architecture papers; this catches them.
3. **Targeted** — a `WebSearch` for the site's recurring themes (attention and
   KV-cache efficiency, world models, test-time compute, optimizer dynamics,
   interpretability, inference hardware) restricted to the last week.

Aim for 25–40 raw candidates before filtering. Record title and arXiv ID for each.

## 2 · Dedupe

Run `bun scripts/catalogue.ts --check <id> <id> ...` on every candidate. It exits
non-zero if any is already covered.

An exact ID match is a hard drop. Then apply judgement: a paper is also a
duplicate when its **core mechanism** is already the subject of an entry, even
under a different name and ID. Check the catalogue table above for adjacency —
proximity is not duplication. An entry on JEPA loss design does not cover a paper
on looped world-model depth; they answer different questions and cross-linking
them makes both stronger. Say which existing entry you checked against and why
the candidate survived or didn't.

## 3 · Rank

Score each survivor 0–3 on five axes. This rubric is the site's identity — a
paper that scores well on citations but badly here is the wrong paper.

**The site exists to be found.** Search demand is the heaviest axis: an entry
nobody searches for is an entry nobody reads, however elegant the mechanism.

- **Search demand (weight ×2).** Will people be typing this into a search engine
  over the next six months, and can this page be the best answer they find?
  Evidence, in descending order of weight: the paper has a *named* artefact
  people can spell (`Recirculation`, `GigaBrain`, `Tapered Language Models`)
  rather than a descriptive sentence title; it is trending on HuggingFace or
  being argued about on social; a lab or model people already follow is behind
  it; and there is no good explainer ranking for it yet. Check that last one —
  `WebSearch` the paper's name and see what already exists. A crowded first page
  of results is a reason to pick the runner-up. Score 3 for a named thing that is
  trending with nothing but the abstract to read; 0 for a descriptive title on a
  topic nobody is looking up.
- **Simulatable.** Can the mechanism be shown as a live toy in a few hundred
  lines of React, where moving a slider makes the point? This is what the site
  offers that a summary blog cannot, and it is why people link to it. If the only
  honest demo is a bar chart of benchmark scores, score 0.
- **One mechanism.** Is there a single crisp idea a reader can hold in their
  head? "Train only the middle layer" scores 3. "We built a 12-component agent
  platform" scores 0.
- **Overturns an assumption.** Does it contradict something a working engineer
  currently believes — that all layers matter equally, that uniform width is
  optimal, that recurrence needs retraining? This is the axis that keeps an entry
  worth reading, and being linked to, a year later.
- **Fills a gap.** Does it extend the catalogue into territory it doesn't cover,
  or add a genuinely new angle to a cluster it does? Entries that cross-link into
  an existing cluster pull traffic through the whole cluster.

Hard exclusions: dataset releases, surveys, and pure leaderboard papers where the
contribution is a number rather than an idea.

One deliberate exception. A **named model or system report with very high search
demand** — the kind of release people will be searching for by name all month —
qualifies *if* you can isolate one real mechanism inside it worth simulating, and
the entry is built around that mechanism. Write about the idea, using the name
people are searching for. If there is no mechanism to isolate, skip it: a page
that just restates a release note will not rank and does not belong here.

Present the shortlist as a table — ID, title, the five scores with search demand
doubled, total, and a one-sentence reason. Then state your pick, why it beat the
runner-up, and what search query this page is trying to win.

## 4 · Verify the source

Before writing a single word, `WebFetch` the arXiv abstract page for the pick and
read the actual abstract. Never write from a title, a trending blurb, or a
summary someone else wrote — those are how wrong claims get published.

Confirm the title, authors, submission date, and the specific numbers you intend
to quote. If the abstract does not support a claim, drop the claim. If the
abstract is thin, fetch the PDF or the project page for the mechanism's details.

## 5 · Write the entry

Follow `AGENTS.md` — it is the authority on content structure, frontmatter,
simulation standards, and visual guidelines. In particular:

- `src/content/ideas/<slug>.md` (or `.mdx` if you need `AlgorithmBlock` or other
  components). Frontmatter must validate against `src/content/config.ts` —
  `status` is one of RESEARCH / CONCEPT / PROTOTYPE / ALPHA / ARCHIVED, and
  `category` one of idea / paper / deep-dive / tutorial / concept.
- `pdfUrl` points at the real arXiv URL. `date` is the paper's submission date.
- `src/components/react/simulations/<Name>Simulation.tsx`, registered by filename
  alone — `DemoView` globs the directory, so no registry edit is needed.
- Cross-link to the adjacent entries you found in step 2. That is what makes this
  a catalogue rather than a pile of posts.

**Write it to be found.** The rubric picked this paper for search demand; the
entry has to earn it:

- `title` leads with the name people will search — the artefact name, then the
  plain-language hook. `"Recirculation: Making a Transformer Think Without
  Tokens"`, not `"On Inference-Time Belief Tracking"`.
- `subtitle` is the meta description. One sentence, under 160 characters, that
  answers the query rather than teasing it.
- `tags` are query terms, not vibes — the model name, the lab, the mechanism, the
  problem it solves. These drive the site's own tag histogram and related-entry
  links.
- Use the paper's own vocabulary in headings. Someone searching a term from the
  abstract should land on a heading that uses it.
- Answer the obvious follow-up questions as headings: what it is, why the old way
  failed, what it costs, whether it works at scale.
- Cross-link both directions — add a link from the adjacent existing entries to
  this one, not just from this one outward. That is what turns 28 pages into a
  catalogue search engines rank as one.

Two rules the design brief adds on top of `AGENTS.md`:

- **Style through the tokens.** The site is a dark terminal — build on the
  semantic Tailwind colours in `tailwind.config.mjs` (`bg0`, `ink`, `mint`,
  `amber`, `azure`, `iris`) and the component classes in `src/styles/global.css`.
  No new hex values, no light theme.
- **Seed every random simulation.** Use the `rng(seed)` mulberry32 helper the
  existing simulations share, so a figure looks the same on every load.

## 6 · Verify the build

Non-negotiable, in order:

1. `bun run build` — must pass clean.
2. `bun run preview`, then load the new page and check the browser console: no
   page errors, no React errors, the simulation actually mounts, and mermaid
   diagrams render. A simulation that throws on mount still builds fine, so the
   build passing is not evidence the page works.
3. Confirm no horizontal scroll at 390px wide.

If anything fails, fix it and re-verify. Do not ship a red build.

## 7 · Ship

Stage **only** the files you created:
`src/content/ideas/<slug>.md*` and `src/components/react/simulations/<Name>Simulation.tsx`.

Never `git add -A` at the repo root. The working tree carries untracked items
that must not be committed — `.leankg/` is 96 MB and `paperlens.io/` is a nested
git repository.

Commit as `feat(paper): <title>` with the standard trailers. Then:

- **Default** — push to `main`. The Pages workflow deploys automatically; watch
  it with `gh run watch` and confirm the live URL returns 200.
- **`--pr`** — branch as `paper/<slug>`, push, and `gh pr create` with the
  shortlist table in the body so the reviewer can see what else was considered.

## Report back

- The shortlist table, including what was rejected as already-covered and which
  entry covered it.
- The pick, the paper's real title and authors, and the mechanism in one sentence.
- The search query this entry is built to win, and what currently ranks for it.
- The live URL, plus what the simulation lets a reader do.
