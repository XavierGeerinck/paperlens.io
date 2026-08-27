---
description: Scan this week's AI research, drop what PaperLens already covers, and publish the best remaining paper as an entry with a working simulation
argument-hint: "[--shortlist] [--days 7] [--pr] [--dry-run] [arxiv-id] [topic]"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch
---

# scan-papers

Arguments: `$ARGUMENTS`

Today is !`date -u +%Y-%m-%d`, ISO week !`date +%G-W%V`.

The site already covers:

!`bun scripts/catalogue.ts`

## The rubric

`docs/scan-rubric.md` is the single source of truth for selection. The CI scripts
read the same file, so the two paths cannot drift. Follow it exactly:

!`cat docs/scan-rubric.md`

## Modes

| Argument | Behaviour |
|---|---|
| *(none)* | Full run: harvest → dedupe → rank → build → verify → ship |
| `--shortlist` | Stop after ranking. Print the table, write nothing. |
| `--days 14` | Widen the harvest window from the default 7 |
| `2608.17981` | Skip harvesting; build that arXiv ID directly |
| a topic, e.g. `world models` | Bias harvesting and ranking toward that topic |
| `--pr` | Ship on a branch and open a pull request instead of pushing to `main` |
| `--dry-run` | Build and verify, then stop. Nothing is committed. |

## 1 · Harvest

`bun scripts/radar.ts --days 7` does this end to end and prints a ranked report.
Prefer it — it uses the same rubric and the same sources, and it is what CI runs.

Working by hand instead, pull from all three, because one source alone misses the
paper worth writing about:

1. **Trending** — `https://huggingface.co/api/daily_papers?limit=100` (the cap is
   100; larger values 400). Carries the search-demand signal.
2. **Firehose** — the arXiv API over `cs.LG`, `cs.AI`, `cs.CL`. Send a
   `User-Agent`; arXiv answers anonymous callers with 429.
3. **Targeted** — a `WebSearch` over the site's recurring themes, restricted to
   the last week, plus a search for each finalist's name to see what already
   ranks for it.

Aim for 25–40 candidates before filtering.

## 2 · Dedupe

`bun scripts/catalogue.ts --check <id> <id> …` exits non-zero if any candidate is
already covered. Then apply the rubric's judgement about core mechanisms.

## 3 · Rank

Apply the rubric above. Present the shortlist as a table — ID, title, the five
scores with search demand doubled, total, and a one-sentence reason. Then state
your pick, why it beat the runner-up, and what search query this page is trying
to win.

## 4 · Verify the source

Before writing a single word, fetch the real arXiv record and read the actual
abstract — `bun -e 'import {fetchPaper} from "./scripts/lib/sources"; console.log(await fetchPaper("<id>"))'`
gives title, authors, date and abstract straight from the API.

Never write from a title, a trending blurb, or someone else's summary. If the
abstract does not support a claim, drop the claim.

## 5 · Write the entry

Follow `AGENTS.md` for content structure and simulation standards, and the
rubric above for how to write it to be found. In particular:

- `src/content/ideas/<slug>.md`, frontmatter validating against
  `src/content/config.ts`.
- `src/components/react/simulations/<Name>Simulation.tsx` — registered by
  filename alone; `DemoView` globs the directory.
- **Style through the tokens.** Semantic Tailwind colours from
  `tailwind.config.mjs` and the component classes in `src/styles/global.css`.
  No new hex values, no light theme.
- **Seed every random simulation** with the shared `rng(seed)` mulberry32 helper.
- **Do not overstate the mechanism.** If the paper's real effect is modest, the
  toy shows a modest effect and says so. Verify the toy actually reproduces the
  paper's qualitative claim before shipping it — if the honest version of the
  dynamics does not, that is a finding about the demo, not a licence to invent
  one. Solve for the constants rather than guessing at them.

## 6 · Verify the build

Non-negotiable, in order:

1. `bun run build` — must pass clean.
2. `bun scripts/verify-page.ts /idea/<slug>/` — starts the preview server and
   loads the page in a real browser, failing on console errors, an unmounted
   simulation, unrendered mermaid, or horizontal scroll at 390px. A simulation
   that throws on mount still builds fine, so a green build is not evidence the
   page works.

Fix and re-verify. Do not ship a red build.

## 7 · Ship

Stage **only** the files you created or edited. Never `git add -A` at the repo
root: the tree carries untracked items that must not be committed — `.leankg/` is
96 MB and `paperlens.io/` is a nested git repository.

Commit as `feat(paper): <title>` with the standard trailers. Then:

- **Default** — push to `main`; the Pages workflow deploys. Confirm the live URL
  returns 200.
- **`--pr`** — branch as `paper/<slug>`, push, and `gh pr create` with the
  shortlist in the body.

## Report back

- The shortlist table, including what was rejected as already covered and which
  entry covered it.
- The pick, the paper's real title and authors, and the mechanism in one sentence.
- The search query this entry is built to win, and what currently ranks for it.
- The live URL, plus what the simulation lets a reader do.
