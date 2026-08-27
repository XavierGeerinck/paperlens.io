# Paper selection rubric

The single source of truth for how PaperLens picks papers. Both the
`/scan-papers` slash command and the CI scripts in `scripts/` read this file, so
edit it here and both paths change together.

## Dedupe

An exact arXiv ID match against the catalogue is a hard drop. Beyond that, a
paper is a duplicate when its **core mechanism** is already the subject of an
entry, even under a different name and ID.

Proximity is not duplication. An entry on JEPA loss design does not cover a
paper on looped world-model depth; they answer different questions, and
cross-linking them makes both stronger. Always name the entry checked against
and say why the candidate survived or didn't.

## Scoring

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
  search the paper's name and see what already exists. A crowded first page of
  results is a reason to pick the runner-up. Score 3 for a named thing that is
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

Total = 2 × search demand + the other four. Maximum 18.

## Exclusions

Hard exclusions: dataset releases, surveys, and pure leaderboard papers where the
contribution is a number rather than an idea.

One deliberate exception. A **named model or system report with very high search
demand** — the kind of release people will be searching for by name all month —
qualifies *if* one real mechanism inside it can be isolated and simulated, and
the entry is built around that mechanism. Write about the idea, using the name
people are searching for. If there is no mechanism to isolate, skip it: a page
that just restates a release note will not rank and does not belong here.

## Writing for search

The rubric picks a paper for search demand; the entry has to earn it.

- `title` leads with the name people will search — the artefact name, then the
  plain-language hook. `"Recirculation: Making a Transformer Think Without
  Tokens"`, not `"On Inference-Time Belief Tracking"`.
- `subtitle` is the meta description. One sentence, under 160 characters, that
  answers the query rather than teasing it.
- `tags` are query terms, not vibes — the model name, the lab, the mechanism, the
  problem it solves.
- Use the paper's own vocabulary in headings. Someone searching a term from the
  abstract should land on a heading that uses it.
- Answer the obvious follow-up questions as headings: what it is, why the old way
  failed, what it costs, whether it works at scale.
- Cross-link both directions — add a link from the adjacent existing entries to
  this one, not just from this one outward. Internal links are plain paths,
  `/idea/<slug>/`. There is no wikilink plugin in this project; `[[slug]]`
  renders as a dead link.

## Honesty

- Never write from a title, a trending blurb, or someone else's summary. Fetch
  the real abstract first and quote only what it supports.
- A simulation must not overstate the mechanism. If the real effect is modest,
  show a modest effect and say so. Inventing a dramatic result, or a penalty the
  paper never reports, misrepresents the work.
- Label a toy as a toy, and put the paper's measured numbers in the entry.
