---
title: "LeWorldModel — Stable End-to-End JEPA from Pixels: article + simulation"
date: 2026-05-16
status: APPROVED
paper: 2603.19312v1
authors: Maes, Le Lidec, Scieur, LeCun, Balestriero
---

# Goal

Add a paperlens.io post for the LeWorldModel paper, with an interactive in-browser simulation that visually demonstrates the paper's central contribution: a JEPA world model that trains stably end-to-end from pixels using only **two** losses — a next-embedding prediction loss and a Gaussian regularizer — without EMA target networks, pre-trained encoders, or auxiliary supervision.

The demo angle is **collapse-vs-stability**: naive end-to-end JEPA collapses its latent space to a constant; LeWM's Gaussian regularizer keeps it spread and informative.

# Non-goals

- Reproducing the paper's full DMControl benchmark numbers.
- Training on real pixel data — we use a synthetic 2D bouncing-particle environment.
- Implementing real backprop through an arbitrary autograd library — we hand-derive gradients for a tiny MLP (sufficient for the visual point).

# Architecture

## Files

- `src/content/ideas/leworldmodel-stable-jepa.mdx` — the article. MDX format because we want Mermaid + math.
- `src/components/react/simulations/LeWorldModelSimulation.tsx` — the simulation, registered in `DemoView.tsx` under the key `LeWorldModel` (matches the `simulation:` frontmatter field).
- `docs/superpowers/specs/2026-05-16-leworldmodel-design.md` — this file.

No changes outside these three.

## Article structure

Follows the cadence of `turboquant-polarquant.mdx`:

1. **Frontmatter**
   - `title: "The Collapse-Proof World Model"`
   - `subtitle: "How LeWorldModel trains a JEPA from pixels with two losses instead of six"`
   - `date: 2026-05-16`
   - `status: RESEARCH`
   - `category: paper`
   - `impact: "6→1 loss reduction · 48× faster planning · 15M params"`
   - `readTime: "14m"`
   - `tags: [JEPA, World Models, Self-Supervised, LeCun, Balestriero, Representation Collapse]`
   - `coverImage: https://picsum.photos/seed/leworldmodel/800/600?grayscale`
   - `simulation: LeWorldModel`
   - `pdfUrl: https://arxiv.org/pdf/2603.19312`
   - `featured: true`

2. **Executive Summary** — anchors on the two-loss claim and the 48× planning speedup, references LeCun's bigger JEPA program.

3. **The Collapse Problem** — when both encoder and predictor are trainable, the constant-encoder solution drives prediction loss to zero. End-to-end JEPA training therefore collapses unless something explicitly prevents it.

4. **What Previous Fixes Cost** — EMA target encoders (DINO, I-JEPA), VICReg's three-term variance/invariance/covariance loss, pre-trained encoders. Each adds hyperparameters and engineering surface.

5. **LeWorldModel's Recipe** — exactly two losses:
   $$L = \underbrace{\|p(z_t, a_t) - \text{sg}(z_{t+1})\|^2}_{L_{\text{pred}}} + \lambda \underbrace{(\|\mu\|^2 + \|\Sigma - I\|_F^2)}_{L_{\text{gauss}}}$$
   where $\mu$ and $\Sigma$ are the empirical mean and covariance of latents in the batch. Notably, the paper keeps a stop-gradient on the target but does **not** require an EMA — both encoders share weights.

6. **Mermaid pipeline** — pixels → encoder → latent → predictor → predicted latent; Gaussian regularizer branch hangs off the latent.

7. **Reference implementation** — ≈25 line PyTorch sketch showing the forward pass and the moment-matching regularizer.

8. **Results & Why It Matters** — wire to MPC story; cross-link `[[jepa-world-models]]` and `[[dreamzero-world-action-models]]`.

## Simulation structure

Four stages, modeled on `TurboQuantSimulation`. Top-level component is a `SchematicCard` with stage-selector buttons. Shared deterministic RNG (mulberry32) seeds all randomness so visuals are reproducible.

### Stage 1 — The pixel environment
- A 2D particle bouncing inside a unit square; rendered into a 16×16 occupancy grid.
- Controls: particle count slider (1–4), bounce speed slider.
- Visual: live frame on the left, recent frames filmstrip on the right.
- Purpose: establish what the encoder consumes.

### Stage 2 — Naive end-to-end JEPA collapses
- Tiny MLP: encoder (256 → 8 → 2) and predictor (2 → 8 → 2). Hand-derived gradients (squared-loss, ReLU, linear) — no autograd lib.
- Loop: one training step per animation tick.
- Two panels: input pixel frame | latent 2D scatter (last N batched latents).
- Animation: latent cloud visibly shrinks toward a point over ~200 steps.
- Readouts: $L_{\text{pred}}$, latent variance, status badge that flips to `COLLAPSED` when $\text{tr}(\Sigma) < 0.05$.

### Stage 3 — Add Gaussian regularizer = LeWM
- Same network, regularizer enabled.
- Controls: $\lambda$ slider (0 → 1.0).
- Latent scatter overlaid with the unit-variance N(0, I) reference ellipse.
- Two loss curves over time: $L_{\text{pred}}$ and $L_{\text{gauss}}$.
- Toggle button: "Naive" / "LeWM" — re-runs training from the same seed for side-by-side fairness.

### Stage 4 — Latent rollout / planning
- Take the LeWM-trained encoder/predictor.
- Fit a linear probe on (latent → 2D position) on a held-out batch.
- Roll out: encode one frame to $z_0$, iterate predictor for $k$ steps; decode each latent through the probe to predicted position.
- Controls: rollout horizon slider (1–60 steps).
- Plot: predicted trajectory vs. ground-truth physics trajectory in the unit square.

## Math: hand-derived gradients

For a 2-layer MLP $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$ trained with $L = \|y - z^*\|^2$ where $z^* = \text{sg}(z_{t+1})$:

- $\partial L / \partial W_2 = 2(y - z^*) h^T$ where $h = \text{ReLU}(W_1 x + b_1)$
- $\partial L / \partial W_1 = (W_2^T \cdot 2(y - z^*)) \odot \mathbb{1}[W_1 x + b_1 > 0] \cdot x^T$

For the Gaussian regularizer $L_{\text{gauss}} = \|\mu\|^2 + \|\Sigma - I\|_F^2$ with $\mu = \frac{1}{N}\sum z_i$, $\Sigma = \frac{1}{N}\sum (z_i - \mu)(z_i - \mu)^T$:

- $\partial L_{\text{gauss}} / \partial z_i = \frac{2}{N}\mu + \frac{4}{N}(\Sigma - I)(z_i - \mu)$

This is enough to add an extra gradient to each latent before backproping through the encoder.

## Data flow

```
particle physics state (x, y, vx, vy) ──► render to 16×16 grid ──► flatten to ℝ^256
                                                                       │
                                                                       ▼
                                                          encoder MLP (256→8→2)
                                                                       │
                                                                       ▼
                                                                 latent z ∈ ℝ²
                                                                ┌──────┴──────┐
                                                                ▼             ▼
                                                       predictor MLP   Gaussian regularizer
                                                       (2→8→2)         (μ, Σ moments)
                                                                │
                                                                ▼
                                                          ẑ_{t+1}  vs  sg(z_{t+1})  ──► L_pred
```

# Testing & verification

- TypeScript build (`bun run build` or whatever the project uses) must pass.
- Manual: load the post in dev (`bun --hot` per project CLAUDE.md) and click through all four stages; confirm Stage 2's latent scatter visibly collapses and Stage 3's stays spread.
- Manual: confirm the post appears on the index page (via the existing content-collection wiring).

# Risks

- **In-browser training cost.** Mitigated by tiny MLP (≤ 256·8 + 8·2 = 2064 params) and small batches (~32). Should be well under 5ms/step.
- **Visual subtlety.** The collapse needs to be obvious; we tune learning rate and step count so it happens in ~3 seconds at default speed.
- **Probe overfit.** Stage 4 fits a linear probe on a fresh batch each rollout to avoid the impression that the probe is memorizing trajectories.

# Out of scope

- Real pixel rendering of e.g. cartpole — too heavy for an MDX demo.
- Action conditioning on the predictor — the paper supports it, but the visual story is the same without actions and simpler to render.
- Comparison against EMA-based JEPA — would require a third training branch; the article describes it in prose instead.
