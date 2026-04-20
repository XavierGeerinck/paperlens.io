# Design: Blog Post + Simulation — TurboQuant & PolarQuant

**Date:** 2026-04-20
**Topic:** A combined paperlens.io blog post covering TurboQuant (arXiv:2502.02617) and PolarQuant (arXiv:2504.19874), with a paired React simulation component.

## Goal

Publish one combined post on paperlens.io that tells the unified story of TurboQuant → PolarQuant. Both papers come from the same Google Research group (Zandieh, Mirrokni et al.) and share a single core insight: *randomly rotate vectors before quantizing them*. TurboQuant is the general-purpose method; PolarQuant is a KV-cache-specific refinement. One post frames them as a coherent narrative; two would repeat the fundamentals.

## Deliverables

1. **Markdown post** at `src/content/ideas/turboquant-polarquant.mdx`
2. **React simulation component** at `src/components/react/simulations/TurboQuantSimulation.tsx`
3. **Simulation registration** wired into the `simulation:` resolver (wherever `SubQuadratic`, `MLA` etc. are registered — to be located during implementation)

All files live in the main site repo at `/Users/xaviergeerinck/Projects/paperlens.io/`, not the (disconnected) worktree the task was dispatched in.

## Post

### Frontmatter

```yaml
title: "The Rotation Revolution: Near-Optimal KV Cache Quantization"
subtitle: "How TurboQuant and PolarQuant use random rotations to squeeze LLM memory to within a constant of the information-theoretic limit"
date: 2026-04-20
status: RESEARCH
category: paper
impact: 4.2× KV Cache Compression, Quality-Neutral at 3.5 bpc
readTime: "15m"
tags: [Quantization, KV Cache, LLM Inference, TurboQuant, PolarQuant, Google Research]
coverImage: https://picsum.photos/seed/turboquant/800/600?grayscale
simulation: TurboQuant
pdfUrl: https://arxiv.org/pdf/2502.02617
featured: true
```

The primary `pdfUrl` points to TurboQuant; PolarQuant's arXiv link appears inline in the body.

### Narrative Arc

Problem → rotation trick (TurboQuant) → polar refinement (PolarQuant) → why it's near-optimal → hardware implications.

### Sections

1. **Executive Summary** (~150 words) — KV caches are the memory bottleneck of long-context LLMs. Two 2025 Google Research papers solve it with the same geometric insight: randomly rotate vectors first, then quantization becomes easy. TurboQuant reaches ~2.7× of the information-theoretic distortion bound at 3.5 bpc with no quality loss. PolarQuant specializes to KV caches via polar coordinates, achieving 4.2× compression.

2. **The Quadratic Tax on Memory** — KV cache growth with context length, the outlier problem that breaks naive quantization, and the hidden cost of per-block scale/zero-point metadata.

3. **TurboQuant — The Rotation Trick** — Random rotation as an isotropy preconditioner. Coordinates become near-independent and Beta-distributed. Per-coordinate scalar quantizer is then near-optimal. Two-stage scheme: MSE quantizer + 1-bit Quantized-JL residual for unbiased inner products. Distortion bound vs Shannon lower bound.

4. **PolarQuant — From Cartesian to Polar** — Recursive polar transform of preconditioned KV embeddings. Key theorem: after random preconditioning, angles have an analytically computable, tightly concentrated distribution → no scale/zero-point storage required. 4.2× compression, SOTA long-context quality.

5. **Mermaid pipeline diagram** — raw vector → random rotation → (branch A: scalar quantize per coordinate | branch B: polar transform + angle quantize) → reconstructed vector.

6. **Code snippet** — minimal PyTorch-style reference implementation of random-rotation + per-coordinate quantizer, with a commented sketch of the polar extension.

7. **Feasibility & Hardware Targets** — kernel fusion of rotation + quantization on NVIDIA Hopper/Blackwell, bandwidth savings vs compute cost, compatibility with flash-attention KV layouts.

8. **The Bigger Picture** — geometry-first quantization. Connection to Johnson-Lindenstrauss, random projections, and the broader "make-the-distribution-nice-before-you-quantize" trend (cf. QuaRot, SpinQuant).

## Simulation Component

`TurboQuantSimulation.tsx` — a single interactive React component with four staged visualizations. Style matches existing simulations (`SubQuadraticSimulation`, `MLASimulation`): pure React + SVG, no three.js.

### Stages

**Stage 1 — The Problem (Raw Distribution)**
- Simulated KV embedding vector (d=128). Coordinates drawn from a heavy-tailed mixture (Gaussian + occasional outliers) to mimic real LLM activations.
- Histogram of coordinate values showing visible outliers.
- Annotation: *"Outliers force wide quantization bins → wasted bits."*

**Stage 2 — TurboQuant's Rotation Trick**
- Apply a random orthogonal rotation (composition of Householder reflections or a Hadamard-based structured rotation for visualization-friendly O(d log d) cost).
- Re-plot the histogram: now approximately Gaussian/Beta-shaped; outliers gone.
- Side panel: pairwise coordinate correlation heatmap collapsing toward the diagonal (near-independence).
- Annotation showing the distortion scaling $D(b) \approx c \cdot 2^{-2b}$.

**Stage 3 — Scalar Quantization + Distortion Curve**
- Bit-width slider (1 → 8 bpc). Live updates:
  - Reconstructed vector overlay on original (post-rotation).
  - MSE distortion plotted vs bit rate.
  - Shannon lower bound plotted as a reference line — visual gap of ~2.7× constant.
- Toggle: "raw quantization" vs "TurboQuant" → visualizes the order-of-magnitude improvement.

**Stage 4 — PolarQuant's Angle Concentration**
- Take 2D slices of the preconditioned vector, plot as (r, θ) scatter.
- Histogram of θ values converging to the analytically derived distribution (overlaid reference curve).
- Annotation: *"The angle distribution has a closed form → no per-block scale/zero-point metadata → extra bits saved."*
- Final readout: "4.2× compression, quality-neutral."

### Controls

- Bit budget slider (1–8 bpc)
- Vector dimension selector (64 / 128 / 256)
- Outlier-intensity toggle
- "Re-sample rotation" button — demonstrates data-obliviousness: any random rotation works.

### Tech Constraints

- Pure React + SVG. No canvas, no three.js.
- Deterministic RNG seeded by component state for reproducibility across renders.
- Target ~400–600 LOC, in line with existing simulations in the project.
- No external math libs beyond what is already used in the project (to be verified during implementation plan).

## Out of Scope

- Benchmarking code or reproduction of paper results.
- Comparisons with QuaRot/SpinQuant beyond a passing mention in the "Bigger Picture" section.
- Changes to the existing simulation registry architecture — only adding a new entry.
- Posts or simulations for other quantization papers.

## Open Questions for Implementation

- Exact location and format of the `simulation:` → component registry (to be discovered during implementation, likely in `src/components/` or `src/content/config.ts`).
- Whether any shared math utilities (e.g., random orthogonal matrix generator) already exist in the repo and should be reused.
