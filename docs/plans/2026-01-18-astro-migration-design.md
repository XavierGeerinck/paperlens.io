# Astro Migration Design

## Overview

Migrate the PaperLens research lab site from React SPA (Vite) to Astro for improved performance, simpler SEO, and better developer experience for content-heavy sites.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Simulation hydration | `client:visible` | Lazy load when scrolled into view, fastest initial load |
| Page transitions | Astro View Transitions | SPA-like smoothness with MPA benefits |
| Content management | Astro Content Collections | Type-safe, built-in validation, eliminates custom build scripts |
| Styling | Tailwind via @astrojs/tailwind | Proper bundling with CSS purging |

## Project Structure

```
valar.github.io/
├── astro.config.mjs
├── package.json
├── tsconfig.json
├── tailwind.config.mjs
│
├── src/
│   ├── content/
│   │   ├── config.ts              # Zod schema for ideas
│   │   └── ideas/                 # Markdown files
│   │
│   ├── pages/
│   │   ├── index.astro            # Homepage
│   │   ├── idea/[slug].astro      # Dynamic idea pages
│   │   ├── 404.astro              # Error page
│   │   └── rss.xml.ts             # RSS feed
│   │
│   ├── layouts/
│   │   └── BaseLayout.astro       # Shared layout
│   │
│   ├── components/
│   │   ├── Header.astro
│   │   ├── Footer.astro
│   │   ├── IdeaCard.astro
│   │   ├── SEO.astro
│   │   └── react/                 # Interactive components
│   │       ├── simulations/       # All 14+ simulations
│   │       ├── DemoView.tsx
│   │       ├── GameOfLife.tsx
│   │       └── GlobeAnimation.tsx
│   │
│   └── styles/
│       └── global.css
│
├── public/
└── dist/
```

## Content Collections Schema

```typescript
// src/content/config.ts
import { defineCollection, z } from 'astro:content';

const ideasCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    subtitle: z.string(),
    date: z.coerce.date(),
    status: z.enum(['RESEARCH', 'CONCEPT', 'PROTOTYPE', 'ALPHA', 'ARCHIVED']),
    category: z.enum(['idea', 'paper']).default('idea'),
    impact: z.string(),
    readTime: z.string(),
    tags: z.array(z.string()),
    coverImage: z.string().url(),
    simulation: z.string().optional(),
    pdfUrl: z.string().url().optional(),
    githubUrl: z.string().url().optional(),
    featured: z.boolean().default(false),
  }),
});

export const collections = { ideas: ideasCollection };
```

## SEO Component

```astro
<!-- src/components/SEO.astro -->
---
interface Props {
  title: string;
  description: string;
  image?: string;
  type?: 'website' | 'article';
  date?: Date;
  canonical?: string;
}

const {
  title,
  description,
  image = 'https://paperlens.io/og-default.png',
  type = 'website',
  date,
  canonical = Astro.url.href,
} = Astro.props;

const siteName = "PaperLens";
---

<title>{title} | {siteName}</title>
<meta name="description" content={description} />
<meta name="robots" content="index, follow" />
<link rel="canonical" href={canonical} />

<meta property="og:type" content={type} />
<meta property="og:title" content={title} />
<meta property="og:description" content={description} />
<meta property="og:image" content={image} />
<meta property="og:url" content={canonical} />
<meta property="og:site_name" content={siteName} />

<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content={title} />
<meta name="twitter:description" content={description} />
<meta name="twitter:image" content={image} />

{type === 'article' && date && (
  <meta property="article:published_time" content={date.toISOString()} />
)}

<script type="application/ld+json" set:html={JSON.stringify(
  type === 'article' ? {
    "@context": "https://schema.org",
    "@type": "Article",
    "headline": title,
    "description": description,
    "image": image,
    "datePublished": date?.toISOString(),
    "author": { "@type": "Person", "name": "Xavier Geerinck" }
  } : {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": siteName,
    "url": "https://paperlens.io"
  }
)} />
```

## Base Layout

```astro
<!-- src/layouts/BaseLayout.astro -->
---
import { ViewTransitions } from 'astro:transitions';
import Header from '../components/Header.astro';
import Footer from '../components/Footer.astro';
import SEO from '../components/SEO.astro';
import '../styles/global.css';

interface Props {
  title: string;
  description: string;
  image?: string;
  type?: 'website' | 'article';
  date?: Date;
}

const { title, description, image, type, date } = Astro.props;
---

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <SEO {title} {description} {image} {type} {date} />
  <ViewTransitions />
  <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" />
</head>
<body class="bg-zinc-950 text-zinc-200 min-h-screen">
  <Header />
  <main transition:animate="fade">
    <slot />
  </main>
  <Footer />
</body>
</html>
```

## Astro Configuration

```javascript
// astro.config.mjs
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://paperlens.io',
  integrations: [
    react(),
    tailwind(),
    sitemap(),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
```

## React Component Integration

Simulations stay as React, loaded as islands:

```astro
<!-- Usage in Astro -->
<DemoView client:visible simulation="DeepSeekEngramSimulation" />
<GameOfLife client:load />
```

Components that become Astro (no JS shipped):
- Header, Footer, IdeaCard
- SEO, MarkdownRenderer

Components that stay React:
- All 14+ simulations
- DemoView, GameOfLife, GlobeAnimation
- useSimulation hook

## Migration Steps

1. Initialize Astro - Create config, restructure to `src/`
2. Set up Content Collections - Move markdown, create Zod schema
3. Create BaseLayout - SEO, View Transitions, global styles
4. Convert static components - Header, Footer, IdeaCard to `.astro`
5. Build pages - index.astro, [slug].astro, 404.astro
6. Move React components - Simulations to `components/react/`
7. Add RSS/Sitemap - Via integrations and endpoint
8. Remove old files - React Router, build scripts, generated-ideas.ts
9. Test and verify - SEO, simulations, navigation

## Files to Delete After Migration

- `App.tsx`, `index.tsx`
- `scripts/build.ts`, `scripts/generate-ideas.ts`
- `generated-ideas.ts`
- `vite.config.ts`
- `components/SEO.tsx` (replaced by Astro version)
- `hooks/useIdeas.ts` (replaced by Content Collections)

## Benefits

- **Faster builds** - Astro's partial hydration
- **Better SEO** - Static HTML by default, no hydration needed for meta
- **Simpler architecture** - No custom SSG scripts
- **Type-safe content** - Zod validation catches errors at build time
- **Smaller JS bundle** - Only simulations ship JavaScript
