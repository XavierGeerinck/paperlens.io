# Astro Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the PaperLens React SPA to Astro with Content Collections, View Transitions, and React islands for simulations.

**Architecture:** Static-first site with Astro handling content/routing/SEO. React components hydrate as islands for interactive simulations. Tailwind bundled via official integration. View Transitions provide SPA-like navigation.

**Tech Stack:** Astro 5.x, @astrojs/react, @astrojs/tailwind, @astrojs/sitemap, Bun, React 19, Tailwind CSS 4

---

## Task 1: Initialize Astro Project

**Files:**
- Create: `package.json` (overwrite)
- Create: `astro.config.mjs`
- Create: `tsconfig.json` (overwrite)
- Create: `src/env.d.ts`
- Delete: `vite.config.ts`

**Step 1: Install Astro and integrations**

Run:
```bash
cd /Users/xaviergeerinck/Projects/valar.github.io/.worktrees/astro-migration
rm -rf node_modules package.json bun.lockb
bun init -y
bun add astro @astrojs/react @astrojs/tailwind @astrojs/sitemap
bun add react react-dom
bun add -d @types/react @types/react-dom typescript
```

**Step 2: Create Astro config**

Create `astro.config.mjs`:
```javascript
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://paperlens.io',
  integrations: [
    react(),
    tailwind(),
    sitemap(),
  ],
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
```

**Step 3: Create TypeScript config**

Create `tsconfig.json`:
```json
{
  "extends": "astro/tsconfigs/strict",
  "compilerOptions": {
    "jsx": "react-jsx",
    "jsxImportSource": "react",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"]
    }
  }
}
```

**Step 4: Create env.d.ts**

Create `src/env.d.ts`:
```typescript
/// <reference path="../.astro/types.d.ts" />
/// <reference types="astro/client" />
```

**Step 5: Update package.json scripts**

Edit `package.json` scripts section:
```json
{
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  }
}
```

**Step 6: Delete old Vite config**

Run:
```bash
rm -f vite.config.ts
```

**Step 7: Commit**

```bash
git add -A
git commit -m "chore: initialize Astro with React and Tailwind integrations"
```

---

## Task 2: Set Up Tailwind CSS

**Files:**
- Create: `tailwind.config.mjs`
- Create: `src/styles/global.css`

**Step 1: Create Tailwind config**

Create `tailwind.config.mjs`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        heading: ['Space Grotesk', 'sans-serif'],
        sketch: ['Caveat', 'cursive'],
      },
      gridTemplateColumns: {
        '16': 'repeat(16, minmax(0, 1fr))',
        '32': 'repeat(32, minmax(0, 1fr))',
      },
    },
  },
  plugins: [],
};
```

**Step 2: Create global CSS**

Create `src/styles/global.css`:
```css
@import "tailwindcss";

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #18181b;
}

::-webkit-scrollbar-thumb {
  background: #3f3f46;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #52525b;
}

/* Grid background pattern */
.bg-grid {
  background-image: radial-gradient(circle, #27272a 1px, transparent 1px);
  background-size: 24px 24px;
}
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: configure Tailwind CSS with custom theme"
```

---

## Task 3: Create Content Collection Schema

**Files:**
- Create: `src/content/config.ts`
- Move: `content/*.md` → `src/content/ideas/*.md`

**Step 1: Create content config**

Create `src/content/config.ts`:
```typescript
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
    coverImage: z.string(),
    simulation: z.string().optional(),
    pdfUrl: z.string().optional(),
    githubUrl: z.string().optional(),
    featured: z.boolean().default(false),
  }),
});

export const collections = { ideas: ideasCollection };
```

**Step 2: Move markdown files**

Run:
```bash
mkdir -p src/content/ideas
mv content/*.md src/content/ideas/
rm -rf content/.obsidian
```

**Step 3: Verify collection works**

Run:
```bash
bun run astro sync
```

Expected: No errors, `.astro/` directory created with types.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: set up Astro content collection for ideas"
```

---

## Task 4: Create SEO Component

**Files:**
- Create: `src/components/SEO.astro`

**Step 1: Create SEO component**

Create `src/components/SEO.astro`:
```astro
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

const siteName = 'PaperLens';
const fullTitle = `${title} | ${siteName}`;
---

<!-- Primary Meta Tags -->
<title>{fullTitle}</title>
<meta name="title" content={fullTitle} />
<meta name="description" content={description} />
<meta name="robots" content="index, follow" />
<link rel="canonical" href={canonical} />

<!-- Open Graph / Facebook -->
<meta property="og:type" content={type} />
<meta property="og:url" content={canonical} />
<meta property="og:title" content={title} />
<meta property="og:description" content={description} />
<meta property="og:image" content={image} />
<meta property="og:site_name" content={siteName} />

<!-- Twitter -->
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:url" content={canonical} />
<meta name="twitter:title" content={title} />
<meta name="twitter:description" content={description} />
<meta name="twitter:image" content={image} />

{type === 'article' && date && (
  <meta property="article:published_time" content={date.toISOString()} />
)}

<!-- JSON-LD Structured Data -->
<script
  type="application/ld+json"
  set:html={JSON.stringify(
    type === 'article'
      ? {
          '@context': 'https://schema.org',
          '@type': 'Article',
          headline: title,
          description: description,
          image: image,
          datePublished: date?.toISOString(),
          author: {
            '@type': 'Person',
            name: 'Xavier Geerinck',
          },
          publisher: {
            '@type': 'Organization',
            name: siteName,
            url: 'https://paperlens.io',
          },
        }
      : {
          '@context': 'https://schema.org',
          '@type': 'WebSite',
          name: siteName,
          url: 'https://paperlens.io',
        }
  )}
/>
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: create SEO component with meta tags and JSON-LD"
```

---

## Task 5: Create Base Layout

**Files:**
- Create: `src/layouts/BaseLayout.astro`

**Step 1: Create base layout**

Create `src/layouts/BaseLayout.astro`:
```astro
---
import { ViewTransitions } from 'astro:transitions';
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

<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <SEO {title} {description} {image} {type} {date} />
    <ViewTransitions />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;700&display=swap"
      rel="stylesheet"
    />

    <!-- KaTeX for math -->
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
    />
  </head>
  <body class="bg-zinc-950 text-zinc-200 min-h-screen antialiased">
    <main transition:animate="fade">
      <slot />
    </main>
  </body>
</html>
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: create base layout with View Transitions"
```

---

## Task 6: Create Header Component

**Files:**
- Create: `src/components/Header.astro`

**Step 1: Create header**

Create `src/components/Header.astro`:
```astro
---
const isHome = Astro.url.pathname === '/';
---

<header
  class="fixed top-0 left-0 right-0 z-50 bg-zinc-950/80 backdrop-blur-sm border-b border-zinc-800"
>
  <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
    <a
      href="/"
      class="font-heading text-lg font-bold text-zinc-100 hover:text-indigo-400 transition-colors"
    >
      PAPERLENS
    </a>

    {isHome && (
      <div class="hidden md:flex items-center gap-4 text-xs font-mono">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span class="text-zinc-500">SYSTEM STATUS:</span>
          <span class="text-emerald-400">NORMAL</span>
        </div>
      </div>
    )}

    <a
      href="mailto:geerinck.xavier@gmail.com"
      class="px-4 py-1.5 border border-zinc-700 text-xs font-mono uppercase tracking-wider text-zinc-300 hover:bg-zinc-800 hover:border-zinc-600 transition-all"
    >
      Connect
    </a>
  </div>
</header>
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: create Header component"
```

---

## Task 7: Create Footer Component

**Files:**
- Create: `src/components/Footer.astro`

**Step 1: Create footer**

Create `src/components/Footer.astro`:
```astro
---
const currentYear = new Date().getFullYear();

const socialLinks = [
  { name: 'GitHub', href: 'https://github.com/xaviergeerinck' },
  { name: 'Twitter', href: 'https://twitter.com/XavierGeerinck' },
  { name: 'LinkedIn', href: 'https://linkedin.com/in/xaviergeerinck' },
  { name: 'Website', href: 'https://xaviergeerinck.com' },
];
---

<footer class="border-t border-zinc-800 bg-zinc-950 py-8 mt-16">
  <div class="max-w-7xl mx-auto px-4">
    <div class="flex flex-col md:flex-row items-center justify-between gap-4">
      <div class="text-xs text-zinc-500 font-mono">
        &copy; {currentYear} Xavier Geerinck. All rights reserved.
      </div>

      <div class="flex items-center gap-6">
        {socialLinks.map((link) => (
          <a
            href={link.href}
            target="_blank"
            rel="noopener noreferrer"
            class="text-xs text-zinc-500 hover:text-zinc-300 transition-colors font-mono uppercase tracking-wider"
          >
            {link.name}
          </a>
        ))}
      </div>
    </div>

    <div class="mt-4 text-center text-[10px] text-zinc-700 font-mono">
      If you can read this, you might be the kind of person we're looking for.
    </div>
  </div>
</footer>
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: create Footer component"
```

---

## Task 8: Create IdeaCard Component

**Files:**
- Create: `src/components/IdeaCard.astro`

**Step 1: Create idea card**

Create `src/components/IdeaCard.astro`:
```astro
---
import type { CollectionEntry } from 'astro:content';

interface Props {
  idea: CollectionEntry<'ideas'>;
  variant?: 'standard' | 'featured';
}

const { idea, variant = 'standard' } = Astro.props;
const { title, subtitle, status, coverImage, tags, impact, readTime } = idea.data;

const statusColors: Record<string, string> = {
  RESEARCH: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  CONCEPT: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  PROTOTYPE: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  ALPHA: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  ARCHIVED: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30',
};
---

{variant === 'featured' ? (
  <a
    href={`/idea/${idea.slug}/`}
    class="group block bg-zinc-900/50 border border-zinc-800 rounded-lg overflow-hidden hover:border-zinc-700 transition-all"
  >
    <div class="grid md:grid-cols-2 gap-6">
      <div class="aspect-video overflow-hidden">
        <img
          src={coverImage}
          alt={title}
          class="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500"
        />
      </div>
      <div class="p-6 flex flex-col justify-center">
        <div class="flex items-center gap-2 mb-3">
          <span class={`px-2 py-0.5 text-[10px] font-mono uppercase border rounded ${statusColors[status]}`}>
            {status}
          </span>
        </div>
        <h3 class="text-xl font-heading font-bold text-zinc-100 mb-2 group-hover:text-indigo-400 transition-colors">
          {title}
        </h3>
        <p class="text-sm text-zinc-400 mb-4">{subtitle}</p>
        <div class="grid grid-cols-2 gap-4 text-xs font-mono">
          <div class="bg-black/30 p-2 rounded border border-zinc-800">
            <div class="text-zinc-500 uppercase mb-1">Impact</div>
            <div class="text-zinc-300">{impact}</div>
          </div>
          <div class="bg-black/30 p-2 rounded border border-zinc-800">
            <div class="text-zinc-500 uppercase mb-1">Read Time</div>
            <div class="text-zinc-300">{readTime}</div>
          </div>
        </div>
      </div>
    </div>
  </a>
) : (
  <a
    href={`/idea/${idea.slug}/`}
    class="group block bg-zinc-900/50 border border-zinc-800 rounded-lg overflow-hidden hover:border-zinc-700 transition-all"
  >
    <div class="aspect-video overflow-hidden">
      <img
        src={coverImage}
        alt={title}
        class="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500"
      />
    </div>
    <div class="p-4">
      <div class="flex items-center gap-2 mb-2">
        <span class={`px-2 py-0.5 text-[10px] font-mono uppercase border rounded ${statusColors[status]}`}>
          {status}
        </span>
      </div>
      <h3 class="text-lg font-heading font-bold text-zinc-100 mb-1 group-hover:text-indigo-400 transition-colors line-clamp-2">
        {title}
      </h3>
      <p class="text-sm text-zinc-400 line-clamp-2">{subtitle}</p>
      <div class="flex flex-wrap gap-1 mt-3">
        {tags.slice(0, 3).map((tag) => (
          <span class="px-2 py-0.5 text-[10px] font-mono text-zinc-500 bg-zinc-800/50 rounded">
            {tag}
          </span>
        ))}
      </div>
    </div>
  </a>
)}
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: create IdeaCard component with featured variant"
```

---

## Task 9: Move React Components

**Files:**
- Move: `components/simulations/*` → `src/components/react/simulations/*`
- Move: `components/GameOfLife.tsx` → `src/components/react/GameOfLife.tsx`
- Move: `components/GlobeAnimation.tsx` → `src/components/react/GlobeAnimation.tsx`
- Move: `components/DemoView.tsx` → `src/components/react/DemoView.tsx`
- Move: `components/SketchElements.tsx` → `src/components/react/SketchElements.tsx`
- Move: `hooks/useSimulation.ts` → `src/hooks/useSimulation.ts`

**Step 1: Create directories and move files**

Run:
```bash
mkdir -p src/components/react/simulations
mkdir -p src/hooks

# Move React components
mv components/simulations/* src/components/react/simulations/
mv components/GameOfLife.tsx src/components/react/
mv components/GlobeAnimation.tsx src/components/react/
mv components/DemoView.tsx src/components/react/
mv components/SketchElements.tsx src/components/react/
mv components/ScrambleText.tsx src/components/react/
mv components/LensAnimation.tsx src/components/react/

# Move hooks
mv hooks/useSimulation.ts src/hooks/
```

**Step 2: Update DemoView imports**

Edit `src/components/react/DemoView.tsx` - update simulation imports to use relative paths:

Replace all imports like:
```typescript
import('./simulations/DeepSeekEngramSimulation')
```

With:
```typescript
import('./simulations/DeepSeekEngramSimulation')
```

(Paths should remain the same since simulations moved with DemoView)

**Step 3: Update simulation imports**

Each simulation imports from `../../hooks/useSimulation` and `../SketchElements`. Update to:
```typescript
import { useSimulation } from '../../hooks/useSimulation';
import { SchematicCard, SchematicButton } from './SketchElements';
```

Run this to fix imports in all simulation files:
```bash
cd src/components/react/simulations
for f in *.tsx; do
  sed -i '' 's|../../hooks/useSimulation|../../../hooks/useSimulation|g' "$f"
  sed -i '' 's|../SketchElements|../SketchElements|g' "$f"
done
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: move React components to src/components/react"
```

---

## Task 10: Create Homepage

**Files:**
- Create: `src/pages/index.astro`

**Step 1: Create homepage**

Create `src/pages/index.astro`:
```astro
---
import { getCollection } from 'astro:content';
import BaseLayout from '../layouts/BaseLayout.astro';
import Header from '../components/Header.astro';
import Footer from '../components/Footer.astro';
import IdeaCard from '../components/IdeaCard.astro';
import { Lightbulb, FileText } from 'lucide-react';

const allIdeas = await getCollection('ideas');
const sortedIdeas = allIdeas.sort(
  (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
);

const featured = sortedIdeas.find((idea) => idea.data.featured);
const papers = sortedIdeas.filter((i) => i.data.category === 'paper');
const ideas = sortedIdeas.filter((i) => i.data.category === 'idea');
---

<BaseLayout
  title="Research Lab"
  description="Xavier Geerinck's research lab exploring AI, systems, and beyond."
>
  <Header />

  <div class="pt-20 min-h-screen">
    <!-- Hero Section -->
    <section class="max-w-7xl mx-auto px-4 py-16">
      <div class="flex flex-col md:flex-row items-center gap-8">
        <div class="flex-1">
          <div class="text-xs font-mono text-zinc-500 uppercase tracking-wider mb-4">
            Research Directive 2026
          </div>
          <h1 class="text-4xl md:text-5xl font-heading font-bold text-zinc-100 mb-4">
            Make the <span class="text-indigo-400">impossible</span> inevitable
          </h1>
          <p class="text-lg text-zinc-400 max-w-xl">
            Exploring the frontiers of AI, systems architecture, and emerging technologies.
          </p>
        </div>
        <div class="w-48 h-48 rounded-full overflow-hidden border-2 border-zinc-800">
          <img
            src="https://media.licdn.com/dms/image/v2/D4E03AQFaL40i2fJ6Jw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1700217102565?e=1740614400&v=beta&t=MxXaVr_eA_ZhxYuFpRHPv-HNd3XSMgXE8M8cCuLXhqQ"
            alt="Xavier Geerinck"
            class="w-full h-full object-cover"
          />
        </div>
      </div>
    </section>

    <!-- Featured -->
    {featured && (
      <section class="max-w-7xl mx-auto px-4 py-8">
        <div class="flex items-center gap-2 mb-6 text-xs font-mono text-zinc-500 uppercase">
          <span class="w-2 h-2 rounded-full bg-indigo-500" />
          Featured Research
        </div>
        <IdeaCard idea={featured} variant="featured" />
      </section>
    )}

    <!-- My Thoughts -->
    {ideas.length > 0 && (
      <section class="max-w-7xl mx-auto px-4 py-8">
        <div class="flex items-center gap-2 mb-6 text-xs font-mono text-zinc-500 uppercase">
          <Lightbulb className="w-4 h-4" />
          My Thoughts
        </div>
        <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {ideas.filter((i) => i.slug !== featured?.slug).map((idea) => (
            <IdeaCard idea={idea} />
          ))}
        </div>
      </section>
    )}

    <!-- Papers -->
    {papers.length > 0 && (
      <section class="max-w-7xl mx-auto px-4 py-8">
        <div class="flex items-center gap-2 mb-6 text-xs font-mono text-zinc-500 uppercase">
          <FileText className="w-4 h-4" />
          Papers
        </div>
        <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {papers.filter((i) => i.slug !== featured?.slug).map((idea) => (
            <IdeaCard idea={idea} />
          ))}
        </div>
      </section>
    )}
  </div>

  <Footer />
</BaseLayout>
```

**Step 2: Verify build works**

Run:
```bash
bun run build
```

Expected: Build completes without errors.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: create homepage with content collections"
```

---

## Task 11: Create Idea Detail Page

**Files:**
- Create: `src/pages/idea/[slug].astro`

**Step 1: Create dynamic idea page**

Create `src/pages/idea/[slug].astro`:
```astro
---
import { getCollection, type CollectionEntry } from 'astro:content';
import BaseLayout from '../../layouts/BaseLayout.astro';
import Header from '../../components/Header.astro';
import Footer from '../../components/Footer.astro';
import DemoView from '../../components/react/DemoView';

export async function getStaticPaths() {
  const ideas = await getCollection('ideas');
  return ideas.map((idea) => ({
    params: { slug: idea.slug },
    props: { idea },
  }));
}

interface Props {
  idea: CollectionEntry<'ideas'>;
}

const { idea } = Astro.props;
const { title, subtitle, status, date, tags, simulation, pdfUrl, coverImage } = idea.data;
const { Content } = await idea.render();

const statusColors: Record<string, string> = {
  RESEARCH: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  CONCEPT: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  PROTOTYPE: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  ALPHA: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  ARCHIVED: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30',
};

const formattedDate = date.toLocaleDateString('en-US', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});
---

<BaseLayout
  title={title}
  description={subtitle}
  image={coverImage}
  type="article"
  date={date}
>
  <Header />

  <div class="pt-20 min-h-screen">
    <article class="max-w-4xl mx-auto px-4 py-8">
      <!-- Breadcrumb -->
      <nav class="mb-8 text-sm font-mono">
        <a href="/" class="text-zinc-500 hover:text-zinc-300">Home</a>
        <span class="text-zinc-700 mx-2">/</span>
        <span class="text-zinc-400">{title}</span>
      </nav>

      <!-- Header -->
      <header class="mb-8">
        <div class="flex items-center gap-2 mb-4">
          <span class={`px-2 py-0.5 text-[10px] font-mono uppercase border rounded ${statusColors[status]}`}>
            {status}
          </span>
          <span class="text-xs text-zinc-500 font-mono">{formattedDate}</span>
        </div>
        <h1 class="text-3xl md:text-4xl font-heading font-bold text-zinc-100 mb-4">
          {title}
        </h1>
        <p class="text-lg text-zinc-400">{subtitle}</p>
        <div class="flex flex-wrap gap-2 mt-4">
          {tags.map((tag) => (
            <span class="px-2 py-1 text-xs font-mono text-zinc-500 bg-zinc-800/50 rounded">
              {tag}
            </span>
          ))}
        </div>
      </header>

      <!-- Cover Image -->
      <div class="aspect-video overflow-hidden rounded-lg border border-zinc-800 mb-8">
        <img
          src={coverImage}
          alt={title}
          class="w-full h-full object-cover"
        />
      </div>

      <!-- Content -->
      <div class="prose prose-invert prose-zinc max-w-none prose-headings:font-heading prose-code:font-mono prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800">
        <Content />
      </div>

      <!-- Simulation -->
      {simulation && (
        <section class="mt-12 pt-8 border-t border-zinc-800">
          <h2 class="text-xl font-heading font-bold text-zinc-100 mb-6">
            Interactive Simulation
          </h2>
          <DemoView client:visible simulation={simulation} />
        </section>
      )}

      <!-- PDF Link -->
      {pdfUrl && (
        <section class="mt-8 pt-8 border-t border-zinc-800">
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            class="inline-flex items-center gap-2 px-4 py-2 border border-zinc-700 text-zinc-300 hover:bg-zinc-800 transition-colors font-mono text-sm"
          >
            View Original Paper (PDF)
          </a>
        </section>
      )}
    </article>
  </div>

  <Footer />
</BaseLayout>
```

**Step 2: Verify build**

Run:
```bash
bun run build
```

Expected: Build generates `/idea/{slug}/index.html` for each idea.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: create dynamic idea detail pages with simulations"
```

---

## Task 12: Create 404 Page

**Files:**
- Create: `src/pages/404.astro`

**Step 1: Create 404 page**

Create `src/pages/404.astro`:
```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import Header from '../components/Header.astro';
import Footer from '../components/Footer.astro';
---

<BaseLayout title="Not Found" description="Page not found">
  <meta slot="head" name="robots" content="noindex, nofollow" />

  <Header />

  <div class="pt-20 min-h-screen flex items-center justify-center">
    <div class="text-center">
      <div class="text-6xl font-heading font-bold text-zinc-700 mb-4">404</div>
      <h1 class="text-2xl font-heading font-bold text-zinc-100 mb-2">
        Page Not Found
      </h1>
      <p class="text-zinc-400 mb-8">
        The requested page does not exist.
      </p>
      <a
        href="/"
        class="inline-block px-6 py-2 border border-zinc-700 text-zinc-300 hover:bg-zinc-800 transition-colors font-mono text-sm"
      >
        Return Home
      </a>
    </div>
  </div>

  <Footer />
</BaseLayout>
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: create 404 page with noindex"
```

---

## Task 13: Create RSS Feed

**Files:**
- Create: `src/pages/rss.xml.ts`

**Step 1: Install RSS package**

Run:
```bash
bun add @astrojs/rss
```

**Step 2: Create RSS endpoint**

Create `src/pages/rss.xml.ts`:
```typescript
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
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add RSS feed endpoint"
```

---

## Task 14: Add Typography Plugin for Prose

**Files:**
- Modify: `tailwind.config.mjs`

**Step 1: Install typography plugin**

Run:
```bash
bun add -d @tailwindcss/typography
```

**Step 2: Update Tailwind config**

Edit `tailwind.config.mjs`:
```javascript
import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        heading: ['Space Grotesk', 'sans-serif'],
        sketch: ['Caveat', 'cursive'],
      },
      gridTemplateColumns: {
        '16': 'repeat(16, minmax(0, 1fr))',
        '32': 'repeat(32, minmax(0, 1fr))',
      },
    },
  },
  plugins: [typography],
};
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add Tailwind typography plugin for prose styling"
```

---

## Task 15: Clean Up Old Files

**Files:**
- Delete: `App.tsx`
- Delete: `index.tsx`
- Delete: `index.html`
- Delete: `scripts/`
- Delete: `generated-ideas.ts`
- Delete: `components/` (remaining old files)
- Delete: `hooks/useIdeas.ts`
- Delete: `pages/`
- Delete: `context/`
- Delete: `services/`
- Delete: `utils/`
- Delete: `types.ts`
- Delete: `config.ts`

**Step 1: Remove old React SPA files**

Run:
```bash
rm -f App.tsx index.tsx index.html generated-ideas.ts types.ts config.ts
rm -rf scripts pages context services utils
rm -rf components/Header.tsx components/Footer.tsx components/IdeaCard.tsx
rm -rf components/SEO.tsx components/ContactModal.tsx components/MarkdownRenderer.tsx
rm -f hooks/useIdeas.ts
# Keep components directory if react folder exists
rmdir components 2>/dev/null || true
rmdir hooks 2>/dev/null || true
```

**Step 2: Verify build still works**

Run:
```bash
bun run build
```

Expected: Build completes successfully.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old React SPA files"
```

---

## Task 16: Final Verification

**Step 1: Run full build**

Run:
```bash
bun run build
```

Expected: Build succeeds, generates static HTML for all pages.

**Step 2: Preview the site**

Run:
```bash
bun run preview
```

Open http://localhost:4321 and verify:
- [ ] Homepage loads with all ideas
- [ ] Navigation works (View Transitions)
- [ ] Idea detail pages render content
- [ ] Simulations load when scrolled into view
- [ ] SEO meta tags present in page source
- [ ] RSS feed accessible at /rss.xml
- [ ] Sitemap generated

**Step 3: Verify SEO**

Run:
```bash
curl -s http://localhost:4321 | grep -E '<title>|<meta property="og:|<link rel="canonical"'
```

Expected: Title, OG tags, and canonical link present.

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete Astro migration with full SEO support"
```

---

## Summary

After completing all tasks:

1. **Static site** - Every page pre-rendered at build time
2. **SEO preserved** - Meta tags, JSON-LD, sitemap, RSS
3. **View Transitions** - SPA-like navigation
4. **React islands** - Simulations hydrate on scroll
5. **Content Collections** - Type-safe markdown with Zod
6. **Tailwind bundled** - Proper CSS purging

Files deleted: ~15 React SPA files
Files created: ~20 Astro files
Bundle size: Significantly smaller (no React runtime for static pages)
