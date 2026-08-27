import { defineCollection, z } from 'astro:content';

const ideasCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    subtitle: z.string(),
    date: z.coerce.date(),
    status: z.enum(['RESEARCH', 'CONCEPT', 'PROTOTYPE', 'ALPHA', 'ARCHIVED']),
    category: z.enum(['idea', 'paper', 'deep-dive', 'tutorial', 'concept']).default('idea'),
    impact: z.string(),
    readTime: z.string(),
    tags: z.array(z.string()),
    /** optional override; entries fall back to the card scripts/og.ts draws */
    coverImage: z.string().optional(),
    simulation: z.string().optional(),
    pdfUrl: z.string().optional(),
    githubUrl: z.string().optional(),
    webUrl: z.string().optional(),
    featured: z.boolean().default(false),
  }),
});

export const collections = { ideas: ideasCollection };
