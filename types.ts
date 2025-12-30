export type ProjectStatus = 'CONCEPT' | 'PROTOTYPE' | 'ALPHA' | 'ARCHIVED';

export interface Idea {
  id: string;
  title: string;
  subtitle: string;
  date: string;
  status: ProjectStatus;
  impact: string; // e.g., "100x efficiency gain"
  readTime: string;
  tags: string[];
  coverImage: string;
  markdownPath: string; // Path to the .md file
  pdfUrl?: string;
  demoUrl?: string;
  githubUrl?: string;
  featured?: boolean;
}