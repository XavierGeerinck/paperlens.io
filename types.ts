export type ProjectStatus = "CONCEPT" | "PROTOTYPE" | "ALPHA" | "ARCHIVED";
export type ProjectCategory = "idea" | "deep-dive";

export interface Idea {
	id: string;
	title: string;
	subtitle: string;
	date: string;
	status: ProjectStatus;
	category?: ProjectCategory;
	impact: string; // e.g., "100x efficiency gain"
	readTime: string;
	tags: string[];
	coverImage: string;
	markdownPath: string; // Path to the .md file
	pdfUrl?: string;
	demoUrl?: string;
	githubUrl?: string;
	featured?: boolean;
	simulation?: string; // Name of the simulation component to load
	content?: string; // The body of the markdown file
}
