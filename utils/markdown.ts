import { Idea, ProjectStatus, ProjectCategory } from "../types";

interface ParsedMarkdown {
	metadata: Partial<Idea>;
	content: string;
}

export const parseMarkdown = (
	text: string,
	id: string,
	path: string,
): Idea & { content: string } => {
	const frontmatterRegex = /^---\n([\s\S]*?)\n---\n/;
	const match = text.match(frontmatterRegex);

	const metadata: any = {
		id,
		markdownPath: path,
		tags: [],
		// Defaults
		title: "Untitled Project",
		subtitle: "No description available",
		date: new Date().toISOString().split("T")[0],
		status: "CONCEPT",
		category: "idea",
		impact: "Unknown",
		readTime: "5m",
		coverImage: "https://picsum.photos/800/600?grayscale",
		featured: false,
	};

	let content = text;

	if (match) {
		const frontmatterBlock = match[1];
		content = text.replace(frontmatterRegex, "");

		// Parse key-value pairs
		const lines = frontmatterBlock.split("\n");
		let currentKey: string | null = null;

		lines.forEach((line) => {
			// Handle array items (tags)
			if (line.trim().startsWith("- ") && currentKey === "tags") {
				const value = line.trim().substring(2).trim();
				if (value) metadata.tags.push(value);
				return;
			}

			const keyMatch = line.match(/^([a-zA-Z0-9]+):\s*(.*)/);
			if (keyMatch) {
				const key = keyMatch[1];
				let value = keyMatch[2].trim();

				// Handle boolean
				if (value === "true") value = true as any;
				if (value === "false") value = false as any;

				// Handle start of array
				if (!value) {
					currentKey = key;
				} else {
					currentKey = null;
					metadata[key] = value;
				}
			}
		});
	}

	return {
		...metadata,
		content,
	};
};
