import { readdir, readFile, writeFile } from "node:fs/promises";
import { join, basename } from "node:path";
import { parseMarkdown } from "../utils/markdown";

const CONTENT_DIR = join(process.cwd(), "content");
const OUTPUT_FILE = join(process.cwd(), "generated-ideas.ts");

async function generate() {
	console.log("Generating ideas data...");

	try {
		const files = await readdir(CONTENT_DIR);
		const mdFiles = files.filter((f) => f.endsWith(".md"));

		const ideas = await Promise.all(
			mdFiles.map(async (file) => {
				const filePath = join(CONTENT_DIR, file);
				const content = await readFile(filePath, "utf-8");
				const id = basename(file, ".md");
				const path = `./content/${file}`;

				return parseMarkdown(content, id, path);
			}),
		);

		// Sort by date descending
		ideas.sort(
			(a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
		);

		const tsContent = `// This file is auto-generated. Do not edit manually.
import { Idea } from "./types";

export interface IdeaWithContent extends Idea {
	content: string;
}

export const IDEAS: IdeaWithContent[] = ${JSON.stringify(ideas, null, 2)};
`;

		await writeFile(OUTPUT_FILE, tsContent);
		console.log(
			`Successfully generated ${mdFiles.length} ideas to ${OUTPUT_FILE}`,
		);
	} catch (error) {
		console.error("Failed to generate ideas:", error);
		process.exit(1);
	}
}

generate();
