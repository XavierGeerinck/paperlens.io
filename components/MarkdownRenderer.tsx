import React, { useEffect, useState, useId } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Terminal, Copy, Check } from "lucide-react";
import mermaid from "mermaid";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

interface MarkdownRendererProps {
	content: string;
}

const MermaidDiagram = ({ chart }: { chart: string }) => {
	const [svg, setSvg] = useState("");
	const id = useId().replace(/:/g, ""); // Ensure valid ID for DOM

	useEffect(() => {
		mermaid.initialize({
			startOnLoad: false,
			theme: "base",
			securityLevel: "loose",
			fontFamily: '"JetBrains Mono", monospace',
			themeVariables: {
				darkMode: true,
				background: "#09090b",
				primaryColor: "#18181b",
				primaryTextColor: "#e4e4e7",
				primaryBorderColor: "#3f3f46",
				lineColor: "#6366f1",
				secondaryColor: "#27272a",
				tertiaryColor: "#18181b",
				noteBkgColor: "#18181b",
				noteTextColor: "#a1a1aa",
			},
		});

		const renderChart = async () => {
			try {
				const { svg } = await mermaid.render(`mermaid-${id}`, chart);
				setSvg(svg);
			} catch (error) {
				console.error("Mermaid render error:", error);
				setSvg(
					'<div class="text-red-500 font-mono text-xs p-4 border border-red-900/50 bg-red-950/20 rounded">Failed to render diagram syntax</div>',
				);
			}
		};
		renderChart();
	}, [chart, id]);

	return (
		<div
			className="my-10 flex justify-center bg-zinc-950/40 p-6 rounded-xl border border-zinc-800/50 overflow-x-auto shadow-inner"
			dangerouslySetInnerHTML={{ __html: svg }}
		/>
	);
};

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
	const [copiedIndex, setCopiedIndex] = React.useState<number | null>(null);

	const handleCopy = (code: string, index: number) => {
		navigator.clipboard.writeText(code);
		setCopiedIndex(index);
		setTimeout(() => setCopiedIndex(null), 2000);
	};

	return (
		<div className="w-full">
			<ReactMarkdown
				remarkPlugins={[remarkMath, remarkGfm]}
				rehypePlugins={[rehypeKatex]}
				components={{
					table: ({ node, ...props }) => (
						<div className="overflow-x-auto my-10 rounded-xl border border-zinc-800 bg-zinc-900/40 backdrop-blur-sm">
							<table className="w-full text-left border-collapse" {...props} />
						</div>
					),
					thead: ({ node, ...props }) => (
						<thead
							className="bg-zinc-900/80 border-b border-zinc-700"
							{...props}
						/>
					),
					tr: ({ node, ...props }) => (
						<tr
							className="border-b border-zinc-800/50 last:border-0 hover:bg-white/5 transition-colors"
							{...props}
						/>
					),
					th: ({ node, ...props }) => (
						<th
							className="px-6 py-4 font-bold text-white text-sm uppercase tracking-wider font-space"
							{...props}
						/>
					),
					td: ({ node, ...props }) => (
						<td
							className="px-6 py-4 text-zinc-300 text-sm leading-relaxed"
							{...props}
						/>
					),
					h1: ({ node, ...props }) => (
						<h1
							className="text-4xl md:text-5xl font-bold text-white mb-8 mt-4 font-space tracking-tight leading-tight"
							{...props}
						/>
					),
					h2: ({ node, ...props }) => (
						<div className="mt-16 mb-6">
							<h2
								className="text-2xl md:text-3xl font-bold text-indigo-200 font-space flex items-center gap-3"
								{...props}
							/>
							<div className="h-px w-full bg-gradient-to-r from-indigo-500/50 to-transparent mt-4" />
						</div>
					),
					h3: ({ node, ...props }) => (
						<h3
							className="text-xl md:text-2xl font-bold text-zinc-100 mt-10 mb-4 font-space"
							{...props}
						/>
					),
					p: ({ node, ...props }) => (
						<p
							className="text-zinc-400 leading-8 mb-6 text-lg font-light tracking-wide"
							{...props}
						/>
					),
					ul: ({ node, ...props }) => (
						<ul className="list-none space-y-3 mb-8 ml-2" {...props} />
					),
					ol: ({ node, ...props }) => (
						<ol
							className="list-decimal list-outside ml-6 mb-8 space-y-3 text-zinc-400"
							{...props}
						/>
					),
					li: ({ node, ...props }) => (
						<li className="flex gap-3 text-zinc-300 items-start">
							<span className="mt-2 w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0" />
							<span className="leading-relaxed">{props.children}</span>
						</li>
					),
					blockquote: ({ node, ...props }) => (
						<blockquote className="relative border-l-4 border-indigo-500 bg-zinc-900/40 p-8 rounded-r-xl italic text-zinc-300 my-10 shadow-lg backdrop-blur-sm">
							<span className="absolute top-4 left-4 text-6xl text-indigo-500/10 font-serif leading-none">
								"
							</span>
							{props.children}
						</blockquote>
					),
					a: ({ node, ...props }) => (
						<a
							className="text-indigo-400 hover:text-white font-medium underline underline-offset-4 decoration-indigo-500/30 hover:decoration-indigo-500 transition-all"
							{...props}
						/>
					),
					strong: ({ node, ...props }) => (
						<strong className="text-white font-semibold" {...props} />
					),
					hr: ({ node, ...props }) => (
						<hr className="border-zinc-800 my-12" {...props} />
					),
					img: ({ node, ...props }) => (
						<figure className="my-10">
							<img
								className="rounded-xl shadow-2xl border border-zinc-800 w-full"
								{...props}
								alt={props.alt || ""}
							/>
							{props.alt && (
								<figcaption className="text-center text-xs text-zinc-500 mt-3 font-mono uppercase tracking-widest">
									{props.alt}
								</figcaption>
							)}
						</figure>
					),
					code(props: any) {
						const { node, inline, className, children, ...rest } = props;
						const match = /language-(\w+)/.exec(className || "");
						const codeString = String(children).replace(/\n$/, "");
						const codeId = codeString.length;

						// Check for mermaid
						if (!inline && match && match[1] === "mermaid") {
							return <MermaidDiagram chart={codeString} />;
						}

						return !inline && match ? (
							<div className="rounded-xl overflow-hidden my-10 border border-zinc-800 shadow-2xl bg-[#0d0d0d] group relative">
								{/* Window Header */}
								<div className="flex items-center justify-between px-4 py-3 bg-zinc-900/80 border-b border-zinc-800 backdrop-blur-md">
									<div className="flex gap-2">
										<div className="w-3 h-3 rounded-full bg-[#ff5f56] border border-[#e0443e]" />
										<div className="w-3 h-3 rounded-full bg-[#ffbd2e] border border-[#dea123]" />
										<div className="w-3 h-3 rounded-full bg-[#27c93f] border border-[#1aab29]" />
									</div>
									<div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2 text-xs text-zinc-500 font-mono font-medium opacity-60 group-hover:opacity-100 transition-opacity">
										<Terminal className="w-3 h-3" />
										{match[1].toUpperCase()}
									</div>
									<button
										onClick={() => handleCopy(codeString, codeId)}
										className="text-zinc-500 hover:text-white transition-colors p-1.5 rounded hover:bg-white/10"
										title="Copy Code"
									>
										{copiedIndex === codeId ? (
											<Check className="w-3.5 h-3.5 text-green-500" />
										) : (
											<Copy className="w-3.5 h-3.5" />
										)}
									</button>
								</div>

								{/* Code Content */}
								<SyntaxHighlighter
									style={vscDarkPlus as any}
									language={match[1]}
									PreTag="div"
									customStyle={{
										margin: 0,
										padding: "1.5rem",
										background: "transparent",
										fontSize: "0.875rem",
										lineHeight: "1.6",
										fontFamily: '"JetBrains Mono", monospace',
									}}
									showLineNumbers={true}
									lineNumberStyle={{
										minWidth: "2.5em",
										paddingRight: "1em",
										color: "#52525b",
										textAlign: "right",
									}}
									{...rest}
								>
									{codeString}
								</SyntaxHighlighter>
							</div>
						) : (
							<code
								className="bg-zinc-800/80 text-indigo-300 px-1.5 py-0.5 rounded font-mono text-sm border border-zinc-700/50 shadow-sm"
								{...rest}
							>
								{children}
							</code>
						);
					},
				}}
			>
				{content}
			</ReactMarkdown>
		</div>
	);
};

export default MarkdownRenderer;
