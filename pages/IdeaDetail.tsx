import React, { useState, useEffect } from "react";
import { useParams, Link, useSearchParams } from "react-router-dom";
import {
	ArrowLeft,
	FileText,
	Code,
	Share2,
	Download,
	ExternalLink,
	Github,
	Terminal,
	AlertCircle,
	Loader2,
	PlayCircle,
	MessageSquare,
} from "lucide-react";
import MarkdownRenderer from "../components/MarkdownRenderer";
import DemoView from "../components/DemoView";
import { useUI } from "../context/UIContext";
import {
	SketchArrowRight,
	SketchCircle,
	SketchHighlight,
	TechBadge,
} from "../components/SketchElements";
import { useIdeas } from "../hooks/useIdeas";
import SEO from "../components/SEO";

const IdeaDetail: React.FC = () => {
	const { id } = useParams<{ id: string }>();
	const [searchParams, setSearchParams] = useSearchParams();
	const { ideas, loading: ideasLoading } = useIdeas();
	const { openContact } = useUI();

	// Tab state derived from URL
	const activeTab =
		(searchParams.get("tab") as "paper" | "demo" | "pdf") || "paper";

	const idea = ideas.find((i) => i.id === id);

	const handleTabChange = (tab: "paper" | "demo" | "pdf") => {
		setSearchParams({ tab }, { replace: true });
	};

	if (ideasLoading) {
		return (
			<div className="flex items-center justify-center min-h-[50vh]">
				<Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
			</div>
		);
	}

	if (!idea) {
		return (
			<div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
				<div className="font-mono text-red-500 text-6xl mb-4">404</div>
				<h2 className="text-2xl font-bold text-white mb-4">
					DATA CORRUPTED OR MISSING
				</h2>
				<Link
					to="/"
					className="text-zinc-400 hover:text-white font-mono uppercase tracking-widest text-sm border-b border-zinc-600 hover:border-white pb-1"
				>
					Return to Index
				</Link>
			</div>
		);
	}

	const statusColor = idea.status === "CONCEPT" ? "#3b82f6" : "#22c55e";
	const ctaText =
		idea.category === "deep-dive" ? "Questions or Remarks?" : "Discuss Idea";

	return (
		<div className="animate-in fade-in duration-500 min-h-screen">
			<SEO
				title={idea.title}
				description={idea.subtitle}
				image={idea.coverImage}
				type="article"
			/>
			<div className="fixed inset-0 bg-grid z-[-1] opacity-10 pointer-events-none" />

			{/* Top Bar Navigation */}
			<div className="flex items-center text-xs font-mono text-zinc-500 mb-8 uppercase tracking-widest gap-2">
				<Link to="/" className="hover:text-white transition-colors">
					Index
				</Link>
				<span>/</span>
				<span className="text-zinc-300">Projects</span>
				<span>/</span>
				<span className="text-indigo-400">{idea.id}</span>
			</div>

			{/* Project Header */}
			<header className="mb-12 border-b border-zinc-800 pb-12 relative">
				<div className="flex flex-col md:flex-row md:items-start justify-between gap-8">
					<div className="flex-1">
						<div className="inline-block transform -rotate-2 mb-6">
							<SketchCircle color={statusColor}>
								<span
									className="px-4 py-1 font-sketch text-2xl font-bold text-white"
									style={{ color: statusColor }}
								>
									Status: {idea.status}
								</span>
							</SketchCircle>
						</div>

						<h1 className="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight tracking-tight relative z-10">
							{idea.title}
							<div className="absolute -bottom-2 left-0 w-32 h-2 opacity-50 z-[-1]">
								<SketchHighlight color="#6366f1" />
							</div>
						</h1>

						<p className="text-2xl text-zinc-400 font-sketch max-w-2xl leading-relaxed border-l-4 border-zinc-800 pl-6 transform rotate-[0.5deg]">
							"{idea.subtitle}"
						</p>
					</div>

					<div className="w-full md:w-64 shrink-0 space-y-6 relative">
						{/* Hand drawn arrow pointing to buttons */}
						<div className="absolute -left-20 top-1/2 hidden md:block opacity-50 transform rotate-12">
							<SketchArrowRight color="#71717a" className="w-16" />
						</div>

						<div className="p-4 bg-zinc-900/30 border-2 border-dashed border-zinc-800 transform rotate-1">
							<div className="text-xl font-sketch text-zinc-400 mb-1">
								Target Impact
							</div>
							<div className="text-sm text-indigo-300 font-medium font-mono">
								{idea.impact}
							</div>
						</div>

						<div className="p-4 bg-zinc-900/30 border border-zinc-800">
							<div className="text-[10px] font-mono text-zinc-500 uppercase mb-2">
								Tech Stack
							</div>
							<div className="flex flex-wrap gap-2">
								{idea.tags.map((tag) => (
									<span
										key={tag}
										className="text-sm font-sketch text-zinc-300 px-2 py-1 border border-zinc-800"
									>
										#{tag}
									</span>
								))}
							</div>
						</div>

						{idea.githubUrl && (
							<a
								href={idea.githubUrl}
								target="_blank"
								rel="noreferrer"
								className="flex items-center justify-center gap-2 w-full py-3 bg-zinc-100 text-black hover:bg-white font-bold text-sm transition-colors shadow-[4px_4px_0px_0px_rgba(255,255,255,0.2)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px]"
							>
								<Github className="w-4 h-4" />
								ACCESS REPO
							</a>
						)}
					</div>
				</div>
			</header>

			{/* Main Content Area */}
			<div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
				{/* Left Column: Content */}
				<div className="lg:col-span-9">
					{/* Tabs */}
					<div className="flex border-b border-zinc-800 mb-8 gap-6 overflow-x-auto">
						<button
							onClick={() => handleTabChange("paper")}
							className={`pb-3 text-xs font-mono uppercase tracking-widest transition-colors border-b-2 flex items-center gap-2 ${activeTab === "paper" ? "border-indigo-500 text-white" : "border-transparent text-zinc-500 hover:text-zinc-300"}`}
						>
							<FileText className="w-4 h-4" />
							Documentation
						</button>
						{/* Conditionally render Simulation tab if simulationName is present */}
						{idea.simulation && (
							<button
								onClick={() => handleTabChange("demo")}
								className={`pb-3 text-xs font-mono uppercase tracking-widest transition-colors border-b-2 flex items-center gap-2 ${activeTab === "demo" ? "border-indigo-500 text-white" : "border-transparent text-zinc-500 hover:text-zinc-300"}`}
							>
								<PlayCircle className="w-4 h-4" />
								Simulation
							</button>
						)}
						{idea.pdfUrl && (
							<button
								onClick={() => handleTabChange("pdf")}
								className={`pb-3 text-xs font-mono uppercase tracking-widest transition-colors border-b-2 flex items-center gap-2 ${activeTab === "pdf" ? "border-indigo-500 text-white" : "border-transparent text-zinc-500 hover:text-zinc-300"}`}
							>
								<Download className="w-4 h-4" />
								Source PDF
							</button>
						)}
						{idea.githubUrl && (
							<a href={idea.githubUrl} target="_blank" rel="noreferrer">
								<button className="pb-3 text-xs font-mono uppercase tracking-widest text-zinc-500 hover:text-zinc-300 flex items-center gap-2">
									<Github className="w-4 h-4" />
									GitHub Repo
									<ExternalLink className="w-3 h-3 opacity-50" />
								</button>
							</a>
						)}
					</div>

					<div className="min-h-[500px]">
						{activeTab === "paper" && (
							<div className="prose prose-invert prose-zinc max-w-none animate-in fade-in slide-in-from-bottom-4 duration-500">
								<div className="mb-8 p-1 border border-zinc-800 bg-zinc-900/50 relative">
									<img
										src={idea.coverImage}
										alt="Schematic"
										className="w-full h-auto opacity-80 grayscale hover:grayscale-0 transition-all duration-700"
									/>
									{/* Image Note */}
									<div className="absolute bottom-4 right-4 bg-yellow-100 text-black px-4 py-2 transform -rotate-2 shadow-lg hidden md:block">
										<span className="font-sketch text-xl font-bold">
											Fig 1.1 - Initial Draft
										</span>
									</div>
								</div>

								<MarkdownRenderer content={idea.content} />
							</div>
						)}

						{activeTab === "demo" && (
							<DemoView simulationName={idea.simulation || undefined} />
						)}

						{activeTab === "pdf" && (
							<div className="w-full h-[800px] border border-zinc-800 bg-zinc-900 animate-in fade-in slide-in-from-bottom-4 duration-500">
								<iframe
									src={idea.pdfUrl}
									className="w-full h-full"
									title="PDF Viewer"
								/>
							</div>
						)}
					</div>
				</div>

				{/* Right Column: Meta & Actions */}
				<aside className="lg:col-span-3 space-y-8">
					<div className="border border-zinc-800 p-6 bg-zinc-900/20">
						<h3 className="font-mono text-xs uppercase text-zinc-500 mb-4 tracking-widest">
							Metadata
						</h3>
						<ul className="space-y-4 text-sm">
							<li className="flex justify-between">
								<span className="text-zinc-500">Date</span>
								<span className="text-zinc-300 font-mono">{idea.date}</span>
							</li>
							<li className="flex justify-between">
								<span className="text-zinc-500">Read Time</span>
								<span className="text-zinc-300 font-mono">{idea.readTime}</span>
							</li>
							<li className="flex justify-between">
								<span className="text-zinc-500">Category</span>
								<TechBadge
									label={idea.category || "idea"}
									color={
										idea.category === "deep-dive"
											? "text-purple-400"
											: "text-blue-400"
									}
								/>
							</li>
						</ul>
					</div>

					{/* Sticky Note Style Callout */}
					<div className="relative bg-[#fef3c7] text-zinc-900 p-6 shadow-xl transform rotate-1 transition-transform hover:rotate-0">
						<div className="absolute -top-3 left-1/2 -translate-x-1/2 w-32 h-8 bg-black/10 blur-sm transform -rotate-1" />

						<div className="flex items-start gap-3 mb-2">
							<AlertCircle className="w-6 h-6 text-orange-600 shrink-0" />
							<h3 className="font-sketch font-bold text-2xl text-zinc-900 leading-none mt-1">
								Peer Review
							</h3>
						</div>
						<p className="font-sketch text-xl leading-snug text-zinc-800 mb-4">
							"This is still in hypothesis phase. I need critical feedback on
							the implementation logic!"
						</p>

						<button
							onClick={openContact}
							className="w-full py-2 bg-transparent hover:bg-black/5 text-black border-2 border-black border-dashed font-mono text-xs uppercase tracking-widest flex items-center justify-center gap-2"
						>
							<MessageSquare className="w-4 h-4" />
							Discuss Findings
						</button>
					</div>
				</aside>
			</div>

			{/* Floating CTA Button - Sketchy Style */}
			<button
				onClick={openContact}
				className="fixed bottom-8 right-8 z-40 flex items-center gap-2 bg-[#09090b] text-white px-6 py-4 rounded-sm shadow-[4px_4px_0px_0px_rgba(99,102,241,1)] border-2 border-indigo-500 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none transition-all duration-200 group"
			>
				<MessageSquare className="w-6 h-6 group-hover:animate-bounce" />
				<span className="font-sketch text-2xl font-bold">{ctaText}</span>
			</button>
		</div>
	);
};

export default IdeaDetail;
