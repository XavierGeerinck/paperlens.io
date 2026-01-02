import React from "react";
import { Link } from "react-router-dom";
import { ArrowRight, FileText, Activity } from "lucide-react";
import { Idea } from "../types";
import { LabCard, DataReadout, TechBadge } from "./SketchElements";

interface IdeaCardProps {
	idea: Idea;
	variant?: "standard" | "featured";
}

const IdeaCard: React.FC<IdeaCardProps> = ({ idea, variant = "standard" }) => {
	const statusColor = {
		CONCEPT: "text-blue-400 bg-blue-400/10 border-blue-400/20",
		PROTOTYPE: "text-amber-400 bg-amber-400/10 border-amber-400/20",
		ALPHA: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
		ARCHIVED: "text-zinc-500 bg-zinc-500/10 border-zinc-500/20",
	}[idea.status];

	if (variant === "featured") {
		return (
			<Link to={`/idea/${idea.id}`} className="group block w-full">
				<LabCard
					title={`REF: ${idea.id.toUpperCase()}`}
					className="border-indigo-500/30 bg-zinc-900/20 hover:border-indigo-500/60"
				>
					<div className="flex flex-col md:flex-row gap-8">
						{/* Visual */}
						<div className="w-full md:w-5/12 aspect-video md:aspect-auto relative bg-zinc-950 border border-zinc-800 overflow-hidden group-hover:border-indigo-500/30 transition-colors">
							<img
								src={idea.coverImage}
								alt={idea.title}
								className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity duration-500 grayscale group-hover:grayscale-0"
							/>
							<div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />
							<div className="absolute bottom-2 left-2 px-2 py-1 bg-black/80 text-[9px] font-mono text-zinc-400 border border-zinc-800">
								FIG 1.0 // PREVIEW
							</div>
						</div>

						{/* Spec Sheet */}
						<div className="w-full md:w-7/12 flex flex-col justify-center py-2">
							<div className="flex items-center gap-3 mb-4">
								<span
									className={`px-2 py-0.5 text-[9px] font-mono uppercase tracking-widest border ${statusColor}`}
								>
									{idea.status}
								</span>
								<span className="text-[10px] font-mono text-zinc-500">
									{idea.date}
								</span>
							</div>

							<h3 className="text-3xl font-bold text-white mb-4 font-space tracking-tight group-hover:text-indigo-400 transition-colors">
								{idea.title}
							</h3>

							<p className="text-zinc-400 font-light leading-relaxed mb-8 border-l border-zinc-800 pl-4">
								{idea.subtitle}
							</p>

							<div className="grid grid-cols-2 gap-6 mb-6 bg-zinc-950/50 p-4 border border-zinc-800/50">
								<DataReadout label="PROJECTED IMPACT" value={idea.impact} />
								<DataReadout label="READ TIME" value={idea.readTime} />
							</div>

							<div className="flex items-center gap-2 text-indigo-400 font-mono text-xs font-bold uppercase tracking-widest mt-auto">
								<span className="group-hover:underline underline-offset-4">
									Access File
								</span>
								<ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
							</div>
						</div>
					</div>
				</LabCard>
			</Link>
		);
	}

	// Standard Module
	return (
		<Link to={`/idea/${idea.id}`} className="group h-full block">
			<LabCard className="h-full flex flex-col hover:bg-zinc-900/30 transition-all">
				<div className="flex justify-between items-start mb-4">
					<div className="p-1.5 border border-zinc-800 bg-zinc-900 text-zinc-500 group-hover:text-indigo-400 group-hover:border-indigo-500/30 transition-colors">
						<FileText className="w-4 h-4" />
					</div>
					<TechBadge
						label={idea.status}
						color={
							idea.status === "ALPHA" ? "text-emerald-400" : "text-zinc-500"
						}
					/>
				</div>

				<h3 className="text-lg font-bold text-zinc-100 mb-2 font-space leading-tight group-hover:text-indigo-300 transition-colors">
					{idea.title}
				</h3>

				<p className="text-sm text-zinc-500 leading-relaxed mb-6 font-mono line-clamp-3 flex-grow">
					{idea.subtitle}
				</p>

				<div className="pt-4 border-t border-zinc-800/50 mt-auto">
					<div className="flex flex-wrap gap-2 mb-4">
						{idea.tags.slice(0, 3).map((tag) => (
							<span
								key={tag}
								className="text-[9px] text-zinc-600 uppercase font-mono bg-zinc-900 px-1.5 py-0.5 rounded-sm"
							>
								{tag}
							</span>
						))}
					</div>
					<div className="flex items-center justify-between text-zinc-600 group-hover:text-zinc-300 transition-colors">
						<span className="text-[9px] font-mono uppercase tracking-widest">
							View Specs
						</span>
						<ArrowRight className="w-3 h-3" />
					</div>
				</div>
			</LabCard>
		</Link>
	);
};

export default IdeaCard;
