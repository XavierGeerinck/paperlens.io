import React from "react";
import { USER_CONFIG } from "../config";
import IdeaCard from "../components/IdeaCard";
import {
	FlaskConical,
	Microscope,
	GitBranch,
	ArrowDownRight,
	Database,
	Terminal,
	Loader2,
} from "lucide-react";
import { useUI } from "../context/UIContext";
import ScrambleText from "../components/ScrambleText";
import { TechBadge } from "../components/SketchElements";
import { useIdeas } from "../hooks/useIdeas";
import SEO from "../components/SEO";

const Home: React.FC = () => {
	const { ideas, loading, error } = useIdeas();
	const { openContact } = useUI();

	const featuredIdea = ideas.find((i) => i.featured);
	const otherIdeas = ideas.filter((i) => i.id !== featuredIdea?.id);

	// Helper to render bio with bold name
	const renderBio = () => {
		const parts = USER_CONFIG.bio.split("%NAME%");
		return (
			<p className="text-lg text-zinc-400 font-light leading-relaxed max-w-2xl">
				{parts[0]}
				<strong>{USER_CONFIG.name}</strong>
				{parts[1]}
			</p>
		);
	};

	return (
		<div className="relative">
			<SEO />

			{/* Hero / Mission Log */}
			<section className="relative border-b border-zinc-800">
				{/* Personnel Header */}
				<div className="flex flex-col md:flex-row md:items-center justify-between gap-6 border-b border-zinc-800/50 pb-8 mb-8">
					<div className="flex items-center gap-4">
						<div className="relative group">
							<div className="w-14 h-14 bg-zinc-900 border border-zinc-700 overflow-hidden grayscale group-hover:grayscale-0 transition-all duration-500">
								<img
									src={USER_CONFIG.avatar}
									onError={(e) => {
										e.currentTarget.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(USER_CONFIG.name)}&background=18181b&color=71717a`;
									}}
									alt={USER_CONFIG.name}
									className="w-full h-full object-cover opacity-80 group-hover:opacity-100"
								/>
							</div>
							{/* Online Indicator */}
							<div className="absolute -bottom-1 -right-1 w-3.5 h-3.5 bg-emerald-500 border-2 border-zinc-950" />
						</div>
						<div>
							<h2 className="text-base font-bold text-white font-mono tracking-wider uppercase">
								{USER_CONFIG.name}
							</h2>
							<div className="text-[10px] text-indigo-400 font-mono uppercase tracking-widest flex items-center gap-2 mt-1">
								<Terminal className="w-3 h-3" />
								<span>{USER_CONFIG.role}</span>
								<span className="text-zinc-700">|</span>
								<span>{USER_CONFIG.lab}</span>
							</div>
						</div>
					</div>

					<div className="flex gap-4">
						<div className="px-4 py-2 bg-zinc-900/50 border border-zinc-800">
							<div className="text-[9px] text-zinc-500 font-mono uppercase tracking-widest mb-1">
								Clearance
							</div>
							<div className="text-xs text-zinc-300 font-mono font-bold">
								{USER_CONFIG.clearance}
							</div>
						</div>
						<div className="px-4 py-2 bg-zinc-900/50 border border-zinc-800 hidden sm:block">
							<div className="text-[9px] text-zinc-500 font-mono uppercase tracking-widest mb-1">
								Location
							</div>
							<div className="text-xs text-zinc-300 font-mono font-bold">
								{USER_CONFIG.location}
							</div>
						</div>
					</div>
				</div>

				<div className="flex flex-col gap-6">
					<div className="inline-flex items-center gap-2 px-2 py-1 bg-indigo-500/10 border border-indigo-500/20 w-fit">
						<FlaskConical className="w-3 h-3 text-indigo-400" />
						<span className="text-[10px] font-mono text-indigo-300 uppercase tracking-widest">
							Research Directive 2025
						</span>
					</div>

					<h1 className="text-4xl md:text-6xl font-bold text-white tracking-tight font-space max-w-4xl leading-none">
						BUILDING THE <br />
						<span className="text-indigo-500">
							<ScrambleText text="IMPOSSIBLE" />
						</span>
					</h1>

					<div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-4">
						<div className="md:col-span-2">
							{/* Dynamically rendered bio to keep the structure intact but name configurable */}
							{renderBio()}
						</div>
						<div className="border-l border-zinc-800 pl-6 flex flex-col justify-center space-y-2">
							<div className="flex justify-between text-xs font-mono text-zinc-500">
								<span>EXPERIMENTS</span>
								<span className="text-zinc-200">{ideas.length}</span>
							</div>
							<div className="flex justify-between text-xs font-mono text-zinc-500">
								<span>SUCCESS RATE</span>
								<span className="text-zinc-200">73.4%</span>
							</div>
							<div className="w-full h-1 bg-zinc-900 mt-2">
								<div className="h-full bg-indigo-500 w-[73.4%]" />
							</div>
						</div>
					</div>
				</div>
			</section>

			<section className="mt-16 space-y-16 flex flex-col">
				{/* Loading State */}
				{loading && (
					<div className="flex items-center justify-center py-20">
						<Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
					</div>
				)}

				{/* Error State */}
				{error && (
					<div className="p-4 border border-red-500/50 bg-red-900/20 text-red-200 font-mono text-sm text-center">
						{error}
					</div>
				)}

				{!loading && !error && (
					<>
						{/* Featured Experiment */}
						{featuredIdea && (
							<section className="relative z-10">
								<div className="flex items-center gap-2 mb-6">
									<Microscope className="w-4 h-4 text-indigo-400" />
									<h3 className="text-xs font-mono text-zinc-500 uppercase tracking-widest">
										Priority Focus
									</h3>
									<div className="h-px bg-zinc-800 flex-grow ml-4" />
								</div>
								<IdeaCard idea={featuredIdea} variant="featured" />
							</section>
						)}

						{/* Experiment Log */}
						<section className="relative z-10 pb-24">
							<div className="flex items-center justify-between mb-8">
								<div className="flex items-center gap-2">
									<Database className="w-4 h-4 text-zinc-600" />
									<h3 className="text-xs font-mono text-zinc-500 uppercase tracking-widest">
										Project Archive
									</h3>
								</div>

								<div className="flex gap-2">
									<TechBadge label="Sort: Date" color="text-zinc-500" />
									<TechBadge label="Filter: All" color="text-zinc-500" />
								</div>
							</div>

							<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
								{otherIdeas.map((idea) => (
									<IdeaCard key={idea.id} idea={idea} />
								))}
							</div>
						</section>
					</>
				)}
			</section>

			{/* Collaboration Protocol */}
			<section className="py-16 border-t border-zinc-800 bg-zinc-900/20">
				<div className="max-w-2xl mx-auto text-center">
					<GitBranch className="w-8 h-8 text-zinc-500 mx-auto mb-6" />
					<h2 className="text-2xl font-bold text-white mb-4 font-space">
						Initiate Collaboration Protocol
					</h2>
					<p className="text-zinc-500 mb-8 font-mono text-sm">
						Have a hypothesis? Found a bug in my logic?
						<br />
						The lab is open for peer review.
					</p>
					<button
						onClick={openContact}
						className="inline-flex items-center gap-2 bg-zinc-100 text-zinc-900 px-6 py-3 font-mono text-xs font-bold uppercase tracking-widest hover:bg-white transition-colors"
					>
						Open Frequency <ArrowDownRight className="w-4 h-4" />
					</button>
				</div>
			</section>
		</div>
	);
};

export default Home;
