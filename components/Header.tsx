import React from "react";
import { Link, useLocation } from "react-router-dom";
import { Terminal, Activity, Cpu, Radio, LayoutGrid } from "lucide-react";
import { useUI } from "../context/UIContext";

const Header: React.FC = () => {
	const location = useLocation();
	const isHome = location.pathname === "/";
	const { openContact } = useUI();

	return (
		<header className="sticky top-0 z-50 w-full bg-zinc-950/90 backdrop-blur-sm border-b border-zinc-800">
			<div className="container mx-auto px-4 max-w-7xl h-14 flex items-center justify-between">
				{/* Brand */}
				<Link to="/" className="flex items-center gap-3 group">
					<div className="w-8 h-8 bg-zinc-900 border border-zinc-700 flex items-center justify-center group-hover:border-indigo-500 transition-colors">
						<Cpu className="w-4 h-4 text-zinc-400 group-hover:text-indigo-400" />
					</div>
					<div className="flex flex-col">
						<span className="font-mono font-bold text-sm text-zinc-200 tracking-tight leading-none group-hover:text-white">
							MOONSHOT_LABS
						</span>
						<span className="text-[9px] text-zinc-600 font-mono tracking-widest uppercase leading-none mt-1">
							PROTOTYPING FACILITY
						</span>
					</div>
				</Link>

				{/* System Status Ticker */}
				{isHome && (
					<div className="hidden md:flex items-center gap-6 border-l border-r border-zinc-800 px-6 h-full">
						<div className="flex items-center gap-2">
							<div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-[pulse_2s_infinite]" />
							<span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
								SYSTEM: NORMAL
							</span>
						</div>
						<div className="flex items-center gap-2">
							<Activity className="w-3 h-3 text-zinc-600" />
							<span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
								LAB: ACTIVE
							</span>
						</div>
					</div>
				)}

				{/* Controls */}
				<div className="flex items-center gap-4">
					{!isHome && (
						<Link
							to="/"
							className="flex items-center gap-2 text-[10px] font-mono text-zinc-500 hover:text-indigo-400 uppercase tracking-widest transition-colors"
						>
							<LayoutGrid className="w-3 h-3" />
							DASHBOARD
						</Link>
					)}

					<button
						onClick={openContact}
						className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900 border border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500 transition-all text-[10px] font-mono uppercase tracking-widest"
					>
						<Terminal className="w-3 h-3" />
						<span className="hidden sm:inline">CONNECT</span>
					</button>
				</div>
			</div>
		</header>
	);
};

export default Header;
