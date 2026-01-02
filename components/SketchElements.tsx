import React from "react";

// --- Lab / Tech UI Components ---

export const LabCard: React.FC<{
	children: React.ReactNode;
	title?: string;
	status?: string;
	className?: string;
}> = ({ children, title, status, className = "" }) => (
	<div
		className={`relative bg-zinc-950 border border-zinc-800 group hover:border-zinc-600 transition-colors ${className}`}
	>
		{/* Tech Corners */}
		<div className="absolute -top-[1px] -left-[1px] w-2 h-2 border-t border-l border-zinc-500" />
		<div className="absolute -top-[1px] -right-[1px] w-2 h-2 border-t border-r border-zinc-500" />
		<div className="absolute -bottom-[1px] -left-[1px] w-2 h-2 border-b border-l border-zinc-500" />
		<div className="absolute -bottom-[1px] -right-[1px] w-2 h-2 border-b border-r border-zinc-500" />

		{/* Header Strip */}
		{(title || status) && (
			<div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800 bg-zinc-900/50">
				{title && (
					<div className="flex items-center gap-2">
						<div className="w-1.5 h-1.5 bg-indigo-500 rounded-sm animate-pulse" />
						<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-400">
							{title}
						</span>
					</div>
				)}
				{status && (
					<div className="text-[9px] font-mono uppercase tracking-wider text-zinc-500 border border-zinc-700 px-1.5 py-0.5 rounded-sm">
						{status}
					</div>
				)}
			</div>
		)}

		{/* Content */}
		<div className="p-5">{children}</div>
	</div>
);

export const TechBadge: React.FC<{ label: string; color?: string }> = ({
	label,
	color = "text-indigo-400",
}) => (
	<span
		className={`inline-flex items-center px-2 py-1 rounded-sm text-[10px] font-mono uppercase tracking-wider border border-zinc-800 bg-zinc-900/50 ${color}`}
	>
		{label}
	</span>
);

export const DataReadout: React.FC<{
	label: string;
	value: string | React.ReactNode;
}> = ({ label, value }) => (
	<div className="flex flex-col">
		<span className="text-[9px] font-mono text-zinc-600 uppercase tracking-widest mb-0.5">
			{label}
		</span>
		<span className="text-sm font-mono text-zinc-300 truncate">{value}</span>
	</div>
);

// --- Legacy Sketch Components (Kept for compatibility or accent) ---

export const SketchCircle: React.FC<{
	children: React.ReactNode;
	className?: string;
	color?: string;
}> = ({ children, className = "", color = "currentColor" }) => (
	<div className={`relative inline-block ${className}`}>{children}</div>
);

export const SketchBox = LabCard; // Map to new style
export const SchematicCard = LabCard; // Map to new style
export const SchematicButton = ({ onClick, children }: any) => (
	<button
		onClick={onClick}
		className="px-4 py-2 bg-zinc-900 border border-zinc-700 text-zinc-300 font-mono text-xs uppercase hover:bg-zinc-800 hover:text-white transition-colors flex items-center gap-2"
	>
		{children}
	</button>
);
export const DataLabel = DataReadout;

// --- SVGs ---

export const SketchArrowRight: React.FC<{
	className?: string;
	color?: string;
}> = ({ className = "", color = "currentColor" }) => (
	<svg
		className={`overflow-visible ${className}`}
		width="40"
		height="12"
		viewBox="0 0 40 12"
	>
		<path
			d="M0,6 L38,6 M34,2 L39,6 L34,10"
			fill="none"
			stroke={color}
			strokeWidth="1.5"
		/>
	</svg>
);

export const SketchUnderline: React.FC<{
	className?: string;
	color?: string;
}> = ({ className = "", color = "currentColor" }) => (
	<div
		className={`h-[1px] w-full bg-gradient-to-r from-${color} to-transparent opacity-50 ${className}`}
	/>
);

export const SketchHighlight: React.FC<{
	className?: string;
	color?: string;
}> = ({ className = "", color = "rgba(99, 102, 241, 0.1)" }) => (
	<div className={`absolute inset-0 bg-indigo-500/10 ${className}`} />
);

export const SketchFilters = () => null;
