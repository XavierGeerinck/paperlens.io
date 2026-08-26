import React from "react";

/**
 * Shared chrome for the simulations.
 *
 * These follow the site's pane treatment: 1px border, 6px radius, the label
 * sitting in the border like a legend. Colour comes from the tokens in
 * src/styles/global.css, so simulations stay on-brand without knowing about it.
 */

export const LabCard: React.FC<{
	children: React.ReactNode;
	title?: string;
	status?: string;
	className?: string;
}> = ({ children, title, status, className = "" }) => (
	<div
		className={`relative bg-bg0 border border-bg2 rounded-lg ${className}`}
	>
		{(title || status) && (
			<div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-bg2">
				{title && (
					<span className="text-[12.5px] font-mono font-semibold text-mint-400 truncate">
						{title}
					</span>
				)}
				{status && (
					<span className="text-[11px] font-mono text-ink4 whitespace-nowrap">
						<span className="text-mute">// </span>
						{status}
					</span>
				)}
			</div>
		)}

		<div className="p-4">{children}</div>
	</div>
);

export const TechBadge: React.FC<{ label: string; color?: string }> = ({
	label,
	color = "text-mint-400",
}) => (
	<span
		className={`inline-flex items-center px-1.5 rounded text-[11px] font-mono font-semibold tracking-wide border border-bg3 bg-bg0h ${color}`}
	>
		{label}
	</span>
);

export const DataReadout: React.FC<{
	label: string;
	value: string | React.ReactNode;
}> = ({ label, value }) => (
	<div className="flex flex-col">
		<span className="text-[11px] font-mono text-mute mb-0.5">{label}</span>
		<span className="text-sm font-mono text-ink1 tabular-nums truncate">
			{value}
		</span>
	</div>
);

// --- Aliases kept so existing simulations keep working ---

export const SketchCircle: React.FC<{
	children: React.ReactNode;
	className?: string;
	color?: string;
}> = ({ children, className = "" }) => (
	<div className={`relative inline-block ${className}`}>{children}</div>
);

export const SketchBox = LabCard;
export const SchematicCard = LabCard;

export const SchematicButton = ({
	onClick,
	children,
	icon: Icon,
	label,
	active,
	disabled,
}: any) => {
	const iconElement = React.isValidElement(Icon) ? (
		Icon
	) : Icon ? (
		<Icon size={14} />
	) : null;

	return (
		<button
			type="button"
			onClick={onClick}
			disabled={disabled}
			className={`px-3 py-1.5 rounded-lg border font-mono text-[12.5px] transition-colors flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed ${
				active
					? "bg-mint-400 border-mint-400 text-onhue font-semibold"
					: "bg-bg1 border-bg3 text-ink1 hover:border-ink4 hover:text-ink"
			}`}
		>
			{iconElement}
			{children || label}
		</button>
	);
};

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
		aria-hidden="true"
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
}> = ({ className = "" }) => (
	<div className={`h-px w-full bg-bg3 ${className}`} />
);

export const SketchHighlight: React.FC<{
	className?: string;
	color?: string;
}> = ({ className = "" }) => (
	<div className={`absolute inset-0 bg-mint-400/10 ${className}`} />
);

export const SketchFilters = () => null;
