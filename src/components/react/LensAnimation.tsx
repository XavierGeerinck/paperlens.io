import React, { useEffect, useState, useRef } from "react";
import { Scan, Activity, Zap } from "lucide-react";

export const LensAnimation: React.FC = () => {
	const [position, setPosition] = useState(0);
	const containerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		let animationFrame: number;
		const startTime = Date.now();

		const animate = () => {
			const now = Date.now();
			const elapsed = now - startTime;
			// Slower, smoother sine wave
			const t = (Math.sin(elapsed / 2000) + 1) / 2;
			setPosition(t);
			animationFrame = requestAnimationFrame(animate);
		};

		animate();

		return () => cancelAnimationFrame(animationFrame);
	}, []);

	// Dimensions for the SVG coordinate system
	const width = 400;
	const height = 200;

	// Generate a noisy exponential curve
	const getPoint = (t: number) => {
		const x = t * width;
		// Base exponential curve
		const baseY = Math.pow(t, 2.5);
		// Add some "market noise"
		const noise = Math.sin(t * 20) * 0.05 * t;

		const ny = baseY + noise;
		// Invert Y (SVG coords) and scale, keeping padding
		const y = height - 40 - ny * (height - 80);
		return { x, y };
	};

	// Generate path data
	const points = [];
	const futurePoints = [];
	for (let i = 0; i <= 100; i++) {
		const t = i / 100;
		const { x, y } = getPoint(t);
		points.push(`${x},${y}`);

		// Generate "future" projection points (diverging lines)
		if (i > 80 && i % 5 === 0) {
			futurePoints.push({ x, y, t });
		}
	}
	const pathData = `M ${points.join(" L ")}`;

	// Lens position - focuses on the "future" part (right side)
	// Oscillates between 0.6 and 0.95
	const lensT = 0.6 + position * 0.35;
	const lensPos = getPoint(lensT);

	return (
		<div className="w-full h-full min-h-[240px] bg-zinc-900/30 border border-zinc-800/50 rounded-xl overflow-hidden relative group backdrop-blur-sm">
			{/* Grid Background */}
			<div
				className="absolute inset-0"
				style={{
					backgroundImage: `linear-gradient(to right, #27272a 1px, transparent 1px), linear-gradient(to bottom, #27272a 1px, transparent 1px)`,
					backgroundSize: "40px 40px",
					opacity: 0.1,
				}}
			/>

			{/* Header / UI Overlay */}
			<div className="absolute top-4 left-4 right-4 flex justify-between items-start z-10">
				<div className="flex items-center gap-2">
					<div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
					<span className="text-[10px] font-mono text-indigo-400 tracking-widest uppercase">
						Live Inference
					</span>
				</div>
				<div className="flex flex-col items-end">
					<span className="text-2xl font-mono font-bold text-white leading-none">
						{(lensT * 100).toFixed(1)}%
					</span>
					<span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mt-1">
						Confidence
					</span>
				</div>
			</div>

			{/* Main Visualization */}
			<svg
				width="100%"
				height="100%"
				viewBox={`0 0 ${width} ${height}`}
				preserveAspectRatio="none"
				className="absolute inset-0"
			>
				<defs>
					<linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
						<stop offset="0%" stopColor="#52525b" stopOpacity="0" />
						<stop offset="50%" stopColor="#6366f1" stopOpacity="0.5" />
						<stop offset="100%" stopColor="#818cf8" stopOpacity="1" />
					</linearGradient>

					<linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
						<stop offset="0%" stopColor="#6366f1" stopOpacity="0.2" />
						<stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
					</linearGradient>

					<filter id="glow">
						<feGaussianBlur stdDeviation="4" result="coloredBlur" />
						<feMerge>
							<feMergeNode in="coloredBlur" />
							<feMergeNode in="SourceGraphic" />
						</feMerge>
					</filter>

					<mask id="lensMask">
						<circle cx={lensPos.x} cy={lensPos.y} r="60" fill="white" />
					</mask>
				</defs>

				{/* Area under curve */}
				<path
					d={`${pathData} L ${width},${height} L 0,${height} Z`}
					fill="url(#areaGradient)"
					className="opacity-50"
				/>

				{/* Base Curve */}
				<path
					d={pathData}
					fill="none"
					stroke="url(#lineGradient)"
					strokeWidth="2"
					className="opacity-40"
				/>

				{/* Highlighted Curve (Inside Lens) */}
				<g mask="url(#lensMask)">
					<path
						d={pathData}
						fill="none"
						stroke="#a5b4fc"
						strokeWidth="3"
						filter="url(#glow)"
					/>
					{/* Future projection lines inside lens */}
					{futurePoints.map((p, i) => (
						<line
							key={i}
							x1={p.x}
							y1={p.y}
							x2={p.x + 20}
							y2={p.y - 20 - i * 5}
							stroke="#818cf8"
							strokeWidth="1"
							strokeDasharray="2 2"
							opacity="0.5"
						/>
					))}
				</g>

				{/* The Lens UI Elements */}
				<g transform={`translate(${lensPos.x}, ${lensPos.y})`}>
					{/* Lens Circle */}
					<circle
						r="60"
						fill="none"
						stroke="#6366f1"
						strokeWidth="1"
						strokeDasharray="4 4"
						opacity="0.3"
					/>
					<circle
						r="55"
						fill="none"
						stroke="#6366f1"
						strokeWidth="0.5"
						opacity="0.2"
					/>

					{/* Scanning Line */}
					<line
						x1="0"
						y1="-60"
						x2="0"
						y2="60"
						stroke="#818cf8"
						strokeWidth="1"
						opacity="0.8"
					>
						<animate
							attributeName="opacity"
							values="0.8;0.2;0.8"
							dur="2s"
							repeatCount="indefinite"
						/>
					</line>

					{/* Data Points */}
					<circle r="3" fill="#fff" filter="url(#glow)" />

					{/* Floating Label */}
					<g transform="translate(10, -40)">
						<rect
							x="0"
							y="0"
							width="60"
							height="16"
							fill="#18181b"
							rx="2"
							opacity="0.8"
						/>
						<text
							x="5"
							y="11"
							fill="#a5b4fc"
							fontSize="8"
							fontFamily="monospace"
						>
							Analyzing...
						</text>
					</g>
				</g>
			</svg>

			{/* Bottom Stats */}
			<div className="absolute bottom-4 left-4 right-4 flex justify-between items-end border-t border-zinc-800/50 pt-4">
				<div className="flex flex-col gap-1">
					<span className="text-[9px] text-zinc-500 font-mono uppercase tracking-widest">
						Experiments
					</span>
					<div className="flex items-center gap-2 text-zinc-300 font-mono text-xs">
						<Activity className="w-3 h-3 text-indigo-500" />
						<span>Active</span>
					</div>
				</div>

				<div className="flex flex-col gap-1 items-end">
					<span className="text-[9px] text-zinc-500 font-mono uppercase tracking-widest">
						Velocity
					</span>
					<div className="flex items-center gap-2 text-zinc-300 font-mono text-xs">
						<Zap className="w-3 h-3 text-yellow-500" />
						<span>High</span>
					</div>
				</div>
			</div>
		</div>
	);
};
