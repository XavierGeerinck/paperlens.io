import React, { useEffect, useState, useMemo } from "react";
import { Globe, Search, FileText, Database, Network } from "lucide-react";

interface Point3D {
	x: number;
	y: number;
	z: number;
}

interface ProjectedPoint {
	x: number;
	y: number;
	z: number; // kept for z-indexing/opacity
}

export const GlobeAnimation: React.FC = () => {
	const [rotation, setRotation] = useState(0);
	const [pings, setPings] = useState<
		{ id: number; lat: number; lon: number; age: number }[]
	>([]);

	// Animation loop
	useEffect(() => {
		let animationFrame: number;
		let lastTime = Date.now();

		const animate = () => {
			const now = Date.now();
			const delta = now - lastTime;
			lastTime = now;

			setRotation((r) => (r + 0.005) % (Math.PI * 2));

			// Randomly add pings
			if (Math.random() < 0.05) {
				setPings((prev) => [
					...prev,
					{
						id: now,
						lat: (Math.random() * 180 - 90) * (Math.PI / 180),
						lon: (Math.random() * 360 - 180) * (Math.PI / 180),
						age: 0,
					},
				]);
			}

			// Update pings
			setPings((prev) =>
				prev
					.map((p) => ({ ...p, age: p.age + delta }))
					.filter((p) => p.age < 2000),
			);

			animationFrame = requestAnimationFrame(animate);
		};

		animate();
		return () => cancelAnimationFrame(animationFrame);
	}, []);

	const radius = 120;
	const width = 300;
	const height = 300;
	const cx = width / 2;
	const cy = height / 2;

	// Project 3D point to 2D
	const project = (p: Point3D, rot: number): ProjectedPoint => {
		// Rotate around Y axis
		const x1 = p.x * Math.cos(rot) - p.z * Math.sin(rot);
		const z1 = p.x * Math.sin(rot) + p.z * Math.cos(rot);

		// Tilt slightly around X axis (view from slightly above)
		const tilt = 0.3;
		const y2 = p.y * Math.cos(tilt) - z1 * Math.sin(tilt);
		const z2 = p.y * Math.sin(tilt) + z1 * Math.cos(tilt);

		return {
			x: x1 + cx,
			y: y2 + cy,
			z: z2,
		};
	};

	// Generate Globe Lines
	const globeLines = useMemo(() => {
		const lines: Point3D[][] = [];

		// Longitude lines
		for (let lon = 0; lon < 360; lon += 30) {
			const line: Point3D[] = [];
			const lonRad = lon * (Math.PI / 180);
			for (let lat = -90; lat <= 90; lat += 5) {
				const latRad = lat * (Math.PI / 180);
				line.push({
					x: radius * Math.cos(latRad) * Math.cos(lonRad),
					y: radius * Math.sin(latRad),
					z: radius * Math.cos(latRad) * Math.sin(lonRad),
				});
			}
			lines.push(line);
		}

		// Latitude lines
		for (let lat = -60; lat <= 60; lat += 30) {
			const line: Point3D[] = [];
			const latRad = lat * (Math.PI / 180);
			for (let lon = 0; lon <= 360; lon += 5) {
				const lonRad = lon * (Math.PI / 180);
				line.push({
					x: radius * Math.cos(latRad) * Math.cos(lonRad),
					y: radius * Math.sin(latRad),
					z: radius * Math.cos(latRad) * Math.sin(lonRad),
				});
			}
			lines.push(line);
		}

		return lines;
	}, []);

	// Generate Ring Points
	const ringPoints = useMemo(() => {
		const points: Point3D[] = [];
		for (let i = 0; i <= 72; i++) {
			const theta = i * 5 * (Math.PI / 180);
			const r = radius + 40;
			// Create a ring in X-Z plane
			points.push({
				x: r * Math.cos(theta),
				y: 0,
				z: r * Math.sin(theta),
			});
		}
		return points;
	}, []);

	return (
		<div className="w-full h-full min-h-[400px] flex flex-col items-center justify-center relative group">
			{/* Background Glow */}
			<div className="absolute inset-0 bg-indigo-500/5 blur-3xl rounded-full transform scale-75" />

			{/* Globe SVG */}
			<svg
				width="100%"
				height="100%"
				viewBox={`0 0 ${width} ${height}`}
				className="overflow-visible z-10"
			>
				<defs>
					<radialGradient id="globeGradient" cx="50%" cy="50%" r="50%">
						<stop offset="0%" stopColor="#6366f1" stopOpacity="0.1" />
						<stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
					</radialGradient>
				</defs>

				{/* Globe Background Sphere */}
				<circle cx={cx} cy={cy} r={radius} fill="url(#globeGradient)" />

				{/* Wireframe Lines */}
				{globeLines.map((line, i) => {
					const projectedLine = line.map((p) => project(p, rotation));

					// Split line into segments to handle z-sorting (simple approach: if avg z < 0, it's back)
					// For a wireframe globe, we often just draw back lines dimmer
					const isBack =
						projectedLine.reduce((acc, p) => acc + p.z, 0) /
							projectedLine.length <
						0;

					const d = `M ${projectedLine.map((p) => `${p.x},${p.y}`).join(" L ")}`;

					return (
						<path
							key={i}
							d={d}
							fill="none"
							stroke={isBack ? "#3f3f46" : "#818cf8"}
							strokeWidth={isBack ? 1 : 1.5}
							strokeOpacity={isBack ? 0.2 : 0.6}
							className="transition-colors duration-75"
						/>
					);
				})}

				{/* Pings */}
				{pings.map((ping) => {
					const p3d = {
						x: radius * Math.cos(ping.lat) * Math.cos(ping.lon),
						y: radius * Math.sin(ping.lat),
						z: radius * Math.cos(ping.lat) * Math.sin(ping.lon),
					};
					const proj = project(p3d, rotation);

					// Only show if on front side
					if (proj.z < 0) return null;

					const opacity = 1 - ping.age / 2000;
					const scale = 1 + ping.age / 1000;

					return (
						<g key={ping.id} transform={`translate(${proj.x}, ${proj.y})`}>
							<circle r={2 * scale} fill="#fff" fillOpacity={opacity} />
							<circle
								r={4 * scale}
								stroke="#6366f1"
								strokeWidth="1"
								fill="none"
								strokeOpacity={opacity}
							/>
						</g>
					);
				})}

				{/* Scanning Ring (3D) */}
				{(() => {
					// Counter-rotate the ring
					const ringRot = -rotation * 1.5;
					const projectedRing = ringPoints.map((p) => project(p, ringRot));

					// Split into front and back segments for depth sorting
					const frontPoints: string[] = [];
					const backPoints: string[] = [];

					for (let i = 0; i < projectedRing.length - 1; i++) {
						const p1 = projectedRing[i];
						const p2 = projectedRing[i + 1];
						const isBack = (p1.z + p2.z) / 2 < 0;
						const d = `M ${p1.x},${p1.y} L ${p2.x},${p2.y}`;
						if (isBack) backPoints.push(d);
						else frontPoints.push(d);
					}

					return (
						<>
							{/* Back Ring Segments */}
							<path
								d={backPoints.join(" ")}
								fill="none"
								stroke="#3f3f46"
								strokeWidth="1"
								strokeDasharray="4 4"
								className="opacity-30"
							/>
							{/* Front Ring Segments */}
							<path
								d={frontPoints.join(" ")}
								fill="none"
								stroke="#6366f1"
								strokeWidth="2"
								strokeDasharray="4 4"
								className="opacity-80"
							/>
						</>
					);
				})()}
			</svg>

			{/* Overlay UI */}
			<div className="absolute bottom-10 left-0 right-0 flex justify-center gap-8">
				<div className="flex flex-col items-center gap-1">
					<div className="flex items-center gap-2 text-indigo-400">
						<Network className="w-4 h-4" />
						<span className="text-xs font-mono font-bold">GLOBAL_NET</span>
					</div>
					<span className="text-[10px] text-zinc-500 font-mono uppercase tracking-widest">
						Scanning
					</span>
				</div>
				<div className="w-px h-8 bg-zinc-800" />
				<div className="flex flex-col items-center gap-1">
					<div className="flex items-center gap-2 text-emerald-400">
						<Database className="w-4 h-4" />
						<span className="text-xs font-mono font-bold">ARXIV_DB</span>
					</div>
					<span className="text-[10px] text-zinc-500 font-mono uppercase tracking-widest">
						Connected
					</span>
				</div>
			</div>

			{/* Floating "Found" Labels (Simulated) */}
			<div className="absolute top-1/4 left-1/2 -translate-x-1/2 bg-zinc-900/80 backdrop-blur border border-zinc-800 p-2 rounded text-xs font-mono text-zinc-300 animate-pulse">
				<div className="flex items-center gap-2">
					<FileText className="w-3 h-3 text-indigo-400" />
					<span>Indexing New Research...</span>
				</div>
			</div>
		</div>
	);
};
