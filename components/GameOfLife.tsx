import React, { useEffect, useRef, useState } from "react";

const CELL_SIZE = 24;
const DEFAULT_FPS = 15;
const OVERCLOCK_FPS = 60; // Faster when overclocked
const DECAY_RATE = 0.03;

const KONAMI_CODE = [
	"ArrowUp",
	"ArrowUp",
	"ArrowDown",
	"ArrowDown",
	"ArrowLeft",
	"ArrowRight",
	"ArrowLeft",
	"ArrowRight",
	"b",
	"a",
];

const GameOfLife: React.FC = () => {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const gridRef = useRef<number[]>([]);
	const trailRef = useRef<number[]>([]);
	const rowsRef = useRef(0);
	const colsRef = useRef(0);
	const animationRef = useRef<number>(0);
	const lastFrameTime = useRef(0);
	const mouseRef = useRef<{ x: number; y: number } | null>(null);

	// Easter Egg State
	const [isOverclocked, setIsOverclocked] = useState(false);
	const keySequence = useRef<string[]>([]);

	// Listen for Konami Code
	useEffect(() => {
		const handleKeyDown = (e: KeyboardEvent) => {
			keySequence.current = [...keySequence.current, e.key].slice(-10);

			if (JSON.stringify(keySequence.current) === JSON.stringify(KONAMI_CODE)) {
				setIsOverclocked((prev) => !prev); // Toggle mode
				console.log("SYSTEM OVERRIDE: ROOT ACCESS GRANTED");
			}
		};

		window.addEventListener("keydown", handleKeyDown);
		return () => window.removeEventListener("keydown", handleKeyDown);
	}, []);

	useEffect(() => {
		const canvas = canvasRef.current;
		const container = containerRef.current;
		if (!canvas || !container) return;

		const ctx = canvas.getContext("2d", { alpha: true });
		if (!ctx) return;

		const initGrid = () => {
			const width = container.clientWidth || window.innerWidth;
			const height = container.clientHeight || window.innerHeight;

			const dpr = window.devicePixelRatio || 1;
			canvas.width = width * dpr;
			canvas.height = height * dpr;

			ctx.scale(dpr, dpr);

			const cols = Math.ceil(width / CELL_SIZE);
			const rows = Math.ceil(height / CELL_SIZE);

			rowsRef.current = rows;
			colsRef.current = cols;

			// When overclocked, spawn more life (30% vs 12%)
			const spawnRate = isOverclocked ? 0.7 : 0.88;
			const newGrid = new Array(cols * rows)
				.fill(0)
				.map(() => (Math.random() > spawnRate ? 1 : 0));
			gridRef.current = newGrid;
			trailRef.current = new Array(cols * rows).fill(0);
		};

		const getIndex = (x: number, y: number) => {
			const wrappedX = (x + colsRef.current) % colsRef.current;
			const wrappedY = (y + rowsRef.current) % rowsRef.current;
			return wrappedY * colsRef.current + wrappedX;
		};

		const update = () => {
			const grid = gridRef.current;
			const trail = trailRef.current;
			const nextGrid = [...grid];
			const cols = colsRef.current;
			const rows = rowsRef.current;

			// Mouse Interaction
			if (mouseRef.current) {
				const mx = Math.floor(mouseRef.current.x / CELL_SIZE);
				const my = Math.floor(mouseRef.current.y / CELL_SIZE);

				for (let dy = -1; dy <= 1; dy++) {
					for (let dx = -1; dx <= 1; dx++) {
						if (Math.random() > 0.5) {
							const idx = getIndex(mx + dx, my + dy);
							nextGrid[idx] = 1;
							trail[idx] = 1;
						}
					}
				}
			}

			// Aggressive glitch/spawn if overclocked
			if (isOverclocked) {
				// Inject multiple random active cells per frame to simulate noise/activity
				for (let i = 0; i < 5; i++) {
					const randomIdx = Math.floor(Math.random() * grid.length);
					nextGrid[randomIdx] = 1;
					trail[randomIdx] = 1;
				}
			}

			for (let y = 0; y < rows; y++) {
				for (let x = 0; x < cols; x++) {
					const idx = y * cols + x;
					const state = grid[idx];

					let neighbors = 0;
					for (let dy = -1; dy <= 1; dy++) {
						for (let dx = -1; dx <= 1; dx++) {
							if (dx === 0 && dy === 0) continue;
							if (grid[getIndex(x + dx, y + dy)]) neighbors++;
						}
					}

					if (state === 1 && (neighbors < 2 || neighbors > 3)) {
						nextGrid[idx] = 0;
					} else if (state === 0 && neighbors === 3) {
						nextGrid[idx] = 1;
					}

					if (nextGrid[idx] === 1) {
						trail[idx] = 1.0;
					} else {
						trail[idx] = Math.max(0, trail[idx] - DECAY_RATE);
					}
				}
			}

			gridRef.current = nextGrid;
		};

		const draw = () => {
			const dpr = window.devicePixelRatio || 1;
			ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);

			const cols = colsRef.current;
			const rows = rowsRef.current;
			const trail = trailRef.current;
			const width = canvas.width / dpr;
			const height = canvas.height / dpr;

			// Grid Lines
			ctx.strokeStyle = isOverclocked
				? "rgba(34, 197, 94, 0.1)"
				: "rgba(255, 255, 255, 0.01)";
			ctx.lineWidth = 1;
			ctx.beginPath();
			for (let x = 0; x <= cols; x++) {
				ctx.moveTo(x * CELL_SIZE, 0);
				ctx.lineTo(x * CELL_SIZE, height);
			}
			for (let y = 0; y <= rows; y++) {
				ctx.moveTo(0, y * CELL_SIZE);
				ctx.lineTo(width, y * CELL_SIZE);
			}
			ctx.stroke();

			// Draw Cells
			for (let i = 0; i < trail.length; i++) {
				if (trail[i] > 0.01) {
					const x = (i % cols) * CELL_SIZE;
					const y = Math.floor(i / cols) * CELL_SIZE;

					const size = CELL_SIZE * 0.85 * trail[i];
					const offset = (CELL_SIZE - size) / 2;

					// Color switching based on mode
					// Standard: Indigo (#6366f1)
					// Overclocked: Matrix Green (#22c55e)
					const colorString = isOverclocked
						? `rgba(34, 197, 94, ${trail[i] * 0.8})` // Very bright when overclocked
						: `rgba(99, 102, 241, ${trail[i] * 0.25})`;

					ctx.fillStyle = colorString;

					ctx.beginPath();
					if (ctx.roundRect) {
						ctx.roundRect(x + offset, y + offset, size, size, 2);
					} else {
						ctx.rect(x + offset, y + offset, size, size);
					}
					ctx.fill();
				}
			}
		};

		const loop = (timestamp: number) => {
			const currentFPS = isOverclocked ? OVERCLOCK_FPS : DEFAULT_FPS;

			if (timestamp - lastFrameTime.current > 1000 / currentFPS) {
				update();
				draw();
				lastFrameTime.current = timestamp;
			}
			animationRef.current = requestAnimationFrame(loop);
		};

		// Re-init if mode changes to redraw colors immediately
		// If container already has size, this runs immediately.
		// If mounting, the timeout below handles it.
		if (container.clientWidth > 0) {
			initGrid();
			if (!animationRef.current) {
				animationRef.current = requestAnimationFrame(loop);
			}
		}

		// Backup Init for first mount or layout shifts
		const timer = setTimeout(() => {
			if (!gridRef.current.length) {
				initGrid();
				if (!animationRef.current) {
					animationRef.current = requestAnimationFrame(loop);
				}
			}
		}, 50);

		const handleResize = () => initGrid();
		const handleMouseMove = (e: MouseEvent) => {
			if (!canvas) return;
			const rect = canvas.getBoundingClientRect();
			mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
		};
		const handleMouseLeave = () => (mouseRef.current = null);

		window.addEventListener("resize", handleResize);
		window.addEventListener("mousemove", handleMouseMove);
		window.addEventListener("mouseout", handleMouseLeave);

		return () => {
			clearTimeout(timer);
			window.removeEventListener("resize", handleResize);
			window.removeEventListener("mousemove", handleMouseMove);
			window.removeEventListener("mouseout", handleMouseLeave);
			if (animationRef.current) {
				cancelAnimationFrame(animationRef.current);
				animationRef.current = 0; // CRITICAL: Reset ref so loop can restart on next effect run
			}
		};
	}, [isOverclocked]); // Re-run effect when mode changes

	return (
		<div
			ref={containerRef}
			className="absolute inset-0 z-[-1] overflow-hidden bg-[#09090b] pointer-events-none"
		>
			<canvas
				ref={canvasRef}
				className="block w-full h-full"
				style={{ width: "100%", height: "100%" }}
			/>

			{/* Grain Overlay - slightly reduced in overclock mode for clarity */}
			<div
				className="absolute inset-0 opacity-[0.07] mix-blend-overlay pointer-events-none transition-opacity duration-500"
				style={{
					backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
					backgroundRepeat: "repeat",
					opacity: isOverclocked ? 0.04 : 0.07,
				}}
			/>

			<div className="absolute inset-0 bg-gradient-to-t from-[#09090b] via-transparent to-transparent" />
			<div className="absolute inset-0 bg-gradient-to-b from-[#09090b]/90 via-transparent to-[#09090b]/40" />

			{/* Toast for Mode Change */}
			{isOverclocked && (
				<div className="absolute top-20 right-8 bg-green-900/20 border border-green-500/50 text-green-400 px-4 py-2 font-mono text-xs uppercase tracking-widest animate-pulse z-50">
					System Overclocked: Root Access
				</div>
			)}
		</div>
	);
};

export default GameOfLife;
