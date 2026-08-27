import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  SchematicCard,
  SchematicButton,
  DataReadout,
  TechBadge,
} from '../SketchElements';
import {
  ToggleLeft,
  SliderHorizontal,
  Video,
  Speaker,
  Mic,
  Cube,
  Activity,
} from 'lucide-react';

const mulberry32 = (seed: number) => {
  return () => {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

type Mode = 'first' | 'third';

const EchoWMSimulation: React.FC = () => {
  const [mode, setMode] = useState<Mode>('first');
  const [calibration, setCalibration] = useState(1); // 0.5 - 2.0
  const [reducedMotion, setReducedMotion] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [pathPoints, setPathPoints] = useState<{ x: number; z: number }[]>([]);
  const [modalities, setModalities] = useState<string[]>([]);
  const [drawProgress, setDrawProgress] = useState(0); // 0..1
  const rng = useRef(mulberry32(0x12345678));

  // Detect prefers-reduced-motion
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mq.matches);
    const handler = () => setReducedMotion(mq.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // Recompute path when mode or calibration changes
  useEffect(() => {
    const steps = [
      { type: 'forward' as const, label: 'Move Forward' },
      { type: 'turnLeft' as const, label: 'Turn Left' },
      { type: 'forward' as const, label: 'Move Forward' },
      { type: 'turnRight' as const, label: 'Turn Right' },
      { type: 'forward' as const, label: 'Move Forward' },
    ];

    const points: { x: number; z: number }[] = [{ x: 0, z: 0 }];
    let yaw = 0;
    const baseForward = 0.5;
    const baseTurn = 0.2;

    steps.forEach((step) => {
      const jitter = (rng.current() - 0.5) * 0.2; // -0.1..0.1
      const jitterTurn = (rng.current() - 0.5) * 0.1; // -0.05..0.05
      let forward = (baseForward + jitter) * calibration;
      let turn = (baseTurn + jitterTurn) * calibration;

      if (step.type === 'forward') {
        const dx = forward * Math.cos(yaw);
        const dz = forward * Math.sin(yaw);
        const last = points[points.length - 1];
        points.push({ x: last.x + dx, z: last.z + dz });
      } else if (step.type === 'turnLeft') {
        yaw -= turn;
      } else if (step.type === 'turnRight') {
        yaw += turn;
      }
    });

    setPathPoints(points);

    // Assign modalities cyclically: video, sound, speech
    const mods: string[] = [];
    const cycle = ['video', 'sound', 'speech'];
    for (let i = 0; i < points.length - 1; i++) {
      mods.push(cycle[i % cycle.length]);
    }
    setModalities(mods);
  }, [mode, calibration]);

  // Animation loop for drawing progress
  useEffect(() => {
    if (pathPoints.length < 2) return;
    if (reducedMotion) {
      setDrawProgress(1);
      return;
    }
    let startTime: number | null = null;
    const duration = 3000; // ms
    const step = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / duration, 1);
      setDrawProgress(progress);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    return () => {
      startTime = null;
    };
  }, [pathPoints.length, reducedMotion]);

  // Draw on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Resize canvas to container size
    const container = containerRef.current;
    if (!container) return;
    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth * 0.9;
    const height = container.clientWidth * 0.9; // square
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (pathPoints.length < 2) return;

    // Normalize points to fit canvas with padding
    const xs = pathPoints.map((p) => p.x);
    const zs = pathPoints.map((p) => p.z);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    const pad = 20;
    const widthUsable = width - pad * 2;
    const heightUsable = height - pad * 2;
    const rangeX = maxX - minX || 1;
    const rangeZ = maxZ - minZ || 1;
    const scale = Math.min(widthUsable / rangeX, heightUsable / rangeZ);

    const toCanvasX = (x: number) => pad + (x - minX) * scale;
    const toCanvasZ = (z: number) => pad + (maxZ - z) * scale; // flip Z for screen Y

    const drawUpTo = Math.floor(drawProgress * (pathPoints.length - 1));
    ctx.beginPath();
    ctx.strokeStyle = 'var(--fg)';
    ctx.lineWidth = 2;
    for (let i = 0; i <= drawUpTo; i++) {
      const p = pathPoints[i];
      const cx = toCanvasX(p.x);
      const cz = toCanvasZ(p.z);
      if (i === 0) ctx.moveTo(cx, cz);
      else ctx.lineTo(cx, cz);
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = 'var(--orange)';
    for (let i = 0; i <= drawUpTo; i++) {
      const p = pathPoints[i];
      const cx = toCanvasX(p.x);
      const cz = toCanvasZ(p.z);
      ctx.beginPath();
      ctx.arc(cx, cz, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw start and end markers
    ctx.fillStyle = 'var(--green)';
    const start = pathPoints[0];
    ctx.beginPath();
    ctx.arc(toCanvasX(start.x), toCanvasZ(start.z), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'var(--red)';
    const end = pathPoints[drawUpTo];
    ctx.beginPath();
    ctx.arc(toCanvasX(end.x), toCanvasZ(end.z), 5, 0, Math.PI * 2);
    ctx.fill();
  }, [pathPoints, drawProgress]);

  // What to try block
  const whatToTry = (
    <div className="mt-4 text-sm text-ink">
      <strong>What to try:</strong> Switch between first‑ and third‑person
      modes, move the calibration slider, and observe how the same command
      sequence yields a consistent trajectory length. The timeline below shows
      which modality (video, sound, speech) would be generated at each step.
    </div>
  );

  // Note about toy nature
  const note = (
    <div className="mt-3 text-xs text-mute">
      This is a toy abstraction; the paper’s real measured numbers are left to
      the entry.
    </div>
  );

  return (
    <SchematicCard title="EchoWM Control Unification" className="w-full max-w-xl">
      <div className="space-y-4">
        {/* Controls */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <TechBadge label="Camera Mode">
              {mode === 'first' ? 'First‑Person' : 'Third‑Person'}
            </TechBadge>
            <SchematicButton
              onClick={() => setMode(mode === 'first' ? 'third' : 'first')}
              label="Toggle Mode"
              icon={ToggleLeft}
            />
          </div>
          <div className="flex items-center gap-3">
            <TechBadge label="Motion Calibration">
              {calibration.toFixed(2)}×
            </TechBadge>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.05}
              value={calibration}
              onChange={(e) => setCalibration(parseFloat(e.target.value))}
              className="flex-1 h-1"
            />
          </div>
        </div>

        {/* 3D Viewport */}
        <div className="relative w-full">
          <div ref={containerRef} className="w-full aspect-[1/1] bg-bg1 rounded overflow-hidden">
            <canvas ref={canvasRef} className="absolute inset-0" aria-hidden="true" />
          </div>
        </div>

        {/* Timeline Strip */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-[12px] font-mono text-ink4">
            <TechBadge label="Modality Timeline" />
          </div>
          <div className="flex flex-wrap gap-1">
            {modalities.map((mod, idx) => {
              let color = 'var(--fg2)';
              if (mod === 'video') color = 'var(--green)';
              else if (mod === 'sound') color = 'var(--orange)';
              else if (mod === 'speech') color = 'var(--purple)';
              return (
                <div
                  key={idx}
                  className={`w-2 h-2 rounded bg-${color.replace(
                    'var(--',
                    ''
                  ).replace(')', '')}`}
                />
              );
            })}
          </div>
        </div>

        {whatToTry}
        {note}
      </div>
    </SchematicCard>
  );
};

export default EchoWMSimulation;
