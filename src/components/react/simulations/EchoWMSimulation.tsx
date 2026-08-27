import React, { useState } from 'react';
import { LabCard, TechBadge, SchematicButton, DataReadout, SchematicCard } from '../SketchElements';
import { SketchArrowRight } from '../SketchElements';
import { ArrowRight } from 'lucide-react';

function mulberry32(seed) {
  let t = seed >>> 0;
  return function() {
    t = Math.imul(t *= 0x9e3779b9, 0xffffffff);
    let r = t ^ (t >>> 16);
    return (r >>> 0) / 4294967296;
  };
}

const seed = 42;
const rng = mulberry32(seed);

const init = () => (rng() - 0.5) * 2; // range -1 to 1

export default function EchoWMSimulation() {
  const [forward, setForward] = useState(init);
  const [left, setLeft] = useState(init);
  const [up, setUp] = useState(init);
  const [yaw, setYaw] = useState(init);
  const [pitch, setPitch] = useState(init);
  const [roll, setRoll] = useState(init);

  const prefersReduced = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return (
    <LabCard title="EchoWM Simulation" status="toy" className="overflow-x-hidden">
      <div className="space-y-4">
        {/* Controls */}
        <div className="flex flex-col gap-2">
          {['forward', 'left', 'up', 'yaw', 'pitch', 'roll'].map((axis) => (
            <div key={axis} className="flex items-center gap-2">
              <TechBadge label={axis} color="text-mint-400" />
              <input
                type="range"
                min="-1"
                max="1"
                step="0.01"
                value={axis === 'forward' ? forward : axis === 'left' ? left : axis === 'up' ? up : axis === 'yaw' ? yaw : axis === 'pitch' ? pitch : roll}
                onChange={e => {
                  const val = parseFloat(e.target.value);
                  if (axis === 'forward') setForward(val);
                  else if (axis === 'left') setLeft(val);
                  else if (axis === 'up') setUp(val);
                  else if (axis === 'yaw') setYaw(val);
                  else if (axis === 'pitch') setPitch(val);
                  else if (axis === 'roll') setRoll(val);
                }}
                className="w-full"
              />
              <DataReadout
                label="Value"
                value={`${axis === 'forward' ? forward : axis === 'left' ? left : axis === 'up' ? up : axis === 'yaw' ? yaw : axis === 'pitch' ? pitch : roll}.toFixed(2)`}
              />
            </div>
          ))}
        </div>

        {/* Camera icon */}
        <div className="flex items-center gap-4">
          <TechBadge label="Camera" color="text-amber-400" />
          <div className="inline-flex items-center gap-2">
            <div
              className="rounded-full bg-mint-400"
              style={{ transform: `translate(${forward * 10}px, ${left * 10}px, ${up * 10}px) rotateX(${pitch * 10}deg) rotateY(${yaw * 10}deg) rotateZ(${roll * 10}deg)` }}
            />
            <ArrowRight className="text-mint-400" />
          </div>
        </div>

        {/* What to try */}
        <div className="text-sm text-ink text-center mt-4">
          <strong>What to try:</strong> Adjust the sliders to change camera intent. Notice how the camera icon moves and the waveform (if you add one) changes smoothly, illustrating continuous multimodal generation.
        </div>

        {/* Note */}
        <div className="text-xs text-mute mt-2">
          Note: This is a toy abstraction; the paper's real measured numbers are left to the entry.
        </div>
      </div>
    </LabCard>
  );
}
