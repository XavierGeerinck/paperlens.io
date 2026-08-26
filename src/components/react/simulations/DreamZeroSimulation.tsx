import type { FC } from "react";
import React, { useRef, useEffect, useState } from "react";
import {
    Brain,
    Eye
} from "lucide-react";
import { SchematicCard } from "../SketchElements";

interface SimulationState {
    environment: 'training' | 'ood';
    steeringAngle: number; // -45 to 45
}

const DreamZeroSimulation: FC = () => {
    const leftCanvasRef = useRef<HTMLCanvasElement>(null);
    const rightCanvasRef = useRef<HTMLCanvasElement>(null);
    const [state, setState] = useState<SimulationState>({
        environment: 'ood',
        steeringAngle: 0
    });

    // Robot Constants
    const ARM_LENGTH = 140;
    const BASE_X = 200; // Center of 400px canvas
    const BASE_Y = 280; // Bottom

    // Helper: Draw the Lab Environment
    const drawLab = (ctx: CanvasRenderingContext2D, width: number, height: number, env: 'training' | 'ood') => {
        // Background
        ctx.fillStyle = "#0d0f13";
        ctx.fillRect(0,0,width,height);

        // Grid Lines (Technical Look)
        ctx.strokeStyle = "#161a20";
        ctx.lineWidth = 1;
        for(let i=0; i<width; i+=40) {
            ctx.beginPath(); ctx.moveTo(i,0); ctx.lineTo(i,height); ctx.stroke();
        }
        for(let i=0; i<height; i+=40) {
            ctx.beginPath(); ctx.moveTo(0,i); ctx.lineTo(width,i); ctx.stroke();
        }

        // Draw Base
        ctx.fillStyle = "#323944";
        ctx.beginPath();
        ctx.arc(BASE_X, BASE_Y, 20, Math.PI, 0); // Semicircle base
        ctx.fill();
        ctx.fillStyle = "#98a2ae";
        ctx.fillRect(BASE_X - 15, BASE_Y, 30, 10);

        if (env === 'ood') {
            // OOD: A "Fragile Vase" or Obstacle appears in the left quadrant
            // Position: ~ -30 degrees area
            const obsX = BASE_X - 60;
            const obsY = BASE_Y - 100;
            
            // Draw Obstacle (Red Glass Block)
            ctx.fillStyle = "rgba(239, 68, 68, 0.2)"; // faint red fill
            ctx.strokeStyle = "#f5555d";
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.rect(obsX - 25, obsY - 30, 50, 60);
            ctx.fill();
            ctx.stroke();
            
            // Label
            ctx.fillStyle = "#f5555d";
            ctx.font = "10px monospace";
            ctx.textAlign = "center";
            ctx.fillText("UNKNOWN OBSTACLE", obsX, obsY + 45);
        }
    };

    // Helper: Draw Robot Arm
    const drawRobot = (
        ctx: CanvasRenderingContext2D, 
        angleDeg: number, 
        isGhost: boolean,
        collision: boolean
    ) => {
        const rad = (angleDeg - 90) * (Math.PI / 180); // 0deg is right in canvas, so -90 makes 0 up
        
        const endX = BASE_X + Math.cos(rad) * ARM_LENGTH;
        const endY = BASE_Y + Math.sin(rad) * ARM_LENGTH;

        // Arm Link
        ctx.strokeStyle = isGhost ? (collision ? "#e13540" : "#35d492") : "#4c9ef5";
        ctx.lineWidth = isGhost ? 4 : 8;
        if(isGhost) ctx.setLineDash([5,5]);
        
        ctx.beginPath();
        ctx.moveTo(BASE_X, BASE_Y);
        
        if (collision && isGhost) {
            // If it's a crash prediction, draw only part way or show "impact"
            const impactX = BASE_X + Math.cos(rad) * (ARM_LENGTH * 0.7);
            const impactY = BASE_Y + Math.sin(rad) * (ARM_LENGTH * 0.7);
            ctx.lineTo(impactX, impactY);
            ctx.stroke();
            
            // Explosion
            ctx.fillStyle = "#e13540";
            ctx.beginPath();
            ctx.arc(impactX, impactY, 10, 0, Math.PI*2);
            ctx.fill();
            ctx.fillStyle = "#fff";
            ctx.font = "bold 14px sans-serif";
            ctx.fillText("💥", impactX-7, impactY+5);
        } else {
            ctx.lineTo(endX, endY);
            ctx.stroke();

            // Gripper/End Effector
            ctx.save();
            ctx.translate(endX, endY);
            ctx.rotate(rad + Math.PI/2);
            ctx.fillStyle = isGhost ? (collision ? "#e13540" : "#35d492") : "#d5dbe3";
            ctx.fillRect(-10, -10, 20, 20); // Gripper box
            // Fingers
            ctx.fillRect(-15, -15, 5, 15);
            ctx.fillRect(10, -15, 5, 15);
            ctx.restore();
        }
        
        ctx.setLineDash([]);
    };

    // Render Loop
    useEffect(() => {
        // --- LOGIC ---
        // Angle 0 is straight UP. 
        // Obstacle is roughly at -30 deg (Left).
        // Let's define collision zone: -40 to -20 deg.
        // We'll calculate collision once for logic
        const hitZoneStart = -45;
        const hitZoneEnd = -15;
        const isHit = state.environment === 'ood' && (state.steeringAngle > hitZoneStart && state.steeringAngle < hitZoneEnd);

        // --- LEFT CANVAS (Standard) ---
        if (leftCanvasRef.current) {
            const ctx = leftCanvasRef.current.getContext('2d');
            if (ctx) {
                // 1. Lab
                drawLab(ctx, 400, 300, state.environment);
                // 2. Ghost Prediction (Standard Model)
                // Standard model ignores OOD obstacle, predicts movement succeeds
                // It draws the "Ghost" at the target location (angle)
                drawRobot(ctx, state.steeringAngle, true, false); // ghost, no collision
                // 3. Real Robot (Blue) - usually trails behind or matches ghost in sim
                // We just show ghost = prediction for now
            }
        }

        // --- RIGHT CANVAS (DreamZero) ---
        if (rightCanvasRef.current) {
            const ctx = rightCanvasRef.current.getContext('2d');
            if (ctx) {
                // 1. Lab
                drawLab(ctx, 400, 300, state.environment);
                // 2. Ghost Prediction (DreamZero)
                // If hit, predicts crash
                drawRobot(ctx, state.steeringAngle, true, isHit);
            }
        }

    }, [state]);

    // Check collision state for UI feedback
    // Logic matches simulation above
    const hitZoneStart = -45;
    const hitZoneEnd = -15;
    const isCrash = state.environment === 'ood' && (state.steeringAngle > hitZoneStart && state.steeringAngle < hitZoneEnd);

    return (
        <div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
            <SchematicCard title="ROBOTIC_PRECOGNITION_UNIT">
                <div className="flex flex-col gap-6">
                    {/* CONTROLS */}
                    <div className="flex flex-col md:flex-row items-center justify-between gap-4 bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                         {/* Toggle */}
                         <div className="flex bg-slate-900 rounded p-1 border border-slate-700 shrink-0">
                            <button
                                type="button"
                                onClick={() => setState(s => ({...s, environment: 'training'}))}
                                className={`px-4 py-2 text-xs rounded font-mono transition-colors ${
                                    state.environment === 'training' 
                                        ? 'bg-emerald-900/50 text-emerald-400 font-bold border border-emerald-900' 
                                        : 'text-slate-500 hover:text-slate-300'
                                }`}
                            >
                                FACTORY (CLEAN)
                            </button>
                            <button
                                type="button"
                                onClick={() => setState(s => ({...s, environment: 'ood'}))}
                                className={`px-4 py-2 text-xs rounded font-mono transition-colors ${
                                    state.environment === 'ood' 
                                        ? 'bg-rose-900/50 text-rose-400 font-bold border border-rose-900' 
                                        : 'text-slate-500 hover:text-slate-300'
                                }`}
                            >
                                LAB (CLUTTERED)
                            </button>
                        </div>

                        {/* Slider */}
                        <div className="flex-1 w-full max-w-lg flex items-center gap-4">
                            <div className="text-[10px] font-mono whitespace-nowrap text-slate-500">ROTATE LEFT</div>
                            <input
                                type="range"
                                min={-70}
                                max={70}
                                step={1}
                                value={state.steeringAngle}
                                onChange={(e) => setState(s => ({...s, steeringAngle: Number(e.target.value)}))}
                                className="w-full accent-blue-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="text-[10px] font-mono whitespace-nowrap text-slate-500">ROTATE RIGHT</div>
                        </div>
                    </div>

                    {/* SIDE BY SIDE VIEWS */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        
                        {/* LEFT: STANDARD */}
                        <div className="flex flex-col gap-2">
                             <div className="flex items-center justify-between px-2">
                                <div className="flex items-center gap-2 text-slate-400 text-xs font-bold uppercase">
                                    <Brain size={14} /> Standard Helper
                                </div>
                                <div className="bg-slate-800 text-slate-400 text-[10px] px-2 py-0.5 rounded font-mono border border-slate-600">
                                    PREDICTION: OK ✅
                                </div>
                             </div>
                             
                             <div className="relative aspect-[4/3] bg-black rounded-lg border border-slate-700 overflow-hidden">
                                <canvas 
                                    ref={leftCanvasRef}
                                    width={400} height={300}
                                    className="w-full h-full object-contain"
                                />
                                {state.environment === 'ood' && (
                                    <div className="absolute top-2 left-2 right-2 bg-rose-950/80 border border-rose-900 p-2 rounded text-[10px] text-rose-200 backdrop-blur-md">
                                        ⚠️ BLIND ERROR: Model predicts arm can move through the obstacle freely.
                                    </div>
                                )}
                             </div>
                        </div>

                        {/* RIGHT: DREAMZERO */}
                        <div className="flex flex-col gap-2">
                             <div className="flex items-center justify-between px-2">
                                <div className="flex items-center gap-2 text-emerald-400 text-xs font-bold uppercase">
                                    <Eye size={14} /> DreamZero
                                </div>
                                <div className={`text-[10px] px-2 py-0.5 rounded font-mono border transition-colors ${
                                    isCrash 
                                        ? 'bg-rose-900/50 text-rose-400 border-rose-500 animate-pulse' 
                                        : 'bg-emerald-900/50 text-emerald-400 border-emerald-500'
                                }`}>
                                    PREDICTION: {isCrash ? 'COLLISION 💥' : 'CLEAR ✅'}
                                </div>
                             </div>
                             
                             <div className={`relative aspect-[4/3] bg-black rounded-lg border overflow-hidden transition-colors ${
                                 isCrash ? 'border-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.3)]' : 'border-emerald-500/50'
                             }`}>
                                <canvas 
                                    ref={rightCanvasRef}
                                    width={400} height={300}
                                    className="w-full h-full object-contain"
                                />
                                {isCrash && (
                                    <div className="absolute top-2 left-2 right-2 bg-emerald-950/80 border border-emerald-900 p-2 rounded text-[10px] text-emerald-200 backdrop-blur-md">
                                        ✨ PRECOGNITION: The model hallucinates the impact and halts execution.
                                    </div>
                                )}
                             </div>
                        </div>

                    </div>
                </div>
            </SchematicCard>
        </div>
    );
};

export default DreamZeroSimulation;
