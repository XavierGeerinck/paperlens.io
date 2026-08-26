import React, { useRef, useEffect, useState } from "react";
import { Play, Pause, RotateCcw, Zap } from "lucide-react";
import { SchematicCard, SchematicButton } from "../SketchElements";
import { useSimulation } from "../../../hooks/useSimulation";

// --- CONFIGURATION ---
const TICK_RATE = 50; 
const BASE_COLORS = {
  A: "#34a853", // Green
  G: "#ea4335", // Red
  C: "#4285f4", // Blue
  T: "#f5a623"  // Yellow
};

const PALETTE = {
  bg: "#020617",       // Slate 950
  panelBg: "#0d0f13",  // Slate 900
  border: "#161a20",   // Slate 800
  grid: "#22272f",     // Slate 700
  textMain: "#f8fafc", // Slate 50
  textMuted: "#98a2ae",// Slate 400
  accent: "#4c9ef5",   // Blue 500
  success: "#35d492",  // Emerald 500
  danger: "#f5555d",   // Red 500
  warning: "#f5a623",  // Amber 500
};

// --- DATA GENERATORS ---
const TRACK_NAMES = [
  "Gene expression",
  "Accessibility",
  "Histone mod",
  "TF binding",
  "Splicing"
];

const AlphaGenomeSimulation: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [injectedMutation, setInjectedMutation] = useState<{ active: boolean; tick: number } | null>(null);

  // Simulation State
  const { isRunning, start, stop, reset, epoch: tick } = useSimulation({
    tickRate: TICK_RATE,
    initialState: { pos: 0 },
    onTick: (s) => ({ pos: s.pos + 1 })
  });

  // Derived State for UI & Canvas
  const ticksSinceInj = injectedMutation?.active ? tick - injectedMutation.tick : 0;
  const showVariantDetail = injectedMutation?.active && ticksSinceInj > 60;

  // Main Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    // --- LAYOUT CONSTANTS ---
    const MARGIN = 16;
    const COL_INPUT_W = 120;
    const COL_MODEL_W = 180;
    const COL_OUTPUT_W = w - COL_INPUT_W - COL_MODEL_W - (4 * MARGIN);
    
    // Panel positions
    const inputX = MARGIN;
    const modelX = inputX + COL_INPUT_W + MARGIN;
    const outputX = modelX + COL_MODEL_W + MARGIN;
    
    const panelH = h - (2 * MARGIN);
    const panelY = MARGIN;

    // Clear Screen
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, w, h);

    // --- DRAW UTILITIES ---
    const drawPanel = (x: number, y: number, w: number, h: number, title: string) => {
      // Panel Bg
      ctx.fillStyle = PALETTE.panelBg;
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = PALETTE.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, w, h);
      
      // Header
      ctx.fillStyle = PALETTE.bg;
      ctx.fillRect(x, y, w, 32);
      ctx.strokeStyle = PALETTE.border;
      ctx.beginPath(); ctx.moveTo(x, y+32); ctx.lineTo(x+w, y+32); ctx.stroke();
      
      // Title
      ctx.font = "bold 12px monospace";
      ctx.fillStyle = PALETTE.textMuted;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(title.toUpperCase(), x + w/2, y + 16);
    };

    const drawDashedArrow = (x1: number, y1: number, x2: number, y2: number, active: boolean, packetProgress?: number) => {
        ctx.save();
        ctx.strokeStyle = active ? PALETTE.accent : PALETTE.grid;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        
        // Animate Dash Offset
        if (active) {
            ctx.lineDashOffset = -(tick % 20);
        }

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Arrowhead
        if (active) {
            ctx.setLineDash([]);
            ctx.fillStyle = PALETTE.accent;
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - 6, y2 - 4);
            ctx.lineTo(x2 - 6, y2 + 4);
            ctx.fill();
        }

        // Draw Traveling Data Packet (Simulation of inputs moving)
        if (packetProgress !== undefined && packetProgress >= 0 && packetProgress <= 1) {
             const px = x1 + (x2 - x1) * packetProgress;
             const py = y1 + (y2 - y1) * packetProgress;
             
             ctx.shadowColor = PALETTE.danger;
             ctx.shadowBlur = 10;
             ctx.fillStyle = PALETTE.danger;
             ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI*2); ctx.fill();
             ctx.shadowBlur = 0;
             
             // Label
             ctx.fillStyle = "#fff";
             ctx.font = "bold 10px sans-serif";
             ctx.fillText("MUTATION", px, py - 10);
        }

        ctx.restore();
    };


    // --- 1. INPUT PANEL (DNA SCROLL) ---
    drawPanel(inputX, panelY, COL_INPUT_W, panelH, "Input Sequence");
    
    // Scissor region for DNA
    ctx.save();
    ctx.beginPath();
    ctx.rect(inputX, panelY + 32, COL_INPUT_W, panelH - 32);
    ctx.clip();

    const baseH = 20; // Tighter packing for helix
    const numBases = Math.ceil((panelH - 32) / baseH) + 4;
    const scrollOffset = (tick * 2) % baseH; // Scroll speed
    
    // Calculate current "head" position in genome
    const headGenomePos = Math.floor(tick / 10); 

    const centerX = inputX + COL_INPUT_W / 2;
    const helixRadius = 30;

    for (let i = -2; i < numBases; i++) {
        // Position Index
        const genomeIdx = headGenomePos + i;
        // Deterministic Base
        // A <-> T, G <-> C pairing
        const basePairs = [["A", "T"], ["G", "C"], ["C", "G"], ["T", "A"]];
        const pair = basePairs[Math.floor(Math.abs(Math.sin(genomeIdx * 12.34)) * 4) % 4];
        
        const y = (panelY + panelH) - (i * baseH) + scrollOffset - 20; // Flow Up
        
        // Helix Animation Math
        // Twist factor
        const phase = (y * 0.05) + (tick * 0.05); // Spatial frequency + temporal rotation
        const xOffset = Math.sin(phase) * helixRadius;
        const zDepth = Math.cos(phase); // For layering (pseudo-3D)

        // Strand 1 Position (Left/Front/Back)
        const x1 = centerX - xOffset;
        const x2 = centerX + xOffset; // Strand 2 is opposite

        // Mutation Check
        let isMutated = false;
        if (injectedMutation?.active) {
            const mutStartIdx = Math.floor(injectedMutation.tick / 10);
            if (Math.abs(genomeIdx - mutStartIdx - 20) < 2) { 
                isMutated = true;
            }
        }

        const fontSize = 10 + (zDepth + 1) * 2; // Size based on "depth" 0..2 -> 10..14px
        ctx.font = `bold ${Math.floor(fontSize)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        // Draw Connector (Hydrogen Bond)
        if (!isMutated) {
            ctx.strokeStyle = `rgba(148, 163, 184, ${0.3 + (zDepth + 1) * 0.2})`; // Fade based on depth
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x1, y);
            ctx.lineTo(x2, y);
            ctx.stroke();
        }

             // Draw Base & Strands
             
             // Draw Region Markers (Enhancer, Promoter)
             // Let's place them periodically
             // genomeIdx 100-120: Enhancer
             // genomeIdx 200-240: Promoter
             
             const regionType = (gIdx: number) => {
                 const mod = gIdx % 300;
                 if (mod > 50 && mod < 80) return "ENHANCER";
                 if (mod > 150 && mod < 180) return "PROMOTER";
                 return null;
             }
             
             const rType = regionType(genomeIdx);
             if (rType) {
                 ctx.save();
                 ctx.translate(inputX + 10, y);
                 ctx.rotate(-Math.PI/2);
                 ctx.fillStyle = rType === "ENHANCER" ? PALETTE.warning : PALETTE.success;
                 ctx.font = "bold 9px sans-serif";
                 ctx.fillText(rType, 0, 0);
                 ctx.restore();
                 
                 // Highlight background behind helix
                 ctx.fillStyle = rType === "ENHANCER" ? "rgba(245, 158, 11, 0.1)" : "rgba(16, 185, 129, 0.1)";
                 ctx.fillRect(centerX - 40, y - 5, 80, 10);
             }

        const drawBase = (x: number, char: string) => {
            const color = BASE_COLORS[char as keyof typeof BASE_COLORS];
            
            // Halo/Glow
            if (isMutated) {
                 ctx.shadowColor = PALETTE.danger;
                 ctx.shadowBlur = 10;
                 ctx.fillStyle = PALETTE.danger;
                 ctx.fillText("X", x, y);
                 ctx.shadowBlur = 0;
            } else {
                 // Background Circle
                 ctx.fillStyle = PALETTE.panelBg;
                 ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI*2); ctx.fill();
                 
                 // Text
                 ctx.fillStyle = color;
                 ctx.fillText(char, x, y);
            }
        };

        // Order rendering by Z-index? 
        // The simple way: connector is always behind. Bases are on top.
        // x1 is Strand 1. x2 is Strand 2.
        
        drawBase(x1, pair[0]);
        drawBase(x2, pair[1]);
        
        if (isMutated) {
             // Red Overlay for the whole row
             ctx.fillStyle = "rgba(239, 68, 68, 0.1)";
             ctx.fillRect(inputX, y - 10, COL_INPUT_W, 20);
        }
    }
    ctx.restore();


    // --- 2. MODEL PANEL (Processing) ---
    // Architecture: Conv -> Transformer -> Heads
    
    const nodeW = 160;
    const nodeX = modelX + (COL_MODEL_W - nodeW)/2;
    const nodeCenterY = h/2;
    
    // Packet Math
    // Let's say travel time Input->Model is 40 ticks.
    let packet1 = -1;
    let packet2 = -1;
    
    if (injectedMutation?.active) {
         const t = tick - injectedMutation.tick;
         // Phase 1: Input to Node (Context Window)
         if (t >= 0 && t < 40) packet1 = t / 40;
         // Phase 2: Processing (Inside Node) - Handled by opacity/color changes in node
         // Phase 3: Node to Output
         if (t > 60 && t < 100) packet2 = (t - 60) / 40;
    }

    // Draw Connector Lines from Input
    drawDashedArrow(inputX + COL_INPUT_W, h/2, nodeX, h/2, isRunning, packet1 !== -1 ? packet1 : undefined);

    // 1. Convolutional Tower (Motif Extraction)
    const convH = 120;
    const convW = 30;
    const convX = nodeX;
    
    ctx.fillStyle = PALETTE.panelBg;
    ctx.strokeStyle = PALETTE.grid;
    ctx.lineWidth = 1;
    ctx.fillRect(convX, nodeCenterY - convH/2, convW, convH);
    ctx.strokeRect(convX, nodeCenterY - convH/2, convW, convH);
    
    // Animate "scanning" blocks in Conv layer
    if (isRunning) {
        const scanY = nodeCenterY - convH/2 + (tick * 2) % convH;
        ctx.fillStyle = PALETTE.success;
        ctx.globalAlpha = 0.5;
        ctx.fillRect(convX, scanY, convW, 10);
        ctx.globalAlpha = 1.0;
    }
    ctx.fillStyle = PALETTE.textMuted; ctx.font = "9px sans-serif"; ctx.textAlign = "center"; 
    ctx.save(); ctx.translate(convX + 15, nodeCenterY); ctx.rotate(-Math.PI/2); ctx.fillText("CONV TOWERS", 0, 0); ctx.restore();

    // 2. Transformer Core (Attention Matrix)
    const txW = 80;
    const txH = 120;
    const txX = convX + convW + 15;
    
    // Draw Matrix Grid
    ctx.save();
    ctx.translate(txX, nodeCenterY - txH/2);
    
    // Background
    ctx.fillStyle = PALETTE.panelBg;
    ctx.fillRect(0,0, txW, txH);
    ctx.strokeStyle = PALETTE.accent;
    ctx.strokeRect(0,0, txW, txH);
    
    // Cells
    const cellS = 10;
    const cols = txW / cellS;
    const rows = txH / cellS;
    
    for(let r=0; r<rows; r++) {
        for(let c=0; c<cols; c++) {
             // Twinkle effect simulating attention patterns
             const active = Math.sin(tick * 0.1 + r * 0.5 + c) > 0.8;
             if (active) {
                 ctx.fillStyle = `rgba(59, 130, 246, ${Math.random()})`;
                 ctx.fillRect(c*cellS, r*cellS, cellS-1, cellS-1);
             } else {
                 ctx.fillStyle = "rgba(30, 41, 59, 0.5)";
                 ctx.fillRect(c*cellS, r*cellS, cellS-1, cellS-1);
             }
        }
    }
    ctx.restore();
    
    // Label
    ctx.fillStyle = PALETTE.textMain; ctx.fillText("TRANSFORMER", txX + txW/2, nodeCenterY + txH/2 + 15);

    // 3. Multimodal Heads (Fan Out)
    const fanStartX = txX + txW;
    const fanPoints = [
        { y: nodeCenterY - 40, color: PALETTE.accent, label: "Expr" },
        { y: nodeCenterY, color: PALETTE.warning, label: "Access" },
        { y: nodeCenterY + 40, color: PALETTE.danger, label: "3D" }
    ];
    
    fanPoints.forEach(pt => {
        // Line from Transformer to Head Node
        ctx.beginPath();
        ctx.moveTo(fanStartX, nodeCenterY);
        ctx.lineTo(fanStartX + 20, pt.y);
        ctx.strokeStyle = pt.color;
        ctx.stroke();
        
        // Head Node
        ctx.beginPath(); ctx.arc(fanStartX + 25, pt.y, 6, 0, Math.PI*2); 
        ctx.fillStyle = pt.color; ctx.fill();
        
        // Output beam
        drawDashedArrow(fanStartX + 30, pt.y, outputX, pt.y, isRunning, packet2 !== -1 ? packet2 : undefined);
    });


    // --- 3. OUTPUT PANEL (Tracks) ---
    // Redesign: Genome Browser Style
    
    drawPanel(outputX, panelY, COL_OUTPUT_W, panelH, "PREDICTED REGULATORY LANDSCAPE");
    
    // Calculate mutation impact
    // If mutation passed the model (approx 50 ticks after injection), output crashes
    
    // Tracks Area
    const tracksH = (panelH - 60) * 0.6;
    const heatmapH = (panelH - 60) * 0.4;
    const trackYStart = panelY + 40;
    const singleTrackH = tracksH / 5;
    
    const trackContentX = outputX + 100; // Left margin for labels

    const trackContentW = COL_OUTPUT_W - 110;

    // Helper: Pseudo-random feature generator based on coordinate
    // Returns signal height 0..1 at relative genome position
    const getTrackSignal = (trackIdx: number, gPos: number) => {
        // Deterministic noise
        const noise = Math.abs(Math.sin(gPos * 0.13 + trackIdx * 7.1) + Math.sin(gPos * 0.07)) * 0.1;
        
        let signal = noise;
        
        // Define Features via mathematical hotspots
        // Feature = defined by position period
        
        if (trackIdx === 0) { // CAGE / Expression (Broad Mountains)
            // Gene bodies every ~200 ticks
            const period = 200;
            const phase = gPos % period;
            if (phase < 60) { // Active gene
                signal = 0.8 + (Math.sin(gPos * 0.2) * 0.1); // Noisy plateau
            }
        } 
        else if (trackIdx === 1) { // ATAC / Accessibility (Sharp Peaks)
            // Peaks at TSS (start of gene) and Enhancers
            const period = 200;
            const phase = gPos % period;
            // Promoter peak
            if (Math.abs(phase - 5) < 10) signal = 1.0; 
            // Distal enhancer peak
            if (Math.abs(phase - 120) < 6) signal = 0.7;
        }
        else if (trackIdx === 2) { // Histone (Promoter marks)
            // Broader peak at TSS
            const period = 200;
            const phase = gPos % period;
            if (Math.abs(phase - 5) < 15) signal = 0.9;
        }
        else if (trackIdx === 3) { // CTCF (Insulator spikes)
             // Every 300 ticks
             if (gPos % 300 < 5) signal = 1.0;
        }
        else if (trackIdx === 4) { // Splicing (Delta spikes)
             // Inside genes
             const period = 200;
             const phase = gPos % period;
             if (phase < 60 && phase % 15 < 3) signal = 0.6; // Exon boundaries
        }
        
        // Impact Logic
        if (injectedMutation?.active) {
             // Say mutation is at specific tick. 
             // If we represent "gPos" as time, larger gPos = NEWER = Right side of screen
             // Smaller gPos = OLDER = Left side
             // If mutation happened at T_mut, then region T_mut is affected.
             // We are rendering gPos.
             
             // Let's say mutation affects a 100-tick window around it? 
             // Or permanently breaks downstream?
             // Paper says "Variant Effect" -> Localized change.
             
             // Let's define the mutation site specifically
             const mutationSite = injectedMutation.tick;
             
             // If we are drawing the region of the mutation
             if (Math.abs(gPos - mutationSite) < 30) {
                 if (trackIdx === 0 || trackIdx === 1 || trackIdx === 3) {
                     signal *= 0.1; // Local collapse
                 }
             }
        }
        
        return Math.max(0, Math.min(1, signal));
    };

    TRACK_NAMES.forEach((name, idx) => {
        const y = trackYStart + idx * singleTrackH;
        
        // Background Strip
        ctx.fillStyle = idx % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent";
        ctx.fillRect(trackContentX, y, trackContentW, singleTrackH);
        
        // Y-axis line
        ctx.strokeStyle = PALETTE.grid;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(trackContentX, y); ctx.lineTo(trackContentX, y+singleTrackH); ctx.stroke();

        // Label
        ctx.fillStyle = PALETTE.textMuted;
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(name.toUpperCase(), outputX + 90, y + singleTrackH/2 + 3);
        
        // Draw Data
        ctx.beginPath();
        const pts = 200; // Higher fidelity
        const step = trackContentW / pts;
        
        ctx.moveTo(trackContentX, y + singleTrackH); // Start bottom-left

        for(let i=0; i<=pts; i++) {
             const px = trackContentX + i*step;
             
             // Calculate Genomic Coordinate for this pixel
             // Screen Right = Current Tick (Newest)
             // Screen Left = Current Tick - range
             const range = 200; // How many ticks fit on screen
             const mapTime = tick - (range) + (i/pts)*range; 
             
             const signal = getTrackSignal(idx, Math.floor(mapTime));
             const py = (y + singleTrackH) - (signal * (singleTrackH - 4));
             ctx.lineTo(px, py);
        }
        
        ctx.lineTo(trackContentX + trackContentW, y + singleTrackH); // Bottom right
        ctx.lineTo(trackContentX, y + singleTrackH); // Close
        
        // Fill
        const trackColors = [
            "#4c9ef5", // Blue (Expr)
            "#35d492", // Green (Access)
            "#f5a623", // Amber (Histone)
            "#7452e6", // Purple (CTCF)
            "#c257db"  // Pink (Splicing)
        ];
        
        ctx.fillStyle = trackColors[idx % trackColors.length];
        ctx.globalAlpha = 0.5;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = trackColors[idx % trackColors.length];
        ctx.stroke();
    });

    // Heatmap (Bottom)
    const heatY = trackYStart + tracksH + 20;
    
    ctx.fillStyle = PALETTE.textMuted;
    ctx.textAlign = "right";
    ctx.fillText("HI-C CONTACTS", outputX + 90, heatY + heatmapH/2);
    
    // Draw Heatmap: Triangular / Diamond Pattern
    const heatW = trackContentW;
    const cellSize = 5;
    const hCols = Math.ceil(heatW / cellSize);
    const hRows = Math.ceil(heatmapH / cellSize);
    
    // Impact Marker Arrow
    if (injectedMutation?.active) {
         // Where is the mutation roughly on screen?
         // mapTime = tick - (range) + (i/pts)*range
         // We want mapTime == mutation.tick
         // i/pts = (mutation.tick - (tick - range)) / range
         // i/pts = (mutation.tick - tick + 200) / 200
         
         const relPos = (injectedMutation.tick - tick + 200) / 200;
         if (relPos >= 0 && relPos <= 1) {
             const markerX = trackContentX + relPos * trackContentW;
             
             // Draw vertical line through tracks
             ctx.save();
             ctx.strokeStyle = "rgba(239, 68, 68, 0.5)";
             ctx.setLineDash([4, 4]);
             ctx.beginPath();
             ctx.moveTo(markerX, trackYStart);
             ctx.lineTo(markerX, heatY + heatmapH);
             ctx.stroke();
             
             // Draw Label "SNV"
             ctx.fillStyle = PALETTE.danger;
             ctx.font = "bold 9px sans-serif";
             ctx.fillText("SNV", markerX - 10, trackYStart - 5);
             ctx.restore();
         }
    }
    
    // Mask logic
    ctx.save();
    ctx.beginPath();
    ctx.rect(trackContentX, heatY, heatW, heatmapH);
    ctx.clip();
    
    for(let r=0; r<hRows; r++) {
        for(let c=0; c<hCols; c++) {
            // Hi-C logic: intensity decays with distance from diagonal
            // Here 'r' is distance from main diagonal effectively if we view it as rotated
            // Let's just draw scrolling diamonds
            
            // Map grid to time
            const gridTime = tick - (hCols - c);
            
            // Pattern: Contact domains (TADs) are triangles on the diagonal
            // distance from "diagonal" (bottom edge)
            
            const cellY = heatY + heatmapH - (r * cellSize);
            const cellX = trackContentX + c * cellSize;
            
            // Logic: High contact if (gridTime / TAD_SIZE) is integer
            const tadSize = 30;
            const inTad = Math.floor(gridTime / tadSize) % 2 === 0;
            
            let intensity = 0;
            if (inTad) {
                // Inside a TAD, higher contact prob
                // Decay with 'r' (distance from diagonal)
                intensity = Math.max(0, 1.0 - (r / 20)); 
            } else {
                // Background noise
                intensity = Math.random() * 0.1;
            }
            
            // Impact Check
            if (injectedMutation?.active) {
                const impactTime = injectedMutation.tick + 100;
                if (gridTime > impactTime) {
                    intensity *= 0.2; // Domain disruption
                }
            }

            const colorVal = Math.floor(intensity * 255);
            ctx.fillStyle = `rgb(${colorVal}, 0, 0)`; // Red heatmap
            ctx.fillRect(cellX, cellY - cellSize, cellSize, cellSize);
        }
    }
    ctx.restore();

    // Impact Curtain on Heatmap (Removed, redundant with real-time update)


    // --- 4. DETAIL VIEW OVERLAY ---
    // Drawn on top of Output Panel when mutation is detected
    if (showVariantDetail) {
        const detailW = 320;
        const detailH = 180;
        // Position: Bottom Right, over the tracks/heatmap
        const detailX = outputX + COL_OUTPUT_W - detailW - 20;
        const detailY = panelY + panelH - detailH - 20;
        
        // Box
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 20;
        ctx.fillStyle = PALETTE.panelBg;
        ctx.fillRect(detailX, detailY, detailW, detailH);
        ctx.shadowBlur = 0;
        ctx.strokeStyle = PALETTE.accent;
        ctx.strokeRect(detailX, detailY, detailW, detailH);
        
        // Header
        ctx.fillStyle = PALETTE.accent;
        ctx.fillRect(detailX, detailY, detailW, 28);
        ctx.fillStyle = "#fff";
        ctx.textAlign = "left";
        ctx.font = "bold 12px sans-serif";
        ctx.fillText("VARIANT ANALYSIS: GENE EXPRESSION", detailX + 10, detailY + 14);
        // Explanation
        ctx.fillStyle = PALETTE.textMuted;
        ctx.font = "10px sans-serif";
        ctx.fillText("Compare Reference (REF) vs Mutated (ALT) prediction.", detailX + 10, detailY + detailH - 8);

        // Legend
        const legX = detailX + detailW - 80;
        const legY = detailY + 40;
        // Ref
        ctx.strokeStyle = PALETTE.danger; ctx.beginPath(); ctx.moveTo(legX, legY); ctx.lineTo(legX+15, legY); ctx.stroke();
        ctx.fillStyle = PALETTE.textMuted; ctx.fillText("REF (Norm)", legX+20, legY+3);
        // Alt
        ctx.strokeStyle = PALETTE.warning; ctx.beginPath(); ctx.moveTo(legX, legY+15); ctx.lineTo(legX+15, legY+15); ctx.stroke();
        ctx.fillText("ALT (Mut)", legX+20, legY+18);
        
        // Chart Area
        const chX = detailX + 40;
        const chY = detailY + 40;
        const chW = detailW - 100;
        const chH = detailH - 60;
        
        // Axes
        ctx.strokeStyle = PALETTE.grid;
        ctx.beginPath();
        ctx.moveTo(chX, chY); ctx.lineTo(chX, chY+chH); ctx.lineTo(chX+chW, chY+chH);
        ctx.stroke();
        
        // Draw REF (Red, High)
        ctx.strokeStyle = PALETTE.danger;
        ctx.beginPath();
        for(let i=0; i<=chW; i+=5) {
             const t = i/chW;
             let val = Math.exp(-Math.pow(t - 0.5, 2) * 50); // Gaussian peak
             // Deterministic noise
             val += (Math.sin(i * 0.8) * Math.cos(i * 1.5)) * 0.05;
             const y = (chY + chH) - (val * chH * 0.8);
             if(i===0) ctx.moveTo(chX+i, y); else ctx.lineTo(chX+i, y);
        }
        ctx.stroke();

        // Draw ALT (Yellow, Low/Flat)
        ctx.strokeStyle = PALETTE.warning;
        ctx.beginPath();
        for(let i=0; i<=chW; i+=5) {
             const t = i/chW;
             let val = Math.exp(-Math.pow(t - 0.5, 2) * 50) * 0.1; // Suppressed
             // Less noise for suppressed signal
             val += (Math.sin(i * 0.9 + 2) * Math.cos(i * 1.2)) * 0.02;
             const y = (chY + chH) - (val * chH * 0.8);
             if(i===0) ctx.moveTo(chX+i, y); else ctx.lineTo(chX+i, y);
        }
        ctx.stroke();
    }

  }, [tick, injectedMutation, isRunning, showVariantDetail]);

  // Handler
  const handleInject = () => {
       setInjectedMutation({ active: true, tick: tick });
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-4">
      <SchematicCard title="ALPHAGENOME WORKBENCH">
        <canvas 
          ref={canvasRef} 
          width={1000} 
          height={500}
          className="w-full bg-slate-950 rounded-lg shadow-2xl border border-slate-900"
        />
        
        {/* Controls Overlay */}
        <div className="flex items-center justify-between p-4 bg-slate-900 border-t border-slate-800">
             <div className="flex space-x-2">
                <SchematicButton icon={isRunning ? Pause : Play} label={isRunning ? "PAUSE" : "RUN"} onClick={isRunning ? stop : start} active={isRunning} />
                <SchematicButton icon={RotateCcw} label="RESET" onClick={() => { reset(); setInjectedMutation(null); }} />
             </div>
             
             <div className="flex items-center space-x-4">
                 <div className="text-right">
                     <div className="text-xs text-slate-500 uppercase tracking-wider">Status</div>
                     <div className={injectedMutation?.active ? "text-red-400 font-bold" : "text-emerald-400 font-bold"}>
                         {injectedMutation?.active ? (showVariantDetail ? "IMPACT ANALYSIS" : "PROCESSING VARIANT...") : "SYSTEM NOMINAL"}
                     </div>
                 </div>
                 <SchematicButton 
                    icon={Zap} 
                    label="INJECT VARIANT" 
                    variant="danger" 
                    onClick={handleInject} 
                    disabled={!!injectedMutation}
                 />
             </div>
        </div>

        {/* Legend / Guide */}
        <div className="p-5 bg-slate-950/50 border-t border-slate-800 text-xs text-slate-400 mt-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="text-white font-bold mb-2 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span> 
                INPUT & MODEL
              </h4>
              <p className="leading-relaxed">
                The <span className="text-slate-300">Conformer Architecture</span> (left) processes raw DNA sequence (1 hot encoded). 
                The 3D helix visualizes the input stream.
              </p>
            </div>
            <div>
              <h4 className="text-white font-bold mb-2 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                PREDICTION TRACKS
              </h4>
              <p className="leading-relaxed">
                The model outputs high-resolution genomic tracks (right). 
                <span className="text-blue-400"> Blue</span> = Gene Expression, 
                <span className="text-emerald-400"> Green</span> = Chromatin Accessibility.
              </p>
            </div>
            <div>
               <h4 className="text-white font-bold mb-2 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                VARIANT EFFECT
              </h4>
              <p className="leading-relaxed">
                Click <strong className="text-rose-400">INJECT VARIANT</strong> to simulate a Non-Coding mutation. 
                Watch the "Red Packet" propagate and cause a <span className="text-slate-300">local collapse</span> in the predicted expression signals.
              </p>
            </div>
          </div>
        </div>
      </SchematicCard>
    </div>
  );
};

export default AlphaGenomeSimulation;
