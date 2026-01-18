import { useEffect, useRef, useState } from "react";

const CELL_SIZE = 10;
const SIM_FPS = 30;
const OVERCLOCK_FPS = 60;

const KONAMI_CODE = ["ArrowUp", "ArrowUp", "ArrowDown", "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowRight", "b", "a"];

const GameOfLife = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>(0);
  const lastSimTime = useRef<number>(0);
  const stateRef = useRef<Float32Array | null>(null);

  const [isOverclocked, setIsOverclocked] = useState(false);
  const keySequence = useRef<string[]>([]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      keySequence.current = [...keySequence.current, e.key].slice(-10);
      if (JSON.stringify(keySequence.current) === JSON.stringify(KONAMI_CODE)) {
        setIsOverclocked(prev => !prev);
        console.log("🎮 KONAMI CODE ACTIVATED!");
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;

    canvas.width = w;
    canvas.height = h;

    const gl = canvas.getContext("webgl", { alpha: true, preserveDrawingBuffer: true });
    if (!gl) return;

    const gridW = Math.ceil(w / CELL_SIZE);
    const gridH = Math.ceil(h / CELL_SIZE);

    // Initialize CPU-side state (Game of Life simulation runs on CPU, rendering on GPU)
    const density = isOverclocked ? 0.2 : 0.15;
    const state = new Float32Array(gridW * gridH);
    const trail = new Float32Array(gridW * gridH);
    
    for (let i = 0; i < gridW * gridH; i++) {
      state[i] = Math.random() < density ? 1 : 0;
      trail[i] = state[i];
    }
    stateRef.current = state;

    console.log("GameOfLife: grid", gridW, "x", gridH, "cells:", state.filter(v => v > 0).length);

    // Vertex shader
    const vsSource = `
      attribute vec2 position;
      varying vec2 uv;
      void main() {
        uv = position * 0.5 + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;

    // Fragment shader - samples texture and renders cells
    const fsSource = `
      precision highp float;
      uniform sampler2D uTrail;
      uniform vec2 gridSize;
      uniform bool overclocked;
      varying vec2 uv;

      void main() {
        vec2 gridPos = uv * gridSize;
        vec2 cellIndex = floor(gridPos);
        vec2 cellUV = fract(gridPos);
        vec2 texCoord = (cellIndex + 0.5) / gridSize;

        float t = texture2D(uTrail, texCoord).r;

        float pad = 0.1;
        float mask = step(pad, cellUV.x) * step(cellUV.x, 1.0 - pad) *
                     step(pad, cellUV.y) * step(cellUV.y, 1.0 - pad);

        vec3 color = overclocked ? vec3(0.2, 1.0, 0.4) : vec3(0.4, 0.5, 0.9);
        float opacity = overclocked ? 0.8 : 0.35;
        float alpha = t * mask * opacity;

        float gridLine = (step(0.95, cellUV.x) + step(0.95, cellUV.y)) * 0.04;
        gl_FragColor = vec4(color * (alpha + gridLine), alpha + gridLine);
      }
    `;

    // Compile shaders
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);

    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    // Quad buffer
    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1
    ]), gl.STATIC_DRAW);

    // Trail texture (updated each frame from CPU)
    const trailTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, trailTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const posLoc = gl.getAttribLocation(program, "position");
    const gridSizeLoc = gl.getUniformLocation(program, "gridSize");
    const overclockedLoc = gl.getUniformLocation(program, "overclocked");

    // Game of Life step (CPU)
    const simulateStep = () => {
      const newState = new Float32Array(gridW * gridH);
      
      for (let y = 0; y < gridH; y++) {
        for (let x = 0; x < gridW; x++) {
          let neighbors = 0;
          
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = (x + dx + gridW) % gridW;
              const ny = (y + dy + gridH) % gridH;
              neighbors += state[ny * gridW + nx] > 0.5 ? 1 : 0;
            }
          }
          
          const i = y * gridW + x;
          const alive = state[i] > 0.5;
          
          if (alive) {
            newState[i] = (neighbors === 2 || neighbors === 3) ? 1 : 0;
          } else {
            newState[i] = neighbors === 3 ? 1 : 0;
          }
        }
      }
      
      // Copy new state
      for (let i = 0; i < gridW * gridH; i++) {
        state[i] = newState[i];
      }
      
      // Update trail (fade + new cells)
      const decay = isOverclocked ? 0.04 : 0.015;
      for (let i = 0; i < gridW * gridH; i++) {
        trail[i] = Math.max(state[i], trail[i] - decay);
      }
    };

    // Upload trail to texture
    const uploadTrail = () => {
      const data = new Uint8Array(gridW * gridH * 4);
      for (let i = 0; i < gridW * gridH; i++) {
        const v = Math.floor(trail[i] * 255);
        data[i * 4 + 0] = v;
        data[i * 4 + 1] = v;
        data[i * 4 + 2] = v;
        data[i * 4 + 3] = 255;
      }
      gl.bindTexture(gl.TEXTURE_2D, trailTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gridW, gridH, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    };

    const render = () => {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, trailTex);

      gl.uniform2f(gridSizeLoc, gridW, gridH);
      gl.uniform1i(overclockedLoc, isOverclocked ? 1 : 0);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };

    const loop = (time: number) => {
      const fps = isOverclocked ? OVERCLOCK_FPS : SIM_FPS;
      const interval = 1000 / fps;

      if (time - lastSimTime.current >= interval) {
        simulateStep();
        uploadTrail();
        lastSimTime.current = time;
      }
      render();
      animationRef.current = requestAnimationFrame(loop);
    };

    // Initial upload
    uploadTrail();
    
    console.log("GameOfLife: Starting");
    animationRef.current = requestAnimationFrame(loop);

    return () => cancelAnimationFrame(animationRef.current);
  }, [isOverclocked]);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-[-1] overflow-hidden pointer-events-none"
      style={{ backgroundColor: "#09090b" }}
    >
      <canvas ref={canvasRef} className="block w-full h-full" />
      <div className="absolute inset-0 bg-gradient-to-t from-[#09090b] via-transparent to-transparent" />
      <div className="absolute inset-0 bg-gradient-to-b from-[#09090b]/80 via-transparent to-[#09090b]/50" />

      {isOverclocked && (
        <div className="absolute top-20 right-8 bg-green-900/30 border border-green-500/60 text-green-400 px-4 py-2 font-mono text-xs uppercase tracking-widest animate-pulse z-50">
          ⚡ System Overclocked
        </div>
      )}
    </div>
  );
};

export default GameOfLife;
