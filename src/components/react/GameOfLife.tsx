import { useEffect, useRef, useState } from "react";

/**
 * Easter egg. Nothing renders until ↑↑↓↓←→←→BA, then Life runs behind the page
 * until you enter it again. It used to be the ambient background; the redesign
 * wants a flat terminal ground, so it moved behind the cheat code.
 */

const CELL_SIZE = 10;
const SIM_FPS = 45;

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

const GameOfLife = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>(0);
  const lastSimTime = useRef<number>(0);
  const keySequence = useRef<string[]>([]);

  const [active, setActive] = useState(false);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      keySequence.current = [...keySequence.current, e.key].slice(-10);
      if (JSON.stringify(keySequence.current) === JSON.stringify(KONAMI_CODE)) {
        keySequence.current = [];
        setActive((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  useEffect(() => {
    if (!active) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

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

    const state = new Float32Array(gridW * gridH);
    const trail = new Float32Array(gridW * gridH);

    for (let i = 0; i < gridW * gridH; i++) {
      state[i] = Math.random() < 0.18 ? 1 : 0;
      trail[i] = state[i];
    }

    const vsSource = `
      attribute vec2 position;
      varying vec2 uv;
      void main() {
        uv = position * 0.5 + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;

    // mint (#35d492) cells on the page ground
    const fsSource = `
      precision highp float;
      uniform sampler2D uTrail;
      uniform vec2 gridSize;
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

        vec3 color = vec3(0.208, 0.831, 0.573);
        float alpha = t * mask * 0.5;

        float gridLine = (step(0.95, cellUV.x) + step(0.95, cellUV.y)) * 0.03;
        gl_FragColor = vec4(color * (alpha + gridLine), alpha + gridLine);
      }
    `;

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

    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );

    const trailTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, trailTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const posLoc = gl.getAttribLocation(program, "position");
    const gridSizeLoc = gl.getUniformLocation(program, "gridSize");

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
          newState[i] = alive ? (neighbors === 2 || neighbors === 3 ? 1 : 0) : neighbors === 3 ? 1 : 0;
        }
      }

      for (let i = 0; i < gridW * gridH; i++) state[i] = newState[i];
      for (let i = 0; i < gridW * gridH; i++) trail[i] = Math.max(state[i], trail[i] - 0.03);
    };

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

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };

    const loop = (time: number) => {
      if (time - lastSimTime.current >= 1000 / SIM_FPS) {
        simulateStep();
        uploadTrail();
        lastSimTime.current = time;
      }
      render();
      animationRef.current = requestAnimationFrame(loop);
    };

    uploadTrail();
    animationRef.current = requestAnimationFrame(loop);

    return () => cancelAnimationFrame(animationRef.current);
  }, [active]);

  if (!active) return null;

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 z-0 overflow-hidden pointer-events-none"
      aria-hidden="true"
    >
      <canvas ref={canvasRef} className="block w-full h-full" />
      <div className="fixed bottom-[calc(var(--status-h)+.75rem)] right-3 z-[60] px-2 py-1 text-[11px] font-mono text-mint-400 border border-mint-400/40 bg-bg0h">
        life: running &nbsp; <span className="text-ink4">↑↑↓↓←→←→BA to stop</span>
      </div>
    </div>
  );
};

export default GameOfLife;
