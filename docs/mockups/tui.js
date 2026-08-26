/* PaperLens redesign — Direction A "TUI" — shared behaviour for the mockups.
   Everything here is mockup-grade: enough to feel real, not production code. */
(function () {
  'use strict';
  const PL = (window.PL = window.PL || {});

  /* ------------------------------------------------------------ data
     Real entries from src/content/ideas (title/date/status/category/
     readTime/impact/tags are copied verbatim). */
  PL.entries = [
    { slug: 'leworldmodel-stable-jepa', title: 'The Collapse-Proof World Model', subtitle: 'How LeWorldModel trains a JEPA from pixels with two losses instead of six', date: '2026-05-16', status: 'RESEARCH', category: 'paper', impact: '6→1 loss reduction · 48× faster planning · 15M params', readTime: '14m', tags: ['JEPA', 'World Models', 'Self-Supervised', 'LeCun', 'Balestriero', 'Representation Collapse'], sim: 'LeWorldModel', pdf: 'arxiv.org/pdf/2603.19312', featured: true },
    { slug: 'turboquant-polarquant', title: 'The Rotation Revolution: Near-Optimal KV Cache Quantization', subtitle: 'How TurboQuant and PolarQuant use random rotations to squeeze LLM memory to within a constant of the information-theoretic limit', date: '2026-04-20', status: 'RESEARCH', category: 'paper', impact: '4.2× KV cache compression, quality-neutral at 3.5 bpc', readTime: '15m', tags: ['Quantization', 'KV Cache', 'LLM Inference', 'TurboQuant', 'PolarQuant', 'Google Research'], sim: 'TurboQuant', pdf: 'arxiv.org/pdf/2502.02617', featured: true },
    { slug: 'lace-lithography', title: 'Lace Lithography: Printing with Matter Waves', subtitle: 'How neutral helium atom microscopy from a Norwegian university spinout became a bid to push chip fabrication beyond the EUV wall.', date: '2026-03-25', status: 'PROTOTYPE', category: 'deep-dive', impact: 'Sub-EUV chip patterning', readTime: '22m', tags: ['Lithography', 'Semiconductor', 'Atom Optics', 'Quantum Physics', 'EUV', 'Nanotechnology'], sim: 'LaceLithography', pdf: 'arxiv.org/pdf/2111.12582', featured: true },
    { slug: 'towards-autonomous-mathematics-research', title: 'Towards Autonomous Mathematics Research', subtitle: 'Aletheia shows how generator–verifier loops and tool use push math agents beyond Olympiad problems.', date: '2026-02-12', status: 'RESEARCH', category: 'deep-dive', impact: 'Autonomy in mathematical discovery', readTime: '22m', tags: ['Aletheia', 'Gemini Deep Think', 'Math Agents', 'Verification', 'Tool Use'], sim: 'AletheiaSimulation', pdf: 'arxiv.org/pdf/2602.10177' },
    { slug: 'dreamzero-world-action-models', title: 'DreamZero World Action Models', subtitle: 'Joint video-action diffusion that turns world prediction into zero-shot robot control.', date: '2026-02-10', status: 'RESEARCH', category: 'deep-dive', impact: 'Zero-shot embodied control', readTime: '22m', tags: ['Robotics', 'Diffusion', 'World Models', 'Video', 'Control'], sim: 'DreamZero', pdf: 'dreamzero0.github.io/DreamZero.pdf' },
    { slug: 'privacy-preserving-llm-inference', title: 'Privacy-Preserving Inference', subtitle: 'From Trusted Hardware to Homomorphic Encryption: a trajectory for confidential AI.', date: '2026-02-02', status: 'RESEARCH', category: 'deep-dive', impact: 'Confidential AI', readTime: '15m', tags: ['Cryptography', 'Privacy', 'FHE', 'TEE', 'MPC'], sim: 'PrivacyPreservingInference', pdf: 'eprint.iacr.org/2026/105.pdf' },
    { slug: 'alphagenome', title: 'AlphaGenome', subtitle: 'A unifying long-context DNA model for base-resolution regulatory prediction and variant scoring.', date: '2026-01-29', status: 'RESEARCH', category: 'deep-dive', impact: 'Genome-scale regulatory insight', readTime: '22m', tags: ['Genomics', 'Transformers', 'Variant Effect', 'Multimodal', 'DeepMind'], sim: 'AlphaGenome', pdf: 'storage.googleapis.com/deepmind-media/papers/alphagenome.pdf' },
    { slug: 'seal-self-adapting-lms', title: 'SEAL: Self-Adapting Language Models', subtitle: 'Teaching language models to rewrite their own weights through self-generated finetuning data and RL-driven self-edits', date: '2026-01-26', status: 'RESEARCH', category: 'deep-dive', impact: 'Autonomous model adaptation', readTime: '22m', tags: ['Reinforcement Learning', 'Meta-Learning', 'Continual Learning', 'Weight Updates', 'Self-Improvement'], sim: 'SEALAdaptation', pdf: 'arxiv.org/pdf/2506.10943', featured: true },
    { slug: 'deepmind-deep-delta', title: 'Deep Delta Learning: Beyond the Additive Bias of ResNets', subtitle: 'Generalizing residual connections into dynamic geometric reflections, projections, and identity mappings.', date: '2026-01-24', status: 'RESEARCH', category: 'deep-dive', impact: 'Enhanced capacity for complex state transitions', readTime: '12m', tags: ['Deep Delta Learning', 'Residual Networks', 'Householder Transformation', 'Geometric Linear Algebra'], sim: 'DeepmindDeepDelta', pdf: 'arxiv.org/pdf/2601.00417' },
    { slug: 'kona-1', title: 'Kona 1: Energy-Based Reasoning', subtitle: "Logical Intelligence recruits Yann LeCun to productize 'Implicit Chain-of-Thought' reasoning.", date: '2026-01-22', status: 'RESEARCH', category: 'deep-dive', impact: 'Provable correctness', readTime: '15m', tags: ['Architecture', 'EBM', 'Logical Intelligence', 'Implicit CoT'], sim: 'Kona1Suite', pdf: 'arxiv.org/pdf/2511.07124', featured: true },
    { slug: 'deepseek-model1', title: 'DeepSeek MODEL1', subtitle: 'Unveiling Value Vector Position Awareness and Engram Memory in the next-generation FlashMLA codebase.', date: '2026-01-22', status: 'PROTOTYPE', category: 'deep-dive', impact: '30% memory reduction', readTime: '12m', tags: ['Architecture', 'DeepSeek', 'FlashMLA', 'Engram'], sim: 'DeepSeekModel1', pdf: 'arxiv.org/abs/2412.19437', featured: true },
    { slug: 'adam-epsilon-optimization', title: "Adam's Hidden Parameter (The Epsilon Trap)", subtitle: 'How epsilon=1e-10 lets Adam skate through flat loss landscapes that trap the default 1e-8', date: '2026-01-18', status: 'RESEARCH', category: 'deep-dive', impact: 'Better optimization', readTime: '12m', tags: ['Optimization', 'Adam', 'PyTorch', 'Training'], sim: 'AdamEpsilonSimulation', featured: true },
    { slug: 'deepseek-engram-conditional-memory', title: 'DeepSeek Engram (Conditional Memory)', subtitle: 'Adding an O(1) knowledge lookup primitive that complements MoE conditional compute.', date: '2026-01-15', status: 'PROTOTYPE', category: 'paper', impact: 'Infinite memory', readTime: '22m', tags: ['DeepSeek', 'MoE', 'Retrieval', 'Hashing', 'Systems', 'Long Context'], sim: 'DeepSeekEngram', pdf: 'arxiv.org/pdf/2601.07372v1' },
    { slug: 'test-time-training-long-context', title: 'Reimagining LLM Memory: Test-Time Training', subtitle: 'Using context as training data unlocks models that learn at test-time', date: '2026-01-09', status: 'PROTOTYPE', category: 'paper', impact: 'Infinite context w/ constant latency', readTime: '15m', tags: ['TTT', 'Long Context', 'Meta-Learning', 'Transformers'], sim: 'TTTSimulation', pdf: 'arxiv.org/pdf/2512.23675', featured: true },
    { slug: 'digital-red-queen', title: 'Digital Red Queen', subtitle: 'Adversarial evolution and weaponized LLMs in Core War', date: '2026-01-08', status: 'RESEARCH', category: 'paper', impact: 'Automated malware evolution', readTime: '12m', tags: ['Evolutionary Algorithms', 'LLMs', 'Core War', 'Sakana AI', 'Cybersecurity'], sim: 'DigitalRedQueen', pdf: 'arxiv.org/pdf/2601.03335', featured: true },
    { slug: 'control-theoretic-imperative', title: 'The Control-Theoretic Imperative', subtitle: 'Why Model Predictive Control, not autoregression, is the architecture of general intelligence', date: '2026-01-07', status: 'RESEARCH', category: 'paper', impact: 'AGI architecture', readTime: '20m', tags: ['MPC', 'World Models', 'AGI', 'System 2', 'JEPA'], sim: 'ControlTheoretic', featured: true },
    { slug: 'multi-head-latent-attention', title: 'Multi-Head Latent Attention: The Memory-Efficient Future of LLMs', subtitle: 'How DeepSeek-V3 compresses KV caches by 93% using low-rank latent projections and weight absorption.', date: '2026-01-04', status: 'RESEARCH', category: 'paper', impact: 'Massive KV-cache reduction', readTime: '15m', tags: ['DeepSeek', 'MLA', 'Transformer', 'Efficiency'], sim: 'MLASimulation', pdf: 'arxiv.org/pdf/2502.07864v1', featured: true },
    { slug: 'rubin-architecture', title: 'Gigascale Intelligence: Deciphering the NVIDIA Rubin Architecture', subtitle: 'Beyond Blackwell: engineering million-token contexts with HBM4, Vera CPUs, and CPX accelerators.', date: '2026-01-04', status: 'RESEARCH', category: 'paper', impact: 'Million-token context & 600kW rack density', readTime: '22m', tags: ['NVIDIA', 'Rubin', 'HBM4', 'Vera-CPU', 'Infrastructure'], sim: 'RubinArchitecture', pdf: 'GTC2025_Keynote.pdf', featured: true },
    { slug: 'sub-quadratic-scaling', title: 'Breaking the Quadratic Wall: The Rise of Sub-Quadratic Scaling', subtitle: 'Moving beyond Transformers with Mamba, Jamba, and the shift toward linear-time sequence modeling.', date: '2026-01-04', status: 'RESEARCH', category: 'paper', impact: 'Linear scaling & constant memory inference', readTime: '18m', tags: ['Mamba', 'Jamba', 'State Space Models', 'Efficient AI'], sim: 'SubQuadratic', pdf: 'arxiv.org/pdf/2312.00752', featured: true },
    { slug: 'asahi-m1n1', title: 'Asahi Linux m1n1: The Hardware Puppeteer', subtitle: 'Reverse engineering Apple Silicon through real-time MMIO tracing and Python-based hypervisors.', date: '2026-01-02', status: 'RESEARCH', category: 'paper', impact: 'Hardware freedom', readTime: '15m', tags: ['Asahi Linux', 'Hypervisor', 'Reverse Engineering', 'Apple Silicon', 'ARM64'], sim: 'AsahiM1n1' },
    { slug: 'verifiable-rewards', title: 'Beyond the Vibes: Why Verifiable Rewards (RLVR) is the New Scaling Law', subtitle: "Moving from subjective human 'vibes' to objective ground-truth verification in the quest for AGI reasoning.", date: '2025-01-24', status: 'RESEARCH', category: 'paper', impact: 'Reliable machine reasoning', readTime: '12m', tags: ['Reinforcement Learning', 'RLVR', 'DeepSeek-R1', 'GRPO', 'AI Safety'], sim: 'RLVR', pdf: 'arxiv.org/pdf/2501.12948', featured: true },
    { slug: 'objective-verification-rlvr', title: 'The Shift to Objective Verification', subtitle: "Moving from human 'vibes' to ground-truth feedback loops with RLVR and Synthetic Textbooks.", date: '2025-01-04', status: 'RESEARCH', category: 'paper', impact: 'Reliable reasoning', readTime: '15m', tags: ['RLVR', 'Synthetic Data', 'DeepSeek-R1', 'Formal Verification', 'Lean4'], sim: 'ObjectiveVerifier', pdf: 'arxiv.org/pdf/2501.12948', featured: true },
    { slug: 'deepseek-moe', title: 'Advanced Mixture of Experts: The DeepSeek-V3 Architecture', subtitle: 'Mastering efficiency through Shared Experts and Bias-Driven Load Balancing.', date: '2024-12-28', status: 'RESEARCH', category: 'paper', impact: 'Efficient scaling (671B parameters / 37B active)', readTime: '15m', tags: ['DeepSeek', 'MoE', 'Sparse-Models', 'Machine-Learning'], sim: 'DeepSeekMoE', pdf: 'arxiv.org/pdf/2412.19437', featured: true },
    { slug: 'mapping-the-mind', title: 'Mapping the Mind: Decoding LLMs with Sparse Autoencoders', subtitle: 'Using smaller helper models to translate millions of cryptic neurons into human-understandable concepts.', date: '2024-06-15', status: 'RESEARCH', category: 'paper', impact: 'Mechanistic interpretability & AI safety', readTime: '18m', tags: ['Mechanistic Interpretability', 'SAEs', 'AI Safety', 'Monosemanticity'], sim: 'MappingTheMind', pdf: 'arxiv.org/pdf/2406.04093.pdf', featured: true },
    { slug: 'jepa-world-models', title: 'JEPA: The Architecture of Reasoning', subtitle: 'Why AGI requires predicting representations, not pixels.', date: '2024-05-24', status: 'RESEARCH', category: 'paper', impact: 'World models & planning', readTime: '18m', tags: ['JEPA', 'LeCun', 'Self-Supervised Learning', 'World Models'], sim: 'JEPASimulation', pdf: 'arxiv.org/pdf/2301.08243.pdf', featured: true },
    { slug: 'brain-mimetic', title: 'BrainMimetic Intelligence', subtitle: 'Engineering test-time plasticity with Titans architecture to enable continuous learning during inference.', date: '2024-05-21', status: 'PROTOTYPE', category: 'idea', impact: 'Infinite context', readTime: '25m', tags: ['AGI', 'Titans', 'PyTorch', 'Neuroscience'], sim: 'BrainMimetic' },
    { slug: 'deepseek-mhc', title: 'DeepSeek mHC Protocol', subtitle: 'Solving the Signal Survival problem in deep networks using Manifold Constrained Hyper-Connections.', date: '2024-02-14', status: 'ALPHA', category: 'paper', impact: 'Infinite depth', readTime: '18m', tags: ['DeepSeek', 'Math', 'Scaling Laws'], sim: 'DeepSeekMHC', pdf: 'arxiv.org/pdf/2512.24880', featured: true },
    { slug: 'asml-tin-droplets', title: 'ASML Engineering the Perfect Droplet: High-Frequency Liquid Metal Jetting', subtitle: 'How acoustic physics and active control systems generate millions of identical molten tin targets per second for EUV lithography.', date: '2023-10-27', status: 'RESEARCH', category: 'deep-dive', impact: 'Enabling 3nm chip fabrication', readTime: '15m', tags: ['EUV Lithography', 'Fluid Dynamics', 'Control Systems', 'Acoustics'], sim: 'ASMLTinDroplets', pdf: 'spiedigitallibrary.org/…/12.2258237', featured: true },
  ];

  /* ------------------------------------------------------------ utils */
  PL.cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  PL.catColor = (cat) => ({ paper: '--aqua', 'deep-dive': '--orange', idea: '--yellow' }[cat] || '--blue');

  // Small seeded PRNG (mulberry32) so every thumbnail is stable per slug.
  PL.rng = function (seedStr) {
    let h = 1779033703 ^ seedStr.length;
    for (let i = 0; i < seedStr.length; i++) { h = Math.imul(h ^ seedStr.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
    let a = h >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  PL.esc = esc;

  /* ------------------------------------------------------------ thumbnails
     Procedural "oscilloscope" traces per entry: a stand-in for the live
     simulation preview the real site would render. */
  PL.drawThumb = function (canvas, seed, colorVar) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 140;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const rnd = PL.rng(seed);
    ctx.fillStyle = PL.cssVar('--bg0h'); ctx.fillRect(0, 0, w, h);
    // faint grid
    ctx.strokeStyle = PL.cssVar('--bg1'); ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 16) { ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 16) { ctx.beginPath(); ctx.moveTo(0, y + .5); ctx.lineTo(w, y + .5); ctx.stroke(); }
    const col = PL.cssVar(colorVar || '--aqua');
    const traces = 2 + Math.floor(rnd() * 2);
    for (let t = 0; t < traces; t++) {
      const f1 = 1 + rnd() * 3, f2 = 4 + rnd() * 9, ph = rnd() * 6.28, amp = .18 + rnd() * .22, decay = rnd() < .5 ? 0 : rnd() * 1.5;
      ctx.beginPath();
      ctx.strokeStyle = col; ctx.globalAlpha = t === 0 ? .95 : .45; ctx.lineWidth = t === 0 ? 1.6 : 1;
      for (let x = 0; x <= w; x += 2) {
        const u = x / w;
        const env = Math.exp(-decay * u);
        const y = h / 2 + h * amp * env * (Math.sin(u * f1 * 6.28 + ph) * .7 + Math.sin(u * f2 * 6.28) * .3) + (rnd() - .5) * 1.5;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // endpoint marker
    ctx.fillStyle = col; ctx.fillRect(w - 6, h / 2 - 2, 4, 4);
  };

  /* ------------------------------------------------------------ theme */
  const root = document.documentElement;
  PL.currentTheme = function () {
    const stamped = root.getAttribute('data-theme');
    if (stamped) return stamped;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  };
  PL.setTheme = function (t) {
    root.setAttribute('data-theme', t);
    try { localStorage.setItem('pl-theme', t); } catch (e) { /* storage may be unavailable */ }
    document.querySelectorAll('[data-bgword]').forEach((el) => { el.textContent = t === 'dark' ? 'light' : 'dark'; });
    document.dispatchEvent(new CustomEvent('pl:theme', { detail: t }));
  };
  // Dark only by decision (2026-08-26); the theme plumbing stays so canvases can re-read tokens.
  PL.toggleTheme = () => {};

  /* ------------------------------------------------------------ status bar */
  PL.setMode = function (mode) {
    document.querySelectorAll('.statusbar .mode').forEach((el) => { el.dataset.mode = mode; el.textContent = mode; });
  };
  document.addEventListener('focusin', (e) => { if (e.target.matches('input, textarea')) PL.setMode('INSERT'); });
  document.addEventListener('focusout', (e) => { if (e.target.matches('input, textarea')) PL.setMode('NORMAL'); });

  /* ------------------------------------------------------------ help overlay */
  PL.help = function (open) {
    const ov = document.getElementById('help');
    if (!ov) return;
    const willOpen = open === undefined ? !ov.hasAttribute('open') : open;
    if (willOpen) { ov.setAttribute('open', ''); PL.setMode('HELP'); ov.querySelector('button, a')?.focus(); }
    else { ov.removeAttribute('open'); PL.setMode('NORMAL'); }
  };

  /* ------------------------------------------------------------ global keys */
  let pendingG = false;
  document.addEventListener('keydown', (e) => {
    const inField = e.target.matches('input, textarea, [contenteditable]');
    const help = document.getElementById('help');
    if (e.key === 'Escape') {
      if (help && help.hasAttribute('open')) { PL.help(false); return; }
      if (inField) { e.target.blur(); return; }
    }
    if (inField) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key) {
      case '?': e.preventDefault(); PL.help(); return;
      case '/': {
        const f = document.getElementById('filter');
        if (f) { e.preventDefault(); f.focus(); f.select(); }
        return;
      }
      case 'g': if (pendingG) { window.scrollTo({ top: 0 }); pendingG = false; } else { pendingG = true; setTimeout(() => (pendingG = false), 600); } return;
      case 'G': window.scrollTo({ top: document.body.scrollHeight }); return;
      default: break;
    }
    if (/^[0-9]$/.test(e.key)) {
      const tab = document.querySelector(`[data-tab-key="${e.key}"]`);
      if (tab) { e.preventDefault(); tab.click(); }
    }
  });
  document.addEventListener('click', (e) => {
    const a = e.target.closest('[data-action]');
    if (!a) return;
    if (a.dataset.action === 'theme') PL.toggleTheme();
    if (a.dataset.action === 'help') PL.help();
    if (a.dataset.action === 'close-help') PL.help(false);
  });
  document.addEventListener('click', (e) => { if (e.target.id === 'help') PL.help(false); });

  /* ------------------------------------------------------------ ranger browser */
  PL.browser = function (opts) {
    const list = document.getElementById(opts.list || 'list');
    const preview = document.getElementById(opts.preview || 'preview');
    const filter = document.getElementById(opts.filter || 'filter');
    const countEl = document.getElementById(opts.count || 'count');
    const posEl = document.getElementById(opts.pos || 'pos');
    if (!list) return;

    let cat = 'all', q = '', sel = 0, visible = [];

    function matches(en) {
      if (cat !== 'all' && en.category !== cat) return false;
      if (!q) return true;
      const hay = [en.title, en.subtitle, en.status, en.category, en.impact, en.tags.join(' ')].join(' ').toLowerCase();
      return q.split(/\s+/).every((w) => hay.includes(w));
    }

    function render() {
      visible = PL.entries.filter(matches);
      list.innerHTML = visible.map((en, i) => `
        <li>
          <a class="row" href="${opts.href || 'article.html'}" data-i="${i}" aria-label="${esc(en.title)}">
            <span class="mk" aria-hidden="true">▶</span>
            <span class="d">${en.date}</span>
            <span class="st"><span class="chip status-${en.status}">${en.status}</span></span>
            <span class="cat cat-${en.category}">${en.category}</span>
            <span class="t">${esc(en.title)}</span>
            <span class="rt">${en.readTime}</span>
            <span class="fl">${en.sim ? '<span class="c-green" title="has simulation">sim</span>' : ''}${en.pdf ? ' <span class="c-blue" title="source pdf">pdf</span>' : ''}</span>
          </a>
        </li>`).join('');
      if (countEl) countEl.textContent = `${visible.length} ${visible.length === 1 ? 'entry' : 'entries'}${cat !== 'all' ? ' · ' + cat : ''}${q ? ' · /' + q : ''}`;
      select(Math.min(sel, Math.max(visible.length - 1, 0)), false);
      if (!visible.length && preview) {
        preview.innerHTML = `<p class="dim">No entries match <b class="c-yellow">/${esc(q)}</b>${cat !== 'all' ? ' in ' + cat : ''}.</p><p class="sm dim">Clear the filter with <kbd>Esc</kbd> or press <kbd>a</kbd> for all categories.</p>`;
      }
    }

    function select(i, scroll) {
      sel = i;
      list.querySelectorAll('.row').forEach((r, k) => r.classList.toggle('is-sel', k === i));
      const en = visible[i];
      if (posEl) posEl.textContent = visible.length ? `${i + 1}/${visible.length}` : '0/0';
      if (en && preview) renderPreview(en);
      if (scroll) list.querySelector('.row.is-sel')?.scrollIntoView({ block: 'nearest' });
    }

    function renderPreview(en) {
      preview.innerHTML = `
        <div class="pv-head">
          <span class="dim">${esc(en.slug)}.md</span>
        </div>
        <canvas class="pv-thumb" height="150" aria-hidden="true"></canvas>
        <h3 class="pv-title">${esc(en.title)}</h3>
        <p class="pv-sub">${esc(en.subtitle)}</p>
        <dl class="kv">
          <dt>status</dt><dd><span class="chip status-${en.status}">${en.status}</span></dd>
          <dt>type</dt><dd class="cat cat-${en.category}">${en.category}</dd>
          <dt>date</dt><dd>${en.date}</dd>
          <dt>read</dt><dd>${en.readTime}</dd>
          <dt>impact</dt><dd>${esc(en.impact)}</dd>
          ${en.sim ? `<dt>sim</dt><dd class="c-green">${esc(en.sim)}</dd>` : ''}
          ${en.pdf ? `<dt>pdf</dt><dd><a href="https://${esc(en.pdf)}" target="_blank" rel="noopener">${esc(en.pdf)}</a></dd>` : ''}
        </dl>
        <p class="pv-tags">${en.tags.map((t) => `<span class="tag">${esc(t.toLowerCase().replace(/\s+/g, '-'))}</span>`).join(' ')}</p>
        <div class="pv-actions">
          <a class="btn primary" href="${opts.href || 'article.html'}">Read <kbd>⏎</kbd></a>
          ${en.sim ? `<a class="btn" href="${opts.href || 'article.html'}#sim">Run simulation</a>` : ''}
        </div>`;
      const c = preview.querySelector('canvas');
      requestAnimationFrame(() => PL.drawThumb(c, en.slug, PL.catColor(en.category)));
    }

    list.addEventListener('mouseover', (e) => {
      const r = e.target.closest('.row'); if (r) select(+r.dataset.i, false);
    });
    list.addEventListener('focusin', (e) => {
      const r = e.target.closest('.row'); if (r) select(+r.dataset.i, false);
    });

    if (filter) filter.addEventListener('input', () => { q = filter.value.trim().toLowerCase(); sel = 0; render(); });
    document.querySelectorAll('[data-cat]').forEach((b) => b.addEventListener('click', () => {
      cat = b.dataset.cat; sel = 0;
      document.querySelectorAll('[data-cat]').forEach((x) => x.classList.toggle('is-active', x === b));
      render();
    }));

    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea')) {
        if (e.key === 'Enter') { e.target.blur(); list.querySelector('.row.is-sel')?.focus(); }
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const help = document.getElementById('help');
      if (help && help.hasAttribute('open')) return;
      const n = visible.length;
      if (!n) return;
      if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); select((sel + 1) % n, true); }
      else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); select((sel - 1 + n) % n, true); }
      else if (e.key === 'Enter' || e.key === 'l') { const r = list.querySelector('.row.is-sel'); if (r) window.location.href = r.href; }
      else if (['a', 'p', 'd', 'i'].includes(e.key)) {
        const map = { a: 'all', p: 'paper', d: 'deep-dive', i: 'idea' };
        document.querySelector(`[data-cat="${map[e.key]}"]`)?.click();
      }
    });

    document.addEventListener('pl:theme', () => { const en = visible[sel]; if (en && preview) renderPreview(en); });
    render();
  };

  /* ------------------------------------------------------------ scroll position (article) */
  PL.scrollMeter = function (opts) {
    const lnEl = document.getElementById(opts.ln || 'ln');
    const pctEl = document.getElementById(opts.pct || 'pct');
    const LINE = 24;
    function update() {
      const doc = document.documentElement;
      const max = doc.scrollHeight - window.innerHeight;
      const total = Math.max(1, Math.round(doc.scrollHeight / LINE));
      const cur = Math.min(total, 1 + Math.round(window.scrollY / LINE));
      const pct = max <= 0 ? 100 : Math.round((window.scrollY / max) * 100);
      if (lnEl) lnEl.textContent = `Ln ${cur}/${total}`;
      if (pctEl) pctEl.textContent = pct <= 0 ? 'Top' : pct >= 100 ? 'Bot' : pct + '%';
    }
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  };
})();
