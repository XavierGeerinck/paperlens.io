/**
 * PaperLens — TUI behaviour shared by every page.
 *
 * Keys are part of the design, so they must be true everywhere:
 *   0-3      switch window (the numbered tabs in the title bar)
 *   /        browse papers (focuses the filter on the home page)
 *   gg / G   top / bottom
 *   gt / gT  next / previous view (article pages)
 *   ?        key map
 *   Esc      close the key map, leave a field
 *
 * j / k / Enter belong to the paper browser island, which registers its own
 * handler while it is mounted.
 */

type Mode = 'NORMAL' | 'INSERT' | 'HELP';

export function setMode(mode: Mode) {
  document.querySelectorAll<HTMLElement>('.statusbar .mode').forEach((el) => {
    el.dataset.mode = mode;
    el.textContent = mode;
  });
}

function helpEl() {
  return document.getElementById('pl-help');
}

export function toggleHelp(open?: boolean) {
  const ov = helpEl();
  if (!ov) return;
  const willOpen = open === undefined ? !ov.hasAttribute('open') : open;
  if (willOpen) {
    ov.setAttribute('open', '');
    setMode('HELP');
    ov.querySelector<HTMLElement>('button, a')?.focus();
  } else {
    ov.removeAttribute('open');
    setMode('NORMAL');
  }
}

function isTyping(target: EventTarget | null) {
  const el = target as HTMLElement | null;
  return !!el?.matches?.('input, textarea, select, [contenteditable="true"]');
}

/* ---------- global listeners (registered once per document) ---------- */

let pendingG = false;

function onKeydown(e: KeyboardEvent) {
  const help = helpEl();

  if (e.key === 'Escape') {
    if (help?.hasAttribute('open')) {
      toggleHelp(false);
      return;
    }
    if (isTyping(e.target)) (e.target as HTMLElement).blur();
    return;
  }

  if (isTyping(e.target)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  // g-prefixed motions: gg (top), gt / gT (views)
  if (pendingG) {
    pendingG = false;
    if (e.key === 'g') {
      e.preventDefault();
      window.scrollTo({ top: 0 });
      return;
    }
    if (e.key === 't' || e.key === 'T') {
      e.preventDefault();
      cycleView(e.key === 't' ? 1 : -1);
      return;
    }
  }

  switch (e.key) {
    case '?':
      e.preventDefault();
      toggleHelp();
      return;
    case 'g':
      pendingG = true;
      setTimeout(() => (pendingG = false), 700);
      return;
    case 'G':
      window.scrollTo({ top: document.body.scrollHeight });
      return;
    case '/': {
      const filter = document.getElementById('pl-filter') as HTMLInputElement | null;
      e.preventDefault();
      if (filter) {
        filter.focus();
        filter.select();
      } else {
        window.location.href = '/#papers';
      }
      return;
    }
    default:
      break;
  }

  if (/^[0-9]$/.test(e.key) && !help?.hasAttribute('open')) {
    const tab = document.querySelector<HTMLAnchorElement>(`.titlebar .tab[data-key="${e.key}"]`);
    if (tab) {
      e.preventDefault();
      tab.click();
    }
  }
}

function cycleView(dir: number) {
  const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>('.viewstrip [role="tab"]'));
  if (!tabs.length) return;
  const current = tabs.findIndex((t) => t.getAttribute('aria-selected') === 'true');
  const next = tabs[(current + dir + tabs.length) % tabs.length];
  next?.click();
  next?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

function onClick(e: MouseEvent) {
  const target = e.target as HTMLElement;
  const action = target.closest<HTMLElement>('[data-action]');
  if (action?.dataset.action === 'help') toggleHelp();
  if (action?.dataset.action === 'close-help') toggleHelp(false);
  if (target.id === 'pl-help') toggleHelp(false);
}

function onFocusIn(e: FocusEvent) {
  if (isTyping(e.target)) setMode('INSERT');
}
function onFocusOut(e: FocusEvent) {
  if (isTyping(e.target)) setMode('NORMAL');
}

/* ---------- per-page setup (re-runs after view transitions) ---------- */

function decorateCodeBlocks() {
  document.querySelectorAll<HTMLPreElement>('.doc pre').forEach((pre) => {
    if (pre.parentElement?.classList.contains('codeblock')) return;

    const wrap = document.createElement('div');
    wrap.className = 'codeblock';
    pre.replaceWith(wrap);
    wrap.appendChild(pre);

    const lang = pre.getAttribute('data-language');
    if (lang && lang !== 'plaintext') {
      const legend = document.createElement('span');
      legend.className = 'codeblock-lang';
      legend.textContent = lang;
      wrap.appendChild(legend);
    }

    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'codeblock-copy';
    copy.textContent = 'copy';
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        copy.textContent = 'copied';
      } catch {
        copy.textContent = 'select + ⌘C';
      }
      setTimeout(() => (copy.textContent = 'copy'), 1600);
    });
    wrap.appendChild(copy);
  });
}

function scrollMeter() {
  const ln = document.getElementById('pl-ln');
  const pct = document.getElementById('pl-pct');
  if (!ln && !pct) return;

  const LINE = 24; // px per "line", so the counter reads like a buffer
  const update = () => {
    const doc = document.documentElement;
    const max = doc.scrollHeight - window.innerHeight;
    const total = Math.max(1, Math.round(doc.scrollHeight / LINE));
    const cur = Math.min(total, 1 + Math.round(window.scrollY / LINE));
    const p = max <= 0 ? 100 : Math.round((window.scrollY / max) * 100);
    if (ln) ln.textContent = `Ln ${cur}/${total}`;
    if (pct) pct.textContent = p <= 0 ? 'Top' : p >= 100 ? 'Bot' : `${p}%`;
  };

  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
}

function tableOfContents() {
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>('.toc a'));
  if (!links.length) return;
  // ids can start with a digit ("#1-the-thinking-highway"), which querySelector rejects
  const targets = links
    .map((a) => document.getElementById((a.getAttribute('href') || '').replace(/^#/, '')))
    .filter(Boolean) as HTMLElement[];
  if (!targets.length) return;

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => a.classList.toggle('is-active', a.getAttribute('href') === `#${entry.target.id}`));
      });
    },
    { rootMargin: '-10% 0px -75% 0px' }
  );
  targets.forEach((t) => io.observe(t));
}

function viewTabs() {
  document.querySelectorAll<HTMLElement>('.viewstrip').forEach((strip) => {
    if (strip.dataset.bound) return;
    strip.dataset.bound = '1';
    strip.querySelectorAll<HTMLButtonElement>('[role="tab"]').forEach((tab) => {
      tab.addEventListener('click', () => {
        strip.querySelectorAll<HTMLButtonElement>('[role="tab"]').forEach((t) => {
          t.setAttribute('aria-selected', String(t === tab));
        });
        const panelIds = Array.from(strip.querySelectorAll<HTMLButtonElement>('[role="tab"]')).map((t) =>
          t.getAttribute('aria-controls')
        );
        panelIds.forEach((id) => {
          const panel = id ? document.getElementById(id) : null;
          if (panel) panel.hidden = id !== tab.getAttribute('aria-controls');
        });
      });
    });
  });
}

function setup() {
  setMode('NORMAL');
  decorateCodeBlocks();
  scrollMeter();
  tableOfContents();
  viewTabs();
}

if (typeof document !== 'undefined') {
  document.addEventListener('keydown', onKeydown);
  document.addEventListener('click', onClick);
  document.addEventListener('focusin', onFocusIn);
  document.addEventListener('focusout', onFocusOut);
  document.addEventListener('astro:page-load', setup);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup, { once: true });
  } else {
    setup();
  }
}
