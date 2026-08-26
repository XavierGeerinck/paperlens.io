import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { categoryVar, drawThumb } from '../../lib/thumb';

export interface BrowserEntry {
  slug: string;
  title: string;
  subtitle: string;
  date: string;
  status: string;
  category: string;
  impact: string;
  readTime: string;
  tags: string[];
  hasSim: boolean;
  pdfUrl?: string;
  githubUrl?: string;
  webUrl?: string;
  featured: boolean;
}

interface Props {
  entries: BrowserEntry[];
}

const CATEGORIES = [
  { id: 'all', label: 'all', key: 'a' },
  { id: 'paper', label: 'paper', key: 'p' },
  { id: 'deep-dive', label: 'deep-dive', key: 'd' },
  { id: 'idea', label: 'idea', key: 'i' },
];

const KEY_TO_CATEGORY: Record<string, string> = { a: 'all', p: 'paper', d: 'deep-dive', i: 'idea' };

function hostOf(url: string) {
  try {
    const u = new URL(url);
    return (u.hostname + u.pathname).replace(/^www\./, '');
  } catch {
    return url;
  }
}

export default function PaperBrowser({ entries }: Props) {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [selected, setSelected] = useState(0);
  const listRef = useRef<HTMLOListElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const words = q ? q.split(/\s+/) : [];
    return entries.filter((entry) => {
      if (category !== 'all' && entry.category !== category) return false;
      if (!words.length) return true;
      const haystack = [
        entry.title,
        entry.subtitle,
        entry.status,
        entry.category,
        entry.impact,
        entry.tags.join(' '),
      ]
        .join(' ')
        .toLowerCase();
      return words.every((w) => haystack.includes(w));
    });
  }, [entries, query, category]);

  const index = Math.min(selected, Math.max(visible.length - 1, 0));
  const current = visible[index];

  // keep the selection in range when the filter changes
  useEffect(() => {
    setSelected((s) => (s >= visible.length ? 0 : s));
  }, [visible.length]);

  // the status bar carries the position
  useEffect(() => {
    const pos = document.getElementById('pl-pos');
    if (pos) pos.textContent = visible.length ? `${index + 1}/${visible.length}` : `0/${visible.length}`;
  }, [index, visible.length]);

  // preview figure
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !current) return;
    const draw = () => drawThumb(canvas, current.slug, categoryVar(current.category));
    const frame = requestAnimationFrame(draw);
    window.addEventListener('resize', draw);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', draw);
    };
  }, [current?.slug, current?.category]);

  const open = useCallback((entry?: BrowserEntry) => {
    if (entry) window.location.href = `/idea/${entry.slug}/`;
  }, []);

  // anything on the page can drive the filter: document.dispatchEvent(
  //   new CustomEvent('pl:filter', { detail: 'jepa' })
  // )
  useEffect(() => {
    const onFilter = (e: Event) => {
      setQuery((e as CustomEvent<string>).detail || '');
      setCategory('all');
      setSelected(0);
      document.getElementById('papers')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    document.addEventListener('pl:filter', onFilter);
    return () => document.removeEventListener('pl:filter', onFilter);
  }, []);

  // j / k / Enter / l / a p d i — only while this island owns the keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.matches?.('input, textarea, select, [contenteditable="true"]')) {
        if (e.key === 'Enter') {
          e.preventDefault();
          target.blur();
          listRef.current?.querySelector<HTMLAnchorElement>('.row.is-sel')?.focus();
        }
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (document.getElementById('pl-help')?.hasAttribute('open')) return;

      const n = visible.length;

      if (KEY_TO_CATEGORY[e.key]) {
        e.preventDefault();
        setCategory(KEY_TO_CATEGORY[e.key]);
        setSelected(0);
        return;
      }
      if (!n) return;

      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        setSelected((s) => (Math.min(s, n - 1) + 1) % n);
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        setSelected((s) => (Math.min(s, n - 1) - 1 + n) % n);
      } else if (e.key === 'Enter' || e.key === 'l') {
        e.preventDefault();
        open(visible[Math.min(selected, n - 1)]);
      }
    };

    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [visible, selected, open]);

  // keep the selected row on screen when moving by keyboard — but never scroll
  // the page on first paint, which would jump the reader past the hero
  const mounted = useRef(false);
  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true;
      return;
    }
    listRef.current?.querySelector<HTMLElement>('.row.is-sel')?.scrollIntoView({ block: 'nearest' });
  }, [index]);

  return (
    <div className="browser-wrap">
      <div className="toolbar">
        <label className="field">
          <span className="sigil">/</span>
          <input
            id="pl-filter"
            type="search"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(0);
            }}
            placeholder="filter by title, tag, status…"
            autoComplete="off"
            spellCheck={false}
            aria-label="Filter entries"
          />
        </label>

        <div className="filters" role="group" aria-label="Category">
          {CATEGORIES.map((c) => (
            <button
              key={c.id}
              type="button"
              className={c.id === category ? 'is-active' : undefined}
              aria-pressed={c.id === category}
              onClick={() => {
                setCategory(c.id);
                setSelected(0);
              }}
            >
              <b>{c.key}</b>
              {c.label.slice(1)}
            </button>
          ))}
        </div>

        <span className="hint">
          <span className="kbd">j</span>/<span className="kbd">k</span> move · <span className="kbd">⏎</span> open ·{' '}
          <span className="kbd">/</span> filter
        </span>
      </div>

      <div className="ranger">
        <ol className="list" ref={listRef} aria-label="Entries">
          {visible.map((entry, i) => (
            <li key={entry.slug}>
              <a
                className={`row${i === index ? ' is-sel' : ''}`}
                href={`/idea/${entry.slug}/`}
                onMouseEnter={() => setSelected(i)}
                onFocus={() => setSelected(i)}
              >
                <span className="mk" aria-hidden="true">
                  ▶
                </span>
                <span className="d">{entry.date}</span>
                <span className="st">
                  <span className={`chip status-${entry.status}`}>{entry.status}</span>
                </span>
                <span className={`cat cat-${entry.category}`}>{entry.category}</span>
                <span className="t">{entry.title}</span>
                <span className="rt">{entry.readTime}</span>
                <span className="fl">
                  {entry.hasSim && <span className="c-green">sim</span>}
                  {entry.pdfUrl && <span className="c-aqua"> pdf</span>}
                </span>
              </a>
            </li>
          ))}
          {!visible.length && (
            <li className="empty">
              <p>
                No entries match <b className="c-green">/{query}</b>
                {category !== 'all' ? ` in ${category}` : ''}.
              </p>
              <p className="dim">
                Clear the filter with <span className="kbd">Esc</span>, or press <span className="kbd">a</span> for all
                categories.
              </p>
            </li>
          )}
        </ol>

        {current && (
          <aside className="preview" aria-live="polite" aria-label="Preview">
            <p className="pv-head dim">{current.slug}.md</p>
            <canvas className="pv-thumb" ref={canvasRef} aria-hidden="true" />
            <h3 className="pv-title">{current.title}</h3>
            <p className="pv-sub">{current.subtitle}</p>
            <dl className="kv">
              <dt>status</dt>
              <dd>
                <span className={`chip status-${current.status}`}>{current.status}</span>
              </dd>
              <dt>type</dt>
              <dd className={`cat cat-${current.category}`}>{current.category}</dd>
              <dt>date</dt>
              <dd>{current.date}</dd>
              <dt>read</dt>
              <dd>{current.readTime}</dd>
              <dt>impact</dt>
              <dd>{current.impact}</dd>
              {current.hasSim ? <dt>sim</dt> : null}
              {current.hasSim ? <dd className="c-green">yes</dd> : null}
              {current.pdfUrl ? <dt>pdf</dt> : null}
              {current.pdfUrl ? (
                <dd>
                  <a href={current.pdfUrl} target="_blank" rel="noopener noreferrer">
                    {hostOf(current.pdfUrl)}
                  </a>
                </dd>
              ) : null}
            </dl>
            <p className="pv-tags">
              {current.tags.map((tag) => (
                <span className="tag" key={tag}>
                  {tag.toLowerCase().replace(/\s+/g, '-')}
                </span>
              ))}
            </p>
            <div className="pv-actions">
              <a className="btn primary" href={`/idea/${current.slug}/`}>
                Read <span className="kbd">⏎</span>
              </a>
              {current.hasSim && (
                <a className="btn" href={`/idea/${current.slug}/#simulation`}>
                  Run simulation
                </a>
              )}
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
