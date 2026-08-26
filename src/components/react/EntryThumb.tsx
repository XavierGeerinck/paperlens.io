import { useEffect, useRef } from 'react';
import { categoryVar, drawThumb } from '../../lib/thumb';

interface Props {
  slug: string;
  category: string;
  /** drift slowly instead of holding a single frame */
  animated?: boolean;
  className?: string;
}

/**
 * The seeded signal figure for an entry. Decorative texture, not data —
 * no axes, no numbers, and hidden from assistive tech.
 */
export default function EntryThumb({ slug, category, animated = false, className = 'entry-thumb' }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const hue = categoryVar(category);
    let raf = 0;
    let phase = 0;

    const draw = () => drawThumb(canvas, slug, hue, phase);

    if (animated && !reduced) {
      const loop = () => {
        phase += 0.004;
        draw();
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    } else {
      raf = requestAnimationFrame(draw);
    }

    window.addEventListener('resize', draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', draw);
    };
  }, [slug, category, animated]);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}
