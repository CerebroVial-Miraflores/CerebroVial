import { useEffect, useRef, useState } from 'react';

// FASE 1 rediseño UI — countUp del prototipo: rAF + ease-out cúbico 1-(1-p)^3,
// ~950 ms, re-anima desde el último valor mostrado cuando cambia el target.
// Con prefers-reduced-motion (o reducedMotion: true) el display ES el target
// (derivado en render: sin animación, sin flash de 0).

// Formateo del prototipo: separador de miles, decimales fijos y signo − (U+2212).
export function formatNumber(value: number, decimals = 0): string {
  const abs = Math.abs(value).toFixed(decimals);
  const [integer, fraction] = abs.split('.');
  const grouped = integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (value < 0 ? '−' : '') + grouped + (fraction ? `.${fraction}` : '');
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

interface CountUpOptions {
  durationMs?: number;
  decimals?: number;
  reducedMotion?: boolean;
}

export function useCountUp(target: number, options: CountUpOptions = {}): string {
  const { durationMs = 950, decimals = 0 } = options;
  const reduced = options.reducedMotion ?? prefersReducedMotion();
  // Último valor mostrado: punto de partida si el target cambia en caliente.
  const shownRef = useRef(0);
  const [animated, setAnimated] = useState(0);

  useEffect(() => {
    if (reduced) {
      shownRef.current = target;
      return;
    }
    const from = shownRef.current;
    let raf = 0;
    let start: number | null = null;
    const frame = (timestamp: number) => {
      if (start === null) start = timestamp;
      const progress = Math.min(1, (timestamp - start) / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = from + (target - from) * eased;
      shownRef.current = value;
      setAnimated(value);
      if (progress < 1) {
        raf = requestAnimationFrame(frame);
      }
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs, reduced]);

  return formatNumber(reduced ? target : animated, decimals);
}
