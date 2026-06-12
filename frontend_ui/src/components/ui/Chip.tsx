import type { ReactNode } from 'react';

// FASE 1 rediseño UI — chips toggle de capas/filtros (spec: .chip y .hchip del
// prototipo). Chip = paleta brand (capas del mapa); HChip = paleta warn
// (horizontes de predicción / horas del histórico).

interface ChipProps {
  on: boolean;
  onToggle: (next: boolean) => void;
  children: ReactNode;
  className?: string;
}

export function Chip({ on, onToggle, children, className = '' }: ChipProps) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={() => onToggle(!on)}
      className={`flex items-center gap-1.5 rounded-full border px-[11px] py-[5px] text-[11px] font-semibold transition-colors duration-200 ${
        on
          ? 'border-brand/50 bg-brand/13 text-brand-2'
          : 'border-line text-ink-2 hover:border-line-2 hover:text-ink'
      } ${className}`}
    >
      {on && <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-brand-2" />}
      {children}
    </button>
  );
}

export function HChip({ on, onToggle, children, className = '' }: ChipProps) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={() => onToggle(!on)}
      className={`rounded-full border px-2.5 py-[5px] text-[11px] font-bold transition-colors duration-200 ${
        on
          ? 'border-warn/55 bg-warn/13 text-warn'
          : 'border-line text-ink-2 hover:border-line-2 hover:text-ink'
      } ${className}`}
    >
      {children}
    </button>
  );
}
