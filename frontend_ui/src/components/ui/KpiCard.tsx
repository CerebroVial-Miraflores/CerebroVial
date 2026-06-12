import type { ReactNode } from 'react';

import { Sparkline } from './Sparkline';
import { useCountUp } from './useCountUp';

// FASE 1 rediseño UI — tarjeta KPI (spec: .kpi del prototipo). Valor .num con
// countUp en mount, delta pos/neg/neu, Sparkline (o `footer` que la reemplaza —
// caso del KPI de predicción con ConfBar), glowbar en hover y variante warn.
// Root <button> solo si hay onClick.

type DeltaTone = 'pos' | 'neg' | 'neu';

const DELTA_CLASSES: Record<DeltaTone, string> = {
  pos: 'bg-ok/12 text-ok',
  neg: 'bg-bad/12 text-bad',
  neu: 'bg-white/6 text-ink-2',
};

interface KpiCardProps {
  label: string;
  /** null = sin dato (render "—" estático, sin countUp) — política de paridad F3. */
  value: number | null;
  /**
   * Valor no-numérico que reemplaza al countUp (p. ej. "ALTA" del KPI de
   * predicción). El caller lo estiliza (color/tamaño) si hace falta.
   */
  valueLabel?: ReactNode;
  unit?: string;
  decimals?: number;
  delta?: { text: string; tone: DeltaTone };
  spark?: readonly number[];
  /** Clase de color del sparkline (currentColor), p. ej. 'text-warn'. */
  sparkClassName?: string;
  /** Reemplaza al sparkline (p. ej. ConfBar en el KPI de predicción). */
  footer?: ReactNode;
  warn?: boolean;
  onClick?: () => void;
  className?: string;
}

export function KpiCard({
  label,
  value,
  valueLabel,
  unit,
  decimals = 0,
  delta,
  spark,
  sparkClassName = 'text-brand',
  footer,
  warn = false,
  onClick,
  className = '',
}: KpiCardProps) {
  // useCountUp es hook (no condicional): con value null anima sobre 0 pero el
  // render lo ignora y muestra el placeholder estático.
  const display = useCountUp(value ?? 0, { decimals });
  const Root = onClick ? 'button' : 'div';
  const shown = valueLabel ?? (value === null ? '—' : display);

  return (
    <Root
      {...(onClick ? { type: 'button' as const, onClick } : {})}
      className={`group relative overflow-hidden rounded-card border bg-linear-to-b from-white/5 to-white/2 px-[15px] pb-[11px] pt-[13px] text-left transition-all duration-300 ease-fluid hover:-translate-y-[3px] hover:shadow-xl hover:shadow-black/45 ${
        warn ? 'border-warn/40 hover:border-warn/70' : 'border-line hover:border-brand/45'
      } ${className}`}
    >
      <div className="mb-[7px] flex items-center gap-2">
        <span className="flex-1 text-[10px] font-bold uppercase leading-[1.3] tracking-[0.09em] text-ink-2">
          {label}
        </span>
        {delta && (
          <span
            className={`whitespace-nowrap rounded-full px-[7px] py-[2px] text-[10.5px] font-extrabold ${DELTA_CLASSES[delta.tone]}`}
          >
            {delta.text}
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className="num text-[25px] font-bold tracking-[-0.02em]">{shown}</span>
        {unit != null && <span className="text-[11.5px] font-semibold text-ink-2">{unit}</span>}
      </div>

      {footer ?? (spark && spark.length > 1 ? (
        <Sparkline data={spark} className={`mt-2 h-8 w-full ${sparkClassName}`} />
      ) : null)}

      <i
        aria-hidden="true"
        className="absolute inset-x-0 bottom-0 h-[2px] bg-linear-to-r from-transparent via-brand to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100"
      />
    </Root>
  );
}
