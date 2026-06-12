import type { ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';

// FASE 3 rediseño UI — item de alerta expandible (spec: .al del prototipo).
// Severidades crit/pred/hw/info → borde izquierdo 3px + icono cuadrado tintado
// en la escala de estado (bad/sev/warn/info). Cuerpo expandible con el patrón
// grid-rows 0fr→1fr del prototipo (anima la altura sin medirla). Estado "new"
// (entrada dinámica) → animate-new-alert (token --animate-new-alert).
// Controlado: el panel decide qué item está abierto.

export type AlertSeverity = 'crit' | 'pred' | 'hw' | 'info';

const SEVERITY_CLASSES: Record<AlertSeverity, { border: string; icon: string }> = {
  crit: { border: 'border-l-bad', icon: 'bg-bad/14 text-bad' },
  pred: { border: 'border-l-sev', icon: 'bg-sev/14 text-sev' },
  hw: { border: 'border-l-warn', icon: 'bg-warn/14 text-warn' },
  info: { border: 'border-l-info', icon: 'bg-info/12 text-info' },
};

interface AlertAccordionProps {
  severity: AlertSeverity;
  /** Icono del cuadrado tintado (15×15 aprox.); sin icono no se renderiza el cuadrado. */
  icon?: ReactNode;
  title: ReactNode;
  /** Fila de metadatos (lugar · tiempo · confianza). */
  meta?: ReactNode;
  open: boolean;
  onToggle: () => void;
  /** Entrada dinámica recién llegada → animación new-pulse. */
  isNew?: boolean;
  /** Cuerpo expandible (stats, ConfBar, acciones). */
  children?: ReactNode;
  className?: string;
}

export function AlertAccordion({
  severity,
  icon,
  title,
  meta,
  open,
  onToggle,
  isNew = false,
  children,
  className = '',
}: AlertAccordionProps) {
  const sev = SEVERITY_CLASSES[severity];

  return (
    <div
      className={`rounded-[11px] border border-line border-l-[3px] bg-white/2 transition-colors duration-300 ease-fluid hover:border-line-2 ${sev.border} ${
        isNew ? 'animate-new-alert' : ''
      } ${className}`}
    >
      <button
        type="button"
        aria-expanded={open}
        onClick={onToggle}
        className="flex w-full items-start gap-2.5 px-3 py-[11px] text-left"
      >
        {icon != null && (
          <span
            aria-hidden="true"
            className={`grid h-[29px] w-[29px] shrink-0 place-items-center rounded-btn ${sev.icon}`}
          >
            {icon}
          </span>
        )}
        <span className="min-w-0 flex-1">
          <span className="block text-[12.5px] font-semibold leading-[1.35]">{title}</span>
          {meta != null && (
            <span className="mt-1 flex flex-wrap gap-[9px] text-[10.5px] text-ink-2">{meta}</span>
          )}
        </span>
        <ChevronDown
          size={15}
          aria-hidden="true"
          className={`mt-[2px] shrink-0 text-ink-3 transition-transform duration-300 ease-fluid ${
            open ? 'rotate-180' : ''
          }`}
        />
      </button>

      <div
        className={`grid transition-[grid-template-rows] duration-[380ms] ease-fluid ${
          open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'
        }`}
      >
        <div className="overflow-hidden">
          {children != null && (
            <div className="px-3 pb-3 pt-[2px] text-[11.5px] text-ink-2">{children}</div>
          )}
        </div>
      </div>
    </div>
  );
}
