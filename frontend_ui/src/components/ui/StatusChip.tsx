import type { ReactNode } from 'react';

// FASE 1 rediseño UI — chip de estado de intersección/proceso (spec: .stchip del
// prototipo). 'recovery' es el estado "EN RECUPERACIÓN" del flujo de aplicar plan
// (paleta warn, label propio por defecto).

export type Status = 'ok' | 'warn' | 'bad' | 'recovery';

const STATUS_CLASSES: Record<Status, string> = {
  ok: 'border-ok/40 bg-ok/13 text-ok',
  warn: 'border-warn/40 bg-warn/13 text-warn',
  bad: 'border-bad/40 bg-bad/15 text-bad',
  recovery: 'border-warn/40 bg-warn/13 text-warn',
};

const DEFAULT_LABEL: Record<Status, string> = {
  ok: 'OK',
  warn: 'ALERTA',
  bad: 'CRÍTICO',
  recovery: 'EN RECUPERACIÓN',
};

interface StatusChipProps {
  status: Status;
  children?: ReactNode;
  className?: string;
}

export function StatusChip({ status, children, className = '' }: StatusChipProps) {
  return (
    <span
      className={`num whitespace-nowrap rounded-full border px-[9px] py-1 text-[9.5px] font-extrabold tracking-[0.1em] ${STATUS_CLASSES[status]} ${className}`}
    >
      {children ?? DEFAULT_LABEL[status]}
    </span>
  );
}
