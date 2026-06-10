import type { ReactNode } from 'react';

// FASE 1 rediseño UI — pill de estado vivo con dot pulsante (spec: .pill del
// prototipo, "IA ACTIVA"). El dot usa animate-pulse default (mismo approach que
// la pill de sesión del AppShell).

interface PillProps {
  children: ReactNode;
  className?: string;
}

export function Pill({ children, className = '' }: PillProps) {
  return (
    <span
      className={`flex w-fit items-center gap-[7px] rounded-full border border-ok/35 bg-ok/10 px-3 py-[7px] text-[11.5px] font-bold tracking-[0.04em] text-ok ${className}`}
    >
      <span aria-hidden="true" className="h-[7px] w-[7px] animate-pulse rounded-full bg-current" />
      {children}
    </span>
  );
}
