import type { ReactNode } from 'react';

import { CountBadge } from './CountBadge';

// FASE 1 rediseño UI — panel con header (spec: .panel/.panel-h del prototipo).
// No reemplaza a ui/Card.tsx (vistas v1): convive hasta que cada vista migre.

interface PanelProps {
  title: ReactNode;
  count?: number;
  mini?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Panel({ title, count, mini, actions, children, className = '' }: PanelProps) {
  return (
    <section
      className={`overflow-hidden rounded-panel border border-line bg-linear-to-b from-white/4 to-white/1 ${className}`}
    >
      <header className="flex items-center gap-[9px] border-b border-line px-3.5 py-3">
        <b className="text-[12.5px] font-bold">{title}</b>
        {typeof count === 'number' && <CountBadge count={count} variant="panel" showZero />}
        {mini != null && <span className="ml-auto text-[10.5px] text-ink-2">{mini}</span>}
        {actions}
      </header>
      {children}
    </section>
  );
}
