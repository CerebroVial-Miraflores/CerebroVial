import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';

import { acquireBodyLock } from './bodyLock';
import { StatusChip, type Status } from './StatusChip';

// FASE 1 rediseño UI — drawer lateral (spec: .drawer del prototipo). Mobile-first:
// <md es sheet de pantalla completa; ≥md panel derecho de min(480px, 94vw) [el
// brief fija ~480; el prototipo usa 452]. Siempre montado para animar translate-x;
// cerrado queda aria-hidden + inert. Esc/scrim cierran; lock de body mientras
// open. SIN focus-trap completo: deuda a11y — implementar antes de la evaluación
// SUS / pruebas con usuarios de la tesis.

interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  status?: Status;
  statusLabel?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Drawer({ open, onClose, title, status, statusLabel, children, className = '' }: DrawerProps) {
  const panelRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const release = acquireBodyLock();
    panelRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      release();
      previous?.focus?.();
    };
  }, [open, onClose]);

  return createPortal(
    <>
      <div
        data-testid="drawer-scrim"
        aria-hidden="true"
        onClick={onClose}
        className={`fixed inset-0 z-(--z-drawer) bg-canvas/55 backdrop-blur-sm transition-opacity duration-300 ${
          open ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
      />
      <aside
        ref={panelRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-hidden={!open}
        inert={!open}
        className={`fixed inset-0 z-(--z-drawer) flex flex-col border-l border-line-2 bg-linear-to-b from-canvas-2 to-canvas shadow-2xl shadow-black/55 outline-none transition-transform duration-[450ms] ease-fluid md:left-auto md:w-[min(480px,94vw)] ${
          open ? 'translate-x-0' : 'translate-x-[106%]'
        } ${className}`}
      >
        <header className="flex items-center gap-[11px] border-b border-line px-[18px] py-[15px]">
          <h2 className="min-w-0 flex-1 text-[15px] font-bold leading-tight">{title}</h2>
          {status && <StatusChip status={status}>{statusLabel}</StatusChip>}
          <button
            type="button"
            onClick={onClose}
            aria-label="Cerrar"
            className="grid h-[31px] w-[31px] shrink-0 place-items-center rounded-btn border border-line text-ink-2 transition-colors hover:border-line-2 hover:text-ink"
          >
            <X size={14} />
          </button>
        </header>
        <div className="flex-1 overflow-y-auto p-[18px]">{children}</div>
      </aside>
    </>,
    document.body,
  );
}
