import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';

import { acquireBodyLock } from './bodyLock';
import { pushOverlay } from './overlayStack';

// FASE 1 rediseño UI — modal genérico (spec: .modal-wrap/.mscrim/.modal del
// prototipo). Scrim-click y Esc cierran; foco inicial al contenedor y restore al
// cerrar; lock de scroll del body. SIN focus-trap completo: deuda a11y —
// implementar antes de la evaluación SUS / pruebas con usuarios de la tesis.
// FASE 3 (B1): integrado al overlay-stack — con Drawer debajo, Esc cierra solo
// este Modal (deuda de Fase 1 saldada). onClose en ref + deps [open], mismo
// racional que Drawer (el orden del stack no debe depender de la identidad
// del callback).

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Modal({ open, onClose, title, subtitle, children, className = '' }: ModalProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  // Sin deps: sincroniza la identidad vigente tras cada commit (el keydown la
  // lee en tiempo de evento). No se escribe en render (regla refs de hooks v7).
  useEffect(() => {
    onCloseRef.current = onClose;
  });

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const overlay = pushOverlay();
    const release = acquireBodyLock();
    boxRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && overlay.isTop()) onCloseRef.current();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      overlay.release();
      release();
      previous?.focus?.();
    };
  }, [open]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-(--z-modal) grid place-items-center p-6">
      <div
        data-testid="modal-scrim"
        aria-hidden="true"
        onClick={onClose}
        className="animate-fade-in absolute inset-0 bg-canvas/60 backdrop-blur-sm"
      />
      <div
        ref={boxRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        className={`animate-in relative max-h-full w-full max-w-[780px] overflow-y-auto rounded-[18px] border border-line-2 bg-linear-to-b from-canvas-2 to-canvas p-5 shadow-2xl shadow-black/60 outline-none ${className}`}
      >
        <div className="mb-3 flex items-start gap-3">
          <div className="min-w-0 flex-1">
            {title != null && <h2 className="text-base font-bold leading-tight">{title}</h2>}
            {subtitle != null && <p className="mt-1 text-[11.5px] text-ink-2">{subtitle}</p>}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Cerrar"
            className="grid h-[31px] w-[31px] shrink-0 place-items-center rounded-btn border border-line text-ink-2 transition-colors hover:border-line-2 hover:text-ink"
          >
            <X size={14} />
          </button>
        </div>
        {children}
      </div>
    </div>,
    document.body,
  );
}
