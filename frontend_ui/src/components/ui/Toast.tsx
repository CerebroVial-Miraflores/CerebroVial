import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

// FASE 1 rediseño UI — sistema de toasts (spec: .toasts/.toast del prototipo).
// Apilado abajo-derecha; entrada con doble rAF para disparar la transición;
// autodismiss a los 5200 ms y salida de 420 ms antes de remover. En <md el stack
// se eleva sobre la bottom-nav (+ safe-area). El provider limpia TODOS los
// timers y rAF pendientes al desmontar (sin fugas en tests).
// En Fase 1 el provider solo envuelve /ui-lab; la integración al shell es de la
// fase de vistas.

export type ToastKind = 'ok' | 'warn' | 'pred' | 'info';

export interface ToastOptions {
  kind?: ToastKind;
  title: string;
  body?: string;
  durationMs?: number;
}

interface ToastItem {
  id: number;
  kind: ToastKind;
  title: string;
  body?: string;
  state: 'enter' | 'shown' | 'exit';
}

interface ToastContextValue {
  push: (options: ToastOptions) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

const KIND_CLASSES: Record<ToastKind, { card: string; dot: string }> = {
  ok: { card: 'border-ok/45', dot: 'bg-ok' },
  warn: { card: 'border-warn/50', dot: 'bg-warn' },
  pred: { card: 'border-sev/50', dot: 'bg-sev' },
  info: { card: 'border-line-2', dot: 'bg-info' },
};

const DEFAULT_DURATION_MS = 5200;
const EXIT_MS = 420;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(0);
  const timersRef = useRef(new Map<number, ReturnType<typeof setTimeout>>());
  const rafsRef = useRef(new Set<number>());

  const setState = useCallback((id: number, state: ToastItem['state']) => {
    setToasts((current) => current.map((t) => (t.id === id ? { ...t, state } : t)));
  }, []);

  const push = useCallback(
    (options: ToastOptions) => {
      const id = nextId.current++;
      const duration = options.durationMs ?? DEFAULT_DURATION_MS;
      setToasts((current) => [
        ...current,
        { id, kind: options.kind ?? 'info', title: options.title, body: options.body, state: 'enter' },
      ]);

      // Doble rAF (como el prototipo): el nodo monta en 'enter' y recién al
      // segundo frame pasa a 'shown' para que la transición de entrada anime.
      const raf1 = requestAnimationFrame(() => {
        rafsRef.current.delete(raf1);
        const raf2 = requestAnimationFrame(() => {
          rafsRef.current.delete(raf2);
          setState(id, 'shown');
        });
        rafsRef.current.add(raf2);
      });
      rafsRef.current.add(raf1);

      const dismissTimer = setTimeout(() => {
        setState(id, 'exit');
        const removeTimer = setTimeout(() => {
          setToasts((current) => current.filter((t) => t.id !== id));
          timersRef.current.delete(id);
        }, EXIT_MS);
        timersRef.current.set(id, removeTimer);
      }, duration);
      timersRef.current.set(id, dismissTimer);
    },
    [setState],
  );

  useEffect(() => {
    const timers = timersRef.current;
    const rafs = rafsRef.current;
    return () => {
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
      for (const raf of rafs) cancelAnimationFrame(raf);
      rafs.clear();
    };
  }, []);

  const value = useMemo(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      {createPortal(
        <div
          role="status"
          aria-live="polite"
          className="pointer-events-none fixed right-[18px] bottom-[calc(var(--h-bottomnav)+env(safe-area-inset-bottom)+12px)] z-(--z-toast) flex w-[324px] max-w-[calc(100vw-36px)] flex-col gap-[9px] md:bottom-[18px]"
        >
          {toasts.map((toast) => (
            <div
              key={toast.id}
              data-kind={toast.kind}
              className={`pointer-events-auto rounded-ctl border bg-canvas-2/95 px-3.5 py-3 shadow-2xl shadow-black/55 backdrop-blur-md transition-all duration-[400ms] ease-fluid ${
                KIND_CLASSES[toast.kind].card
              } ${toast.state === 'shown' ? 'translate-x-0 opacity-100' : 'translate-x-6 opacity-0'}`}
            >
              <b className="flex items-center gap-2 text-[12.5px] font-bold">
                <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${KIND_CLASSES[toast.kind].dot}`} />
                {toast.title}
              </b>
              {toast.body != null && (
                <span className="mt-1 block text-[11px] leading-relaxed text-ink-2">{toast.body}</span>
              )}
            </div>
          ))}
        </div>,
        document.body,
      )}
    </ToastContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components -- hook colocado al provider (patrón SessionContext)
export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used inside a ToastProvider');
  }
  return ctx;
}
