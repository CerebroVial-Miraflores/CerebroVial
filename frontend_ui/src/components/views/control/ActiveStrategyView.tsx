// HU-05 — Vista pasiva del Operador: estrategia de control vigente.
//
// READ-ONLY por contrato (DHU-020). El Operador SOLO observa: nombre de la
// estrategia (etiqueta de dominio DHU-006), tiempos de verde por acceso,
// y desde cuándo está vigente. Ningún botón en este componente dispara
// /control/recommend NI /control/__internal/activate — es vista, no
// interacción. El playground interactivo está reservado para Admin
// (ControlPlayground.tsx).
//
// Comportamiento:
// 1. Mount → fetch GET /control/active-state/{node_id} (single source of truth).
// 2. Mount → abre stream SSE; cada "active-state-changed" dispara re-fetch.
// 3. Render según estado:
//    - loading: spinner textual mientras se carga el primer estado.
//    - success: tarjeta con etiqueta, subtítulo, ciclo, fases y timestamp.
//    - stale (CA-05.4 / DHU-005 Caso B): mantiene la última estrategia
//      conocida en pantalla, la marca como "No confirmada", y muestra un
//      contador "Última confirmación hace Xs / X min" que avanza en vivo.
//    - error: mensaje neutral cuando no hubo NINGÚN fetch exitoso previo.
//
// Triggers de stale:
// - onError del stream SSE (inmediato, Fase 5).
// - staleTimer de 10 s sin eventos (Fase 6 — el motor está vivo pero dejó
//   de empujar; mismo comportamiento que onError desde la vista del
//   Operador). Cada evento reinicia el timer.
//
// Mapping DHU-006: strategy_mode crudo se traduce a etiqueta legible vía
// labelForStrategy(). El crudo "webster"/"max_pressure" NUNCA debe aparecer
// visible — el guard rail Vitest (ActiveStrategyView.test.tsx) falla si
// llega a renderizarse.
import { useEffect, useState } from 'react';
import { Card } from '../../ui/Card';
import { TimingBar } from './TimingBar';
import { PHASE_SUBTITLES } from './controlTypes';
import {
  controlActiveStateService,
  type ActiveStateResponse,
} from '../../../services/controlActiveStateService';
import { openControlActiveStateStream } from '../../../services/sseClient';
import { labelForStrategy } from './strategyLabels';

type ViewState =
  | { kind: 'loading' }
  | { kind: 'success'; data: ActiveStateResponse }
  | { kind: 'stale'; data: ActiveStateResponse }
  | { kind: 'error'; message: string };

interface ActiveStrategyViewProps {
  nodeId?: string;
}

const DEFAULT_NODE_ID = 'larco_schell';
const STALE_TIMEOUT_MS = 10_000;
const COUNTER_TICK_MS = 1_000;

function formatElapsed(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
  if (totalSeconds < 60) return `${totalSeconds} s`;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const remSeconds = totalSeconds % 60;
  if (totalMinutes < 60) {
    return remSeconds === 0
      ? `${totalMinutes} min`
      : `${totalMinutes} min ${remSeconds} s`;
  }
  const totalHours = Math.floor(totalMinutes / 60);
  const remMinutes = totalMinutes % 60;
  return remMinutes === 0
    ? `${totalHours} h`
    : `${totalHours} h ${remMinutes} min`;
}

export const ActiveStrategyView = ({
  nodeId = DEFAULT_NODE_ID,
}: ActiveStrategyViewProps) => {
  const [state, setState] = useState<ViewState>({ kind: 'loading' });
  const [lastConfirmedAt, setLastConfirmedAt] = useState<Date | null>(null);
  // ``now`` en state — react-hooks/purity prohíbe leer ``Date.now()`` en el
  // body del render. Para evitar el lag del bug visual ("arranca en 0 s y
  // salta a ~11 s"), sincronizamos ``now`` desde los callbacks de transición
  // a stale (staleTimer, onError del SSE, catch del fetch que cae a stale)
  // ADEMÁS del setInterval que tickea mientras estamos en stale.
  const [now, setNow] = useState<Date>(() => new Date());

  useEffect(() => {
    let cancelled = false;
    let staleTimer: number | null = null;

    const armStaleTimer = () => {
      if (staleTimer !== null) {
        window.clearTimeout(staleTimer);
      }
      staleTimer = window.setTimeout(() => {
        if (cancelled) return;
        // Sync ``now`` AL MOMENTO de la transición a stale para que el
        // primer render del contador refleje el tiempo real desde
        // lastConfirmedAt (no el ``now`` viejo del montaje del componente).
        setNow(new Date());
        setState(prev =>
          prev.kind === 'success' || prev.kind === 'stale'
            ? { kind: 'stale', data: prev.data }
            : prev,
        );
      }, STALE_TIMEOUT_MS);
    };

    const fetchAndSet = async () => {
      try {
        const data = await controlActiveStateService.getActiveState(nodeId);
        if (cancelled) return;
        setState({ kind: 'success', data });
        setLastConfirmedAt(new Date());
        armStaleTimer();
      } catch {
        if (cancelled) return;
        // Si tenemos data previa, transicionamos a stale → sync ``now``
        // para que el contador arranque en el valor real, no en cero.
        setNow(new Date());
        setState(prev => {
          if (prev.kind === 'success' || prev.kind === 'stale') {
            // Mantenemos la última estrategia conocida (CA-05.4) — no
            // sobrescribimos a 'error' ni vaciamos la pantalla.
            return { kind: 'stale', data: prev.data };
          }
          return {
            kind: 'error',
            message: 'No se pudo cargar la estrategia vigente.',
          };
        });
      }
    };

    void fetchAndSet();

    const controller = openControlActiveStateStream(nodeId, {
      onMessage: () => {
        // Evento es solo señal de cambio; re-leemos REST como fuente
        // autoritativa (DHU-021 #15) y reseteamos el staleTimer.
        void fetchAndSet();
      },
      onError: () => {
        if (cancelled) return;
        // Sync ``now`` al transicionar a stale por caída del stream.
        setNow(new Date());
        setState(prev =>
          prev.kind === 'success' || prev.kind === 'stale'
            ? { kind: 'stale', data: prev.data }
            : prev,
        );
      },
    });

    return () => {
      cancelled = true;
      if (staleTimer !== null) window.clearTimeout(staleTimer);
      controller.abort();
    };
  }, [nodeId]);

  // Tick por segundo SOLO mientras estamos en stale — evita re-renders
  // innecesarios en el flujo normal. setNow desde el callback (no desde
  // el body del effect) es compatible con react-hooks/no-set-state-in-effects.
  useEffect(() => {
    if (state.kind !== 'stale') return;
    const id = window.setInterval(
      () => setNow(new Date()),
      COUNTER_TICK_MS,
    );
    return () => window.clearInterval(id);
  }, [state.kind]);

  if (state.kind === 'loading') {
    return (
      <Card>
        <h2 className="text-white font-semibold text-lg mb-1">
          Estrategia vigente
        </h2>
        <p className="text-sm text-slate-400">Cargando estrategia activa…</p>
      </Card>
    );
  }

  if (state.kind === 'error') {
    return (
      <Card>
        <h2 className="text-white font-semibold text-lg mb-1">
          Estrategia vigente
        </h2>
        <p className="text-sm text-rose-300">{state.message}</p>
      </Card>
    );
  }

  const { data } = state;
  const { label, subtitle } = labelForStrategy(data.strategy_mode);
  const activatedAt = new Date(data.activated_at);
  // ``now`` se actualiza desde callbacks (no en render) — al entrar en
  // stale ya está sincronizado por los setNow de staleTimer / onError /
  // catch del fetch. Cumple CA-05.4: el primer paint del contador
  // refleja el tiempo real desde lastConfirmedAt, sin arrancar en 0.
  const elapsedSinceConfirmation =
    state.kind === 'stale' && lastConfirmedAt !== null
      ? now.getTime() - lastConfirmedAt.getTime()
      : null;

  return (
    <Card>
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-white font-semibold text-lg">{label}</h2>
        {state.kind === 'stale' && (
          <span
            className="text-amber-300 text-xs font-medium uppercase tracking-wide"
            data-testid="active-strategy-stale-marker"
          >
            No confirmada
          </span>
        )}
      </div>
      <p className="text-xs text-slate-400 mb-3">{subtitle}</p>
      <div className="text-xs text-slate-500 mb-2">
        Activa desde {activatedAt.toLocaleString('es-PE')} · ciclo{' '}
        {data.cycle_seconds.toFixed(1)} s
      </div>
      {state.kind === 'stale' && elapsedSinceConfirmation !== null && (
        <p
          className="text-xs text-amber-300/80 mb-3"
          data-testid="active-strategy-stale-elapsed"
        >
          Última confirmación hace {formatElapsed(elapsedSinceConfirmation)}
        </p>
      )}
      <div>
        {data.phase_timings.map(timing => (
          <TimingBar
            key={timing.phase_id}
            timing={timing}
            cycleSeconds={data.cycle_seconds}
            isNext={false}
            subtitle={PHASE_SUBTITLES[timing.phase_id] ?? ''}
          />
        ))}
      </div>
    </Card>
  );
};
