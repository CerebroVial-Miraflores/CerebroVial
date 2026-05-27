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
//    - stale: misma tarjeta + banner "no confirmada" (Fase 6 ampliará la
//      lógica con timer + tiempo desde última confirmación).
//    - error: mensaje neutral; ningún botón de retry hasta Fase 6.
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

export const ActiveStrategyView = ({
  nodeId = DEFAULT_NODE_ID,
}: ActiveStrategyViewProps) => {
  const [state, setState] = useState<ViewState>({ kind: 'loading' });

  useEffect(() => {
    let cancelled = false;

    const fetchAndSet = async () => {
      try {
        const data = await controlActiveStateService.getActiveState(nodeId);
        if (!cancelled) {
          setState({ kind: 'success', data });
        }
      } catch {
        if (!cancelled) {
          // Si ya teníamos datos antes, no perderlos — pasamos a stale.
          // (En Fase 5 mantenemos versión simple; Fase 6 añade timers y
          // banner "tiempo desde última confirmación".)
          setState(prev => {
            if (prev.kind === 'success' || prev.kind === 'stale') {
              return { kind: 'stale', data: prev.data };
            }
            return {
              kind: 'error',
              message: 'No se pudo cargar la estrategia vigente.',
            };
          });
        }
      }
    };

    void fetchAndSet();

    const controller = openControlActiveStateStream(nodeId, {
      onMessage: () => {
        // Evento es solo señal de cambio; re-leemos REST como fuente
        // autoritativa (DHU-021 #15).
        void fetchAndSet();
      },
      onError: () => {
        // Mantiene el último valor conocido como stale si lo teníamos.
        if (cancelled) return;
        setState(prev => {
          if (prev.kind === 'success' || prev.kind === 'stale') {
            return { kind: 'stale', data: prev.data };
          }
          return prev;
        });
      },
    });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [nodeId]);

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
      <div className="text-xs text-slate-500 mb-4">
        Activa desde {activatedAt.toLocaleString('es-PE')} · ciclo{' '}
        {data.cycle_seconds.toFixed(1)} s
      </div>
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
