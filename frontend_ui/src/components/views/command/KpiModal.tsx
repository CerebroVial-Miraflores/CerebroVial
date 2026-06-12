// FASE 3 (B3) — modal de detalle por KPI (spec: .modal/#bigChart del
// prototipo). Política por card (decisión D2 del plan):
// · idx — serie REAL derivada de useCongestionSeries (día en contexto: ?dia en
//   histórico, hoy en los demás modos). Día vacío → vacío honesto. Forecast
//   omitido con nota mientras /congestion/prediction responda 503.
// · vel / flu / dem — serie MOCK + DemoBadge en el subtítulo (sin fuente real
//   de serie de red; la historia por cámara de /predictions/history no es
//   fuente directa del agregado).
// · sem — lista REAL-PARCIAL de los 5 nodos instrumentados con su estrategia;
//   SIN botón "Reactivar" (acción demo sobre datos reales confunde la
//   señalización — desvío consciente del prototipo).
// · pred — NO llega acá: la card cambia el mapa a modo predicción.
import { BigChart } from '../../ui/BigChart';
import { Button } from '../../ui/Button';
import { DemoBadge } from '../../ui/DemoBadge';
import { Modal } from '../../ui/Modal';
import { MODE_LABEL } from '../control/controlTypes';
import type { UseAdaptiveNodesResult } from './useAdaptiveNodes';
import { MODAL_SERIES_MOCK } from './mockData';
import type { KpiKind } from './KpiStrip';

const TITLES: Record<KpiKind, string> = {
  idx: 'Índice de congestión · red',
  vel: 'Velocidad media de la red',
  dem: 'Demora media · cruces críticos',
  flu: 'Flujo total de la red',
  sem: 'Cruces en modo adaptativo',
};

const MOCK_COLOR: Record<'vel' | 'flu' | 'dem', string> = {
  vel: 'text-bad',
  flu: 'text-brand',
  dem: 'text-warn',
};

export interface KpiModalIdxData {
  /** Serie real del índice (derive.seriesToNetworkIndex); null = día sin datos. */
  series: { points: number[]; tLabels: string[] } | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  day: string;
  /** true mientras la predicción esté caída (503) → el forecast se omite con nota. */
  predictionUnavailable: boolean;
}

interface KpiModalProps {
  kind: KpiKind | null;
  onClose: () => void;
  idx: KpiModalIdxData;
  adaptive: UseAdaptiveNodesResult;
}

/** 'larco_schell' → 'Larco × Schell' (cosmético; el id real va al lado). */
function prettyNodeName(nodeId: string): string {
  return nodeId
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' × ');
}

/** Submuestreo uniforme para el chart (1440 pasos/día → ≤144 puntos). */
function downsample(points: number[], tLabels: string[], cap = 144) {
  if (points.length <= cap) return { points, tLabels };
  const stride = Math.ceil(points.length / cap);
  const sampledPoints: number[] = [];
  const sampledLabels: string[] = [];
  for (let i = 0; i < points.length; i += stride) {
    sampledPoints.push(points[i]);
    sampledLabels.push(tLabels[i]);
  }
  return { points: sampledPoints, tLabels: sampledLabels };
}

function axisLabels(tLabels: string[], count = 5): string[] {
  if (tLabels.length < 2) return tLabels;
  return Array.from(
    { length: count },
    (_, i) => tLabels[Math.round((i / (count - 1)) * (tLabels.length - 1))],
  );
}

function fmt(value: number, decimals: number): string {
  return value.toFixed(decimals);
}

function MStats({ items }: { items: { label: string; value: string }[] }) {
  return (
    <div className="mt-2.5 flex gap-6 border-t border-line pt-2.5">
      {items.map((item) => (
        <div key={item.label}>
          <span className="block text-[9.5px] font-bold uppercase tracking-[0.1em] text-ink-2">
            {item.label}
          </span>
          <b className="num text-[15px]">{item.value}</b>
        </div>
      ))}
    </div>
  );
}

export function KpiModal({ kind, onClose, idx, adaptive }: KpiModalProps) {
  if (kind === null) return null;

  // ── sem: lista real-parcial, sin chart ────────────────────────────────────
  if (kind === 'sem') {
    return (
      <Modal
        open
        onClose={onClose}
        title={TITLES.sem}
        subtitle="Estrategia vigente por nodo instrumentado (consulta real al plano de control)"
      >
        <div className="flex flex-col gap-2">
          {adaptive.loading && <p className="text-[11.5px] text-ink-2">Consultando nodos…</p>}
          {adaptive.nodes?.map((node) => (
            <div
              key={node.nodeId}
              className="flex items-center gap-3 rounded-[11px] border border-line bg-white/2 px-3 py-2.5 text-[12.5px]"
            >
              <div className="min-w-0 flex-1">
                <b>{prettyNodeName(node.nodeId)}</b>
                <span className="num ml-2 text-[10.5px] text-ink-3">{node.nodeId}</span>
              </div>
              {node.kind === 'active' && node.state !== null ? (
                <span className="text-[11px] font-bold text-ok">
                  {MODE_LABEL[node.state.strategy_mode]} · {node.state.cycle_seconds}s
                </span>
              ) : node.kind === 'no-strategy' ? (
                <span className="text-[11px] text-ink-2">Sin estrategia activa</span>
              ) : (
                <span className="text-[11px] text-warn">Sin dato (error al consultar)</span>
              )}
            </div>
          ))}
        </div>
        {/* Sin botón "Reactivar" (D2): la lista es real; una acción demo acá
            rompería la señalización de paridad. */}
      </Modal>
    );
  }

  // ── idx: serie REAL del día en contexto ───────────────────────────────────
  if (kind === 'idx') {
    const sampled =
      idx.series !== null ? downsample(idx.series.points, idx.series.tLabels) : null;
    return (
      <Modal
        open
        onClose={onClose}
        title={TITLES.idx}
        subtitle={
          <>
            Día {idx.day} · serie real (media de jam_level /100)
            {idx.predictionUnavailable &&
              ' · sin punto de predicción (servicio no disponible)'}
          </>
        }
      >
        {idx.loading ? (
          <p className="py-10 text-center text-[12.5px] text-ink-2" role="status">
            Cargando serie del día…
          </p>
        ) : idx.error !== null ? (
          <div className="flex flex-col items-start gap-3 py-6">
            <p className="text-[12.5px] text-bad">{idx.error}</p>
            <Button onClick={idx.onRetry}>Reintentar</Button>
          </div>
        ) : sampled === null || sampled.points.length < 2 ? (
          <p className="py-10 text-center text-[12.5px] text-ink-2">
            El día {idx.day} no tiene datos de congestión — sin serie que graficar.
          </p>
        ) : (
          <>
            <BigChart
              series={sampled.points}
              xLabels={axisLabels(sampled.tLabels)}
              className="my-3 h-[248px] w-full text-warn"
            />
            <MStats
              items={[
                { label: 'Actual', value: `${sampled.points.at(-1)} /100` },
                { label: 'Pico del día', value: `${Math.max(...sampled.points)} /100` },
                { label: 'Predicción', value: '—' },
              ]}
            />
          </>
        )}
      </Modal>
    );
  }

  // ── vel / flu / dem: serie MOCK señalizada ────────────────────────────────
  const mock = MODAL_SERIES_MOCK[kind];
  const decimals = kind === 'vel' ? 1 : 0;
  const unit = kind === 'vel' ? 'km/h' : kind === 'flu' ? 'veh/h' : 's/veh';
  return (
    <Modal
      open
      onClose={onClose}
      title={TITLES[kind]}
      subtitle={
        <span className="flex flex-wrap items-center gap-2">
          Últimas 3 h · línea punteada = predicción del modelo
          <DemoBadge />
        </span>
      }
    >
      <BigChart
        series={mock.series}
        forecast={mock.forecast}
        xLabels={['−3h', '−2h', '−1h', 'ahora', '+45m']}
        className={`my-3 h-[248px] w-full ${MOCK_COLOR[kind]}`}
      />
      <MStats
        items={[
          { label: 'Actual', value: `${fmt(mock.series.at(-1)!, decimals)} ${unit}` },
          { label: 'Pico del día', value: `${fmt(Math.max(...mock.series), decimals)} ${unit}` },
          { label: 'Predicción +45 min', value: `${fmt(mock.forecast.at(-1)!, decimals)} ${unit}` },
        ]}
      />
    </Modal>
  );
}
