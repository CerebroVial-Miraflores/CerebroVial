import { Clock } from 'lucide-react';

import { usePredictionHistory } from '../../../hooks/usePredictionHistory';
import type { PredictionHistoryInterval } from '../../../types/predictionHistory';
import { BigChart } from '../../ui/BigChart';
import { SegmentedControl } from '../../ui/SegmentedControl';
import { StatusChip, type Status } from '../../ui/StatusChip';

// FASE 4 rediseño UI — historia y predicción del detalle de cámara (re-skin del
// contenido de TrafficHistoryWidget v1 sobre el design system). Datos reales de
// GET /predictions/history (modelo servido) vía usePredictionHistory + BigChart.
//
// Simplificación consciente vs v1 (flag del plan F4): BigChart es mono-serie
// (vehículos: histórico sólido + forecast punteado, divisor "ahora" nativo). La
// doble serie de congestión del widget recharts (eje derecho 0-2) NO migra al
// chart: el nivel de congestión por horizonte se muestra como chips +15/+30/+45,
// que es la lectura que importa. Se gana coherencia de tokens y muere recharts
// en esta vista.

interface CameraHistoryPanelProps {
  cameraId: string;
  interval: PredictionHistoryInterval;
  onIntervalChange: (interval: PredictionHistoryInterval) => void;
}

const INTERVAL_OPTIONS: readonly { value: PredictionHistoryInterval; label: string }[] = [
  { value: 1, label: '1m' },
  { value: 2, label: '2m' },
  { value: 5, label: '5m' },
  { value: 10, label: '10m' },
  { value: 15, label: '15m' },
];

/** Normaliza el congestion_level crudo del backend (ES/EN mezclados) a label + estado. */
function normalizeCongestion(level: string): { label: string; status: Status } {
  const l = (level || '').toLowerCase();
  if (l === 'heavy' || l === 'alto' || l === 'severo') return { label: 'Alto', status: 'bad' };
  if (l === 'high' || l === 'moderado' || l === 'moderate' || l === 'medio')
    return { label: 'Moderado', status: 'warn' };
  return { label: 'Bajo', status: 'ok' };
}

export function CameraHistoryPanel({ cameraId, interval, onIntervalChange }: CameraHistoryPanelProps) {
  const { data, loading, error, refetch } = usePredictionHistory(cameraId, interval, {
    intervalMs: 60_000,
  });

  const history = data?.history ?? [];
  const realPoints = history.filter((h) => !h.is_prediction);
  const series = realPoints.map((h) => h.total_vehicles);

  const pred = data?.prediction;
  const forecast = pred
    ? [pred.predicted_vehicles_15min, pred.predicted_vehicles_30min, pred.predicted_vehicles_45min].filter(
        (v): v is number => typeof v === 'number',
      )
    : [];

  const horizons = pred
    ? [
        { t: '+15 min', ...normalizeCongestion(pred.predicted_congestion_15min) },
        { t: '+30 min', ...normalizeCongestion(pred.predicted_congestion_30min) },
        { t: '+45 min', ...normalizeCongestion(pred.predicted_congestion_45min) },
      ]
    : [];

  return (
    <section className="rounded-panel border border-line bg-linear-to-b from-white/4 to-white/1 p-3.5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Clock size={16} className="text-accent" aria-hidden="true" />
          <h3 className="text-[13px] font-bold">Historial y predicción</h3>
        </div>
        <SegmentedControl
          ariaLabel="Intervalo de agregación"
          options={INTERVAL_OPTIONS}
          value={interval}
          onChange={(v) => onIntervalChange(v as PredictionHistoryInterval)}
        />
      </div>

      <p className="mb-3 text-[11px] text-ink-2">
        Vehículos detectados — <span className="font-semibold text-accent">histórico</span> vs{' '}
        <span className="font-semibold text-accent">predicción</span> del modelo servido.
      </p>

      {loading && series.length === 0 ? (
        <div className="grid h-[200px] place-items-center text-[12.5px] text-ink-2">
          Cargando datos…
        </div>
      ) : error !== null ? (
        <div className="flex flex-wrap items-center gap-3 text-[12.5px]">
          <span className="text-bad">{error}</span>
          <button
            type="button"
            onClick={() => void refetch()}
            className="rounded-btn border border-line bg-panel px-3 py-[7px] text-[11.5px] font-semibold text-ink hover:border-line-2"
          >
            Reintentar
          </button>
        </div>
      ) : series.length < 2 ? (
        <div className="grid h-[200px] place-items-center text-center text-[12.5px] text-ink-2">
          Aún no hay suficiente historial para esta cámara.
        </div>
      ) : (
        <>
          <BigChart
            series={series}
            forecast={forecast}
            className="h-[220px] w-full text-accent"
          />
          {horizons.length > 0 && (
            <div className="mt-3 grid grid-cols-3 gap-2">
              {horizons.map((h) => (
                <div
                  key={h.t}
                  className="flex flex-col items-center gap-1.5 rounded-ctl border border-line bg-white/2 px-2 py-2.5"
                >
                  <span className="text-[10px] text-ink-2">{h.t}</span>
                  <StatusChip status={h.status}>{h.label.toUpperCase()}</StatusChip>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
