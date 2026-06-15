// FASE 3 rediseño UI — agregador de visión multi-cámara para los KPIs de red
// del Centro de Comando (flujo total, velocidad media) y las métricas del
// drawer de intersección.
//
// UN effect abre N EventSource nativos (espejo de useVisionStream, patrón que
// v1 DashboardView ya usaba con su array de sources). Elegido sobre
// "componentes probe" invisibles: evita N componentes fantasma con callbacks
// al padre, y `byCamera` alimenta también al drawer (D3.2b prohíbe visión
// on-demand ahí — esta es su única fuente legítima).
//
// ── POLÍTICA REAL-CON-CAVEAT (Fase 3) ──────────────────────────────────────
// Ambas métricas son REALES pero limitadas; la UI que las muestre lleva el
// caveat textual (sin DemoBadge — no son mock):
// · `mean_speed_kmh`: velocidad experimental SIN CALIBRAR (DEUDA-SPEED-CALIB).
//   Caveat UI: "visión · sin calibrar".
// · `flow_vehicles_per_hour`: métrica de PRESENCIA extrapolada, no flujo por
//   line-crossing (resultado negativo del spike de flujo: conteo por cruce de
//   línea estructuralmente imposible a 1 Hz). Caveat UI:
//   "visión · presencia extrapolada".
// ───────────────────────────────────────────────────────────────────────────
//
// DEUDA (F4): sin TTL de muestra — una cámara que deja de emitir conserva su
// último valor en el agregado (paridad con v1). DEUDA BACKEND heredada de
// useVisionStream: el edge no exige JWT.
import { useEffect, useMemo, useState } from 'react';
import { EDGE_API_URL } from '../config/edge';
import type { VisionStreamPayload } from '../types/visionStream';

export interface VisionSample {
  /** km/h experimental sin calibrar; null si la cámara carece de calibración. */
  speed: number | null;
  /** veh/h por presencia extrapolada (NO line-crossing). */
  flow: number;
  /** Epoch ms de la muestra. */
  at: number;
}

export interface VisionAggregate {
  /** Media de `speed` sobre cámaras con muestra y speed no-null; null sin ninguna. */
  meanSpeedKmh: number | null;
  /** Suma de `flow` sobre cámaras con muestra; null sin ninguna. */
  totalFlowVph: number | null;
  camerasWithSignal: number;
}

/** Agregación pura (exportada para test directo y reuso en derivaciones). */
export function aggregateVisionSamples(
  byCamera: Readonly<Record<string, VisionSample>>,
): VisionAggregate {
  const samples = Object.values(byCamera);
  if (samples.length === 0) {
    return { meanSpeedKmh: null, totalFlowVph: null, camerasWithSignal: 0 };
  }
  const speeds = samples
    .map((s) => s.speed)
    .filter((v): v is number => v !== null);
  const meanSpeedKmh =
    speeds.length > 0
      ? speeds.reduce((acc, v) => acc + v, 0) / speeds.length
      : null;
  const totalFlowVph = samples.reduce((acc, s) => acc + s.flow, 0);
  return { meanSpeedKmh, totalFlowVph, camerasWithSignal: samples.length };
}

export interface UseVisionAggregatesResult {
  byCamera: Readonly<Record<string, VisionSample>>;
  aggregate: VisionAggregate;
}

export function useVisionAggregates(
  cameraIds: readonly string[],
): UseVisionAggregatesResult {
  const key = cameraIds.join('|');
  // Identidad estable por contenido: el caller puede pasar un array nuevo por
  // render sin reabrir streams.
  const ids = useMemo(() => key.split('|').filter(Boolean), [key]);

  const [byCamera, setByCamera] = useState<Record<string, VisionSample>>({});

  // Render-guard (patrón useVisionStream): cambio del set de cámaras = streams
  // nuevos → muestras del set anterior fuera en el mismo render.
  const [prevKey, setPrevKey] = useState(key);
  if (prevKey !== key) {
    setPrevKey(key);
    setByCamera({});
  }

  useEffect(() => {
    if (ids.length === 0) return;
    const sources = ids.map((cameraId) => {
      const source = new EventSource(`${EDGE_API_URL}/stream/${cameraId}`);
      const onTrafficUpdate = (event: MessageEvent) => {
        try {
          const payload = JSON.parse(event.data as string) as VisionStreamPayload;
          const m = payload.metrics;
          setByCamera((prev) => ({
            ...prev,
            [cameraId]: {
              speed: m.mean_speed_kmh,
              flow: m.flow_vehicles_per_hour,
              at: Date.now(),
            },
          }));
        } catch {
          // Payload malformado: se ignora, la muestra previa queda.
        }
      };
      source.addEventListener('traffic_update', onTrafficUpdate);
      // Sin onerror custom: EventSource nativo auto-reintenta; el estado de
      // conexión por cámara no se expone (el agregado refleja "sin señal"
      // simplemente por ausencia de muestras).
      return source;
    });

    return () => {
      for (const source of sources) source.close();
    };
  }, [ids]);

  const aggregate = useMemo(() => aggregateVisionSamples(byCamera), [byCamera]);

  return { byCamera, aggregate };
}
