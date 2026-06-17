// FASE 3 rediseño UI — derivaciones puras del Centro de Comando.
//
// Sin React, sin DOM: todo lo que transforma datos de hooks en props de UI
// vive acá con test directo (derive.test.ts). Las fórmulas de los KPIs y la
// key del GeoJSON son CONTRATO (los tests de CommandView las asertan).
import type {
  CongestionPredictionResponse,
  CongestionSeriesResponse,
  CongestionStateResponse,
  GeometryFeatureCollection,
  MergedCongestionFeature,
} from '../../../types/congestion';
import type { IntersectionSummary } from '../../../types/intersections';
import type { ActiveStateResponse } from '../../../services/controlActiveStateService';
import type { ControlRecommendation } from '../../../services/controlService';
import type { Status } from '../../ui/StatusChip';
import type { NodeStatus } from '../../map/nodeIcon';
import {
  JAM_LEVEL_LEGEND,
  PREDICTION_LEGEND,
} from '../../map/edgeStyle';
import {
  LEVEL_LABEL_PREDICTION,
  mergeCongestion,
  mergeCongestionAtIndex,
} from '../../../utils/congestion';

export type CommandMode = 'ahora' | 'historico' | 'prediccion';
export type PredictionHorizon = 15 | 30;

/**
 * KPI índice de congestión de red: media de congestion_level (0-5) sobre las
 * aristas presentes en el estado, escalada a 0-100 y redondeada. Sin aristas
 * (ventana vacía) → null → la card muestra "—" + "sin datos en ventana".
 */
export function networkCongestionIndex(
  state: CongestionStateResponse | null,
): number | null {
  if (!state || state.edges.length === 0) return null;
  const sum = state.edges.reduce((acc, e) => acc + e.congestion_level, 0);
  return Math.round((sum / state.edges.length / 5) * 100);
}

/**
 * Estado del marker desde el `status` real del REST (fluid|moderate|critical,
 * agregado Waze MAX por intersección). Valor desconocido → 'ok' (no alarmar
 * sin dato).
 */
export function nodeStatusFor(status: IntersectionSummary['status']): NodeStatus {
  if (status === 'critical') return 'bad';
  if (status === 'moderate') return 'warn';
  return 'ok';
}

export interface CommandMarker {
  id: string;
  name: string;
  position: [number, number];
  status: NodeStatus;
  critical: boolean;
}

export function markersFrom(list: IntersectionSummary[] | null): CommandMarker[] {
  if (!list) return [];
  return list.map((i) => ({
    id: i.id,
    name: i.name,
    position: [i.lat, i.lng],
    status: nodeStatusFor(i.status),
    critical: i.status === 'critical',
  }));
}

/** Clamp de índice de timestep: fuera de rango → [0, len-1]; len 0 → 0. */
export function clampIndex(i: number, len: number): number {
  if (len <= 0) return 0;
  if (!Number.isFinite(i) || i < 0) return 0;
  return Math.min(Math.floor(i), len - 1);
}

/**
 * KPI predicción de red: media del nivel predicho (0-4) en el paso del
 * horizonte (+15 → índice 14, +30 → 29, clampeado al horizon real) →
 * etiqueta semántica de LEVEL_LABEL_PREDICTION. Sin data/aristas → null.
 */
export function predictionLabelAt(
  p: CongestionPredictionResponse | null,
  offsetMin: PredictionHorizon,
): string | null {
  if (!p || p.edges.length === 0) return null;
  const len = p.edges[0]?.levels.length ?? 0;
  if (len === 0) return null;
  const idx = clampIndex(offsetMin - 1, len);
  let sum = 0;
  let count = 0;
  for (const edge of p.edges) {
    const level = edge.levels[idx];
    if (level !== undefined) {
      sum += level;
      count += 1;
    }
  }
  if (count === 0) return null;
  return LEVEL_LABEL_PREDICTION[Math.round(sum / count)] ?? null;
}

/**
 * Features a pintar según el modo activo. null = el modo aún no tiene datos
 * (loading/empty/error: lo comunica el panel de estado del mapa, no la capa).
 */
export function featuresForMode(args: {
  mode: CommandMode;
  geometry: GeometryFeatureCollection | null;
  state: CongestionStateResponse | null;
  series: CongestionSeriesResponse | null;
  tClamped: number;
  prediction: CongestionPredictionResponse | null;
  horizon: PredictionHorizon;
}): MergedCongestionFeature[] | null {
  const { mode, geometry, state, series, tClamped, prediction, horizon } = args;
  if (!geometry) return null;
  if (mode === 'ahora') {
    if (!state) return null;
    return mergeCongestion(geometry, state);
  }
  if (mode === 'historico') {
    if (!series || series.t0 === null) return null;
    return mergeCongestionAtIndex(geometry, series, tClamped);
  }
  // prediccion
  if (!prediction || prediction.edges.length === 0) return null;
  const len = prediction.edges[0]?.levels.length ?? 0;
  if (len === 0) return null;
  return mergeCongestionAtIndex(geometry, prediction, clampIndex(horizon - 1, len));
}

/** Hoy en YYYY-MM-DD LOCAL (sin bias UTC). Migrado de CongestionMapView (v1). */
export function todayIsoLocal(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

/** Marcas de hora orientativas sobre la pista del slider (cada 3 h, 0..24). */
export const HOUR_MARKS = [0, 3, 6, 9, 12, 15, 18, 21, 24] as const;

/**
 * Label hh:mm del timestep i (contrato de series v1: t0 + i*step_s). Parsea t0
 * naive como UTC (el backend escribe snapshot_timestamp naive y deriva el
 * corte en UTC). Migrado tal cual de CongestionMapView (HU-23).
 */
export function formatHourLabel(t0: string | null, stepS: number, i: number): string {
  if (!t0) return '--:--';
  const base = Date.parse(/[zZ]$|[+-]\d{2}:?\d{2}$/.test(t0) ? t0 : `${t0}Z`);
  const d = new Date(base + i * stepS * 1000);
  const hh = String(d.getUTCHours()).padStart(2, '0');
  const mm = String(d.getUTCMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

/**
 * Serie del índice de red por timestep para el KPI modal: media de levels[i]
 * sobre aristas, escalada /100. Día sin datos (t0 null o sin aristas) → null.
 */
export function seriesToNetworkIndex(
  s: CongestionSeriesResponse | null,
): { points: number[]; tLabels: string[] } | null {
  if (!s || s.t0 === null || s.edges.length === 0) return null;
  const len = s.edges[0]?.levels.length ?? 0;
  if (len === 0) return null;
  const points: number[] = [];
  const tLabels: string[] = [];
  const stepS = s.step_s ?? 60;
  for (let i = 0; i < len; i += 1) {
    let sum = 0;
    let count = 0;
    for (const edge of s.edges) {
      const level = edge.levels[i];
      if (level !== undefined) {
        sum += level;
        count += 1;
      }
    }
    points.push(count > 0 ? Math.round((sum / count / 5) * 100) : 0);
    tLabels.push(formatHourLabel(s.t0, stepS, i));
  }
  return { points, tLabels };
}

/**
 * Nivel real del tramo que toca el nodo: MAX congestion_level entre las
 * aristas cuya source_node/target_node coincide con nodeId y tienen dato en
 * el estado. Sin coincidencia o sin dato → null (el drawer omite la stat).
 */
export function edgeLevelForNode(
  geometry: GeometryFeatureCollection | null,
  state: CongestionStateResponse | null,
  nodeId: string,
): number | null {
  if (!geometry || !state || state.edges.length === 0 || nodeId === '') return null;
  const levelByEdge = new Map(state.edges.map((e) => [e.edge_id, e.congestion_level]));
  let max: number | null = null;
  for (const feature of geometry.features) {
    const { edge_id, source_node, target_node } = feature.properties;
    if (source_node !== nodeId && target_node !== nodeId) continue;
    const level = levelByEdge.get(edge_id);
    if (level === undefined) continue;
    if (max === null || level > max) max = level;
  }
  return max;
}

/**
 * Adapter HU-05 → contrato de TrafficLightCycle (REUSADO TAL CUAL): el ciclo
 * activo real de /control/active-state proyectado al shape que el componente
 * del playground ya sabe animar.
 */
export function adaptActiveState(s: ActiveStateResponse): ControlRecommendation {
  return {
    intersection_id: s.node_id,
    mode: s.strategy_mode,
    cycle_seconds: s.cycle_seconds,
    phase_timings: s.phase_timings,
    next_phase: null,
    reasoning: '',
    adjustments: [],
  };
}

/**
 * Chip del header del drawer. appliedDemo = el operador "aplicó" la
 * recomendación mock (estado local de demo) → EN RECUPERACIÓN.
 */
export function drawerStatusFor(
  status: IntersectionSummary['status'],
  appliedDemo: boolean,
): { status: Status; label: string } {
  if (appliedDemo) return { status: 'recovery', label: 'EN RECUPERACIÓN' };
  if (status === 'critical') return { status: 'bad', label: 'CRÍTICO · ADAPTATIVO EN PAUSA' };
  if (status === 'moderate') return { status: 'warn', label: 'EN OBSERVACIÓN' };
  return { status: 'ok', label: 'ADAPTATIVO · NORMAL' };
}

/**
 * Buffer inmutable del sparkline de la strip: appendea la muestra no-null y
 * recorta a `cap`. null no muta (la línea no inventa ceros sin señal).
 */
export function pushSample(
  history: readonly number[],
  v: number | null,
  cap = 24,
): readonly number[] {
  if (v === null) return history;
  const next = [...history, v];
  return next.length > cap ? next.slice(next.length - cap) : next;
}

/** Leyenda y título del mapa por modo (consume las escalas de edgeStyle). */
export function legendForMode(
  mode: CommandMode,
  tLabel: string,
): { title: string; items: readonly { label: string; swatchClass: string }[] } {
  if (mode === 'prediccion') return { title: 'Demora prevista', items: PREDICTION_LEGEND };
  if (mode === 'historico') return { title: `Histórico · ${tLabel}`, items: JAM_LEVEL_LEGEND };
  return { title: 'Observado', items: JAM_LEVEL_LEGEND };
}
