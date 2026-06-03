/**
 * Helpers puros del mapa de congestión de red (HU-22, Fase 1) — sin DOM.
 *
 * Tres responsabilidades, todas testeables aisladas:
 *  - `congestionStyle`: codificación visual de la escala 0-5 (CA-22.3).
 *  - `mergeCongestion`: cruce geometry × state por `edge_id` (CT-12.8).
 *  - `isStale` / `elapsedSeconds`: detección de "desactualizado" (CA-22.4).
 *
 * NO se extiende `trafficLabels.ts`: aquello es una escala de 3 niveles sobre
 * `mean_occupancy` (visión); esto es la escala ordinal de 6 niveles sobre
 * `congestion_level` (jamLevel de Waze, D-009).
 */
import type {
    GeometryFeatureCollection,
    CongestionStateResponse,
    CongestionSeriesResponse,
    EdgeCongestionState,
    EdgeCongestionSeries,
    MergedCongestionFeature,
} from '../types/congestion';

export interface CongestionStyle {
    color: string;
    weight: number;
}

/**
 * Estilo (color + grosor) por nivel de congestión 0-5, escala aprobada (CA-22.3).
 *
 * El salto de grosor 4 → 6 px en el nivel 3 es DELIBERADO: es la redundancia
 * no-cromática de CA-22.3 — los niveles ≥3 (considerable/alto/cerrado) quedan
 * distinguibles por grosor sin depender solo del color (accesibilidad daltónica).
 */
const STYLE_BY_LEVEL: readonly CongestionStyle[] = [
    { color: '#2ECC71', weight: 3 }, // 0 flujo libre (verde)
    { color: '#A3D83A', weight: 3.5 }, // 1 leve
    { color: '#F4C20D', weight: 4 }, // 2 moderado (amarillo)
    { color: '#F08C1D', weight: 6 }, // 3 considerable (naranja) — salto de grosor
    { color: '#E24B4A', weight: 7.5 }, // 4 alto (rojo)
    { color: '#8E1B1B', weight: 9 }, // 5 vía cerrada (bordó)
];

/** Estilo neutro para "sin dato" / nivel inválido. */
const NEUTRAL_STYLE: CongestionStyle = { color: '#9E9E9E', weight: 3 };

/**
 * Devuelve `{ color, weight }` para un nivel de congestión.
 *
 * Decisión defensiva: `null`/`undefined`/`NaN`/no-entero/fuera de 0-5 → estilo
 * neutro gris. Unifica el caso "arista sin estado" (no-match en `mergeCongestion`)
 * con "nivel inválido del transporte" en un único render neutro. La firma acepta
 * `number | null | undefined` para ser la única puerta de estilo y recibir directo
 * el `congestion_level` de `MergedCongestionFeature`.
 */
export function congestionStyle(
    level: number | null | undefined,
): CongestionStyle {
    if (level == null || !Number.isInteger(level) || level < 0 || level > 5) {
        return NEUTRAL_STYLE;
    }
    return STYLE_BY_LEVEL[level];
}

/**
 * Cruza la geometría (estática) con el estado de congestión por `edge_id`, y
 * devuelve features con `congestion_level` y `snapshot_timestamp` adjuntos.
 *
 * La alineación 375 = 375 = 375 está garantizada por CT-12.8, pero el helper es
 * robusto al edge huérfano: una feature sin estado en `state` recibe
 * `congestion_level: null` y `snapshot_timestamp: null` (no lanza; `congestionStyle`
 * la pinta neutra).
 */
export function mergeCongestion(
    geometry: GeometryFeatureCollection,
    state: CongestionStateResponse,
): MergedCongestionFeature[] {
    const byEdgeId = new Map<string, EdgeCongestionState>(
        state.edges.map((e) => [e.edge_id, e]),
    );
    return geometry.features.map((feature) => {
        const match = byEdgeId.get(feature.properties.edge_id);
        return {
            ...feature,
            properties: {
                ...feature.properties,
                congestion_level: match ? match.congestion_level : null,
                snapshot_timestamp: match ? match.snapshot_timestamp : null,
            },
        };
    });
}

/**
 * Igual que `mergeCongestion`, pero toma el nivel del índice temporal `i` de la
 * serie (`series.edges[].levels[i]`) en vez del último estado — para el recorrido
 * temporal de HU-23. Arista sin serie o índice fuera de rango de `levels` → nivel
 * `null` (estilo neutro vía `congestionStyle`). `snapshot_timestamp` no aplica a la
 * serie (se deriva de `t0 + i*step_s` en la capa de UI), va `null`.
 */
export function mergeCongestionAtIndex(
    geometry: GeometryFeatureCollection,
    series: CongestionSeriesResponse,
    i: number,
): MergedCongestionFeature[] {
    const byEdgeId = new Map<string, EdgeCongestionSeries>(
        series.edges.map((e) => [e.edge_id, e]),
    );
    return geometry.features.map((feature) => {
        const match = byEdgeId.get(feature.properties.edge_id);
        const level =
            match && i >= 0 && i < match.levels.length ? match.levels[i] : null;
        return {
            ...feature,
            properties: {
                ...feature.properties,
                congestion_level: level,
                snapshot_timestamp: null,
            },
        };
    });
}

/**
 * Segundos transcurridos entre `latestTimestamp` y `now`.
 *
 * El backend serializa `datetime` naive en UTC (`ts = DAY_EPOCH + timedelta(...)`,
 * sin tzinfo), p.ej. "2025-01-06T23:59:00". `new Date("...T23:59:00")` lo
 * interpretaría como hora LOCAL: en Lima (UTC-5) el snapshot quedaría ~5 h en el
 * futuro → elapsed negativo → nunca stale. Por eso, si el string no trae designador
 * de zona (`Z` o `±hh:mm`), se normaliza a UTC añadiendo `'Z'` antes de parsear.
 */
export function elapsedSeconds(latestTimestamp: string, now: Date): number {
    const parsedMs = Date.parse(normalizeToUtc(latestTimestamp));
    return (now.getTime() - parsedMs) / 1000;
}

/**
 * `true` si el último snapshot tiene más de `thresholdSec` segundos (CA-22.4).
 *
 * Umbral por defecto 90 s = 1.5× la cadencia de 60 s del feed; configurable.
 */
export function isStale(
    latestTimestamp: string,
    now: Date,
    thresholdSec = 90,
): boolean {
    return elapsedSeconds(latestTimestamp, now) > thresholdSec;
}

/** Detector de zona ISO 8601: `Z` final o un offset `±hh:mm` tras la hora. */
const HAS_TZ = /[zZ]$|[+-]\d{2}:?\d{2}$/;

function normalizeToUtc(timestamp: string): string {
    return HAS_TZ.test(timestamp) ? timestamp : `${timestamp}Z`;
}
