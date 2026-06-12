/**
 * Helpers puros de datos de congestión (HU-22/HU-23) — sin DOM.
 *
 * Responsabilidades, todas testeables aisladas:
 *  - `mergeCongestion` / `mergeCongestionAtIndex`: cruce geometry × niveles
 *    por `edge_id` (CT-12.8 / HU-23).
 *  - `isStale` / `elapsedSeconds`: detección de "desactualizado" (CA-22.4).
 *  - `LEVEL_LABEL_PREDICTION`: semántica de la escala de demora predicha.
 *
 * FASE 3 rediseño UI: murieron `congestionStyle`/`predictionStyle` y sus
 * paletas hex (consumidor único: CongestionMapView v1, eliminada). El estilo
 * de tramos vive SOLO en components/map/edgeStyle.ts (tokens, sin hex).
 *
 * NO se extiende `trafficLabels.ts`: aquello es una escala de 3 niveles sobre
 * `mean_occupancy` (visión); esto es la escala ordinal de 6 niveles sobre
 * `congestion_level` (jamLevel de Waze, D-009).
 */
import type {
    GeometryFeatureCollection,
    CongestionStateResponse,
    EdgeCongestionState,
    IndexedEdgeLevels,
    MergedCongestionFeature,
} from '../types/congestion';

/**
 * Etiquetas de la leyenda de predicción: semántica de timeLoss/demora (NO de velocidad
 * como el observado). La predicción comunica demora esperada, no estado de flujo.
 */
export const LEVEL_LABEL_PREDICTION: Record<number, string> = {
    0: 'Sin demora',
    1: 'Demora leve',
    2: 'Demora moderada',
    3: 'Demora alta',
    4: 'Demora severa',
};

/**
 * Cruza la geometría (estática) con el estado de congestión por `edge_id`, y
 * devuelve features con `congestion_level` y `snapshot_timestamp` adjuntos.
 *
 * La alineación 1660 = 1660 = 1660 está garantizada por CT-12.8, pero el helper es
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
 * fuente (`source.edges[].levels[i]`) en vez del último estado — para el recorrido
 * temporal de HU-23 y la predicción servida de Fase 4. Arista sin entrada o índice
 * fuera de rango de `levels` → nivel `null` (estilo neutro). `snapshot_timestamp` no
 * aplica (se deriva en la capa de UI), va `null`.
 *
 * El parámetro se tipa con el contrato estructural mínimo `IndexedEdgeLevels`, que
 * tanto `CongestionSeriesResponse` (histórico) como `CongestionPredictionResponse`
 * (predicción) satisfacen — el mismo helper cruza ambas sin duplicarse.
 */
export function mergeCongestionAtIndex(
    geometry: GeometryFeatureCollection,
    source: IndexedEdgeLevels,
    i: number,
): MergedCongestionFeature[] {
    const byEdgeId = new Map<string, IndexedEdgeLevels['edges'][number]>(
        source.edges.map((e) => [e.edge_id, e]),
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
