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

// --- Capa de estilo de PREDICCIÓN (Fase 4, escala 0-4 timeLoss/demora) ---
//
// Paralela y SEPARADA del observado (0-5, arriba). Se pinta ENCIMA del observado,
// semitransparente, en paleta fría violeta/azul para leerse a simple vista como una
// capa distinta de la verde→bordó observada. El observado es la capa base intacta.

export interface PredictionStyle {
    color: string;
    weight: number;
    opacity: number;
}

/**
 * Estilo (color + grosor + opacidad) por nivel de congestión PREDICHA 0-4.
 *
 * Gradiente azul→violeta (lavanda claro → violeta profundo), creciente con la
 * congestión predicha — deliberadamente frío para distinguirse de la escala
 * observada verde→bordó cuando se superpone. `opacity` 0.50→0.60 (la predicción
 * severa, más prominente, dentro del rango pedido) para que el observado se lea por
 * debajo. Se conserva la redundancia no-cromática de grosor (creciente, con salto en
 * el tramo alto) como hace el observado, para accesibilidad daltónica.
 */
const STYLE_BY_LEVEL_PREDICTION: readonly PredictionStyle[] = [
    { color: '#C7D2FE', weight: 3, opacity: 0.5 }, // 0 sin demora (lavanda)
    { color: '#A5B4FC', weight: 3.5, opacity: 0.525 }, // 1 demora leve
    { color: '#818CF8', weight: 4.5, opacity: 0.55 }, // 2 demora moderada
    { color: '#6366F1', weight: 6, opacity: 0.575 }, // 3 demora alta (índigo) — salto de grosor
    { color: '#4C1D95', weight: 7.5, opacity: 0.6 }, // 4 demora severa (violeta profundo)
];

/**
 * Neutro de predicción que NO pinta: transparente real (no gris como el observado).
 *
 * A diferencia del `NEUTRAL_STYLE` observado, la predicción se SUPERPONE: un nivel
 * inválido o ausente no debe ensuciar con gris la arista que ya pinta el observado.
 * `opacity: 0` mantiene `predictionStyle` como función total (siempre devuelve un
 * estilo, nunca `null`); Leaflet con `opacity: 0` no dibuja nada visible. Crítico para
 * el horizonte índice 0, donde la predicción no pinta.
 */
const PREDICTION_NEUTRAL_STYLE: PredictionStyle = {
    color: 'transparent',
    weight: 0,
    opacity: 0,
};

/**
 * Devuelve `{ color, weight, opacity }` para un nivel de congestión PREDICHA.
 *
 * Análoga a `congestionStyle` pero en escala 0-4 y con neutro transparente: nivel
 * entero 0-4 → su estilo; `null`/`undefined`/`NaN`/no-entero/fuera de 0-4 →
 * `PREDICTION_NEUTRAL_STYLE` (no pinta). La ausencia de predicción se modela como
 * `null` (p. ej. horizonte índice 0), no como nivel 0: el merge de Gate 3 es quien
 * decide qué nivel/null pasarle.
 */
export function predictionStyle(
    level: number | null | undefined,
): PredictionStyle {
    if (level == null || !Number.isInteger(level) || level < 0 || level > 4) {
        return PREDICTION_NEUTRAL_STYLE;
    }
    return STYLE_BY_LEVEL_PREDICTION[level];
}

/** Niveles 0-4 de la escala de predicción, para iterar en la leyenda (cableada en Gate 3). */
export const LEVELS_PREDICTION = [0, 1, 2, 3, 4] as const;

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
