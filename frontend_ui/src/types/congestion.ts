/**
 * Tipos del mapa de congestión de red (HU-22, Fase 1).
 *
 * Espejan los schemas Pydantic de `core_management_api/.../congestion/.../schemas.py`
 * (contrato en `documentation/contracts/congestion_contract.md`). En el backend
 * `geometry`/`properties` son `dict` sueltos; acá los tipamos estrictos según el
 * shape documentado del contrato (GeoJSON estándar).
 */

// --- GET /congestion/geometry (estática, 1660 aristas de la LCC, CT-12.2) ---

export interface EdgeProperties {
  edge_id: string;
  source_node: string;
  target_node: string;
  distance_m: number; // float (metros)
  lanes: number; // int (nº de carriles)
}

export interface GeometryFeature {
  type: 'Feature';
  // GeoJSON LineString: coordenadas en orden [lon, lat] (NO lat,lon), CRS EPSG:4326.
  geometry: { type: 'LineString'; coordinates: [number, number][] };
  properties: EdgeProperties;
}

export interface GeometryFeatureCollection {
  type: 'FeatureCollection';
  features: GeometryFeature[];
  count: number;
}

// --- GET /congestion/state (último snapshot por arista, CT-12.6/12.7) ---

export interface EdgeCongestionState {
  edge_id: string;
  congestion_level: number; // int 0-5 (D-009, escala jamLevel de Waze)
  snapshot_timestamp: string; // ISO 8601 naive en UTC, p.ej. "2025-01-06T23:59:00"
}

export interface CongestionStateResponse {
  edges: EdgeCongestionState[];
  count: number;
}

// --- Salida de mergeCongestion (lista para una capa GeoJSON de Leaflet en Fase 2) ---

/**
 * Feature de geometría con el estado de congestión adjunto por `edge_id`.
 * `congestion_level`/`snapshot_timestamp` son `null` cuando la arista no tiene
 * estado en el snapshot (caso defensivo; ver `mergeCongestion`).
 */
export interface MergedCongestionFeature extends GeometryFeature {
  properties: EdgeProperties & {
    congestion_level: number | null;
    snapshot_timestamp: string | null;
  };
}

// --- GET /congestion/series?day= (serie temporal del día por arista, TTH-13) ---

export interface EdgeCongestionSeries {
  edge_id: string;
  levels: number[]; // levels[i] = nivel 0-5 en t0 + i*step_s
}

export interface CongestionSeriesResponse {
  day: string; // "YYYY-MM-DD"
  t0: string | null; // ISO 8601, null si el día no tiene datos
  step_s: number | null; // segundos entre muestras (null si día vacío)
  coverage_end: string | null; // ISO 8601, null si día vacío
  count: number;
  edges: EdgeCongestionSeries[];
}

// --- GET /congestion/prediction (predicción GRU baseline por arista, Fase 3) ---

export interface PredictedEdgeCongestion {
  edge_id: string;
  levels: number[]; // levels[i] = nivel predicho 0-4 en base_timestep + 1 + i (horizon=30)
}

export interface CongestionPredictionResponse {
  base_timestep: number; // corte t (minuto del día 0..1439); ventana t-29..t
  horizon: number; // nº de pasos futuros predichos (=30)
  source: string; // día-fuente, p.ej. "seed051 (day_idx=9)"
  source_date: string; // "YYYY-MM-DD" — fecha-calendario del día-fuente (getSeries en modo pinned, Gate 3b)
  count: number;
  edges: PredictedEdgeCongestion[];
}

// --- Contrato estructural mínimo para el cruce por índice temporal ---

/**
 * Fuente de niveles por arista indexados posicionalmente: lo único que
 * `mergeCongestionAtIndex` consume (no `t0`/`step_s`/`day`/`base_timestep`).
 *
 * Tanto `CongestionSeriesResponse` (recorrido histórico, HU-23) como
 * `CongestionPredictionResponse` (predicción servida, Fase 4) lo satisfacen
 * estructuralmente, así que el mismo helper cruza ambas sin duplicarse. Es el
 * contrato del PARÁMETRO del helper; cada caller sigue pasando su tipo completo.
 */
export interface IndexedEdgeLevels {
  edges: { edge_id: string; levels: number[] }[];
}
