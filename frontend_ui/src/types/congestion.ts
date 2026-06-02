/**
 * Tipos del mapa de congestión de red (HU-22, Fase 1).
 *
 * Espejan los schemas Pydantic de `core_management_api/.../congestion/.../schemas.py`
 * (contrato en `documentation/contracts/congestion_contract.md`). En el backend
 * `geometry`/`properties` son `dict` sueltos; acá los tipamos estrictos según el
 * shape documentado del contrato (GeoJSON estándar).
 */

// --- GET /congestion/geometry (estática, 375 aristas de la LCC, CT-12.2) ---

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
