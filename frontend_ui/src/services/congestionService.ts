// HU-22 — Cliente REST de los endpoints de congestión de red (Fase 1).
//
// Reusa el httpClient compartido (interceptor JWT de HU-01: inyecta
// Authorization: Bearer y maneja 401 → authBridge.onUnauthorized). Los endpoints
// /congestion/* están protegidos con require_role(OPERATOR, ADMIN), por lo que
// NO se usa fetch crudo — el token va por el interceptor.
import { httpClient } from './httpClient';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
  CongestionSeriesResponse,
  CongestionPredictionResponse,
} from '../types/congestion';

export const congestionService = {
  /** Geometría de la red (estática, 1660 aristas). Se consume 1× al montar el mapa. */
  async getGeometry(): Promise<GeometryFeatureCollection> {
    const res = await httpClient.get<GeometryFeatureCollection>(
      '/congestion/geometry',
    );
    return res.data;
  },

  /** Último estado de congestión por arista. Se re-consulta por cada wake del SSE. */
  async getState(): Promise<CongestionStateResponse> {
    const res = await httpClient.get<CongestionStateResponse>(
      '/congestion/state',
    );
    return res.data;
  },

  /** Serie temporal de congestión de un día (recorrido HU-23). */
  async getSeries(day: string): Promise<CongestionSeriesResponse> {
    const res = await httpClient.get<CongestionSeriesResponse>(
      '/congestion/series',
      { params: { day } },
    );
    return res.data;
  },

  /** Predicción GRU por arista (Fase 3). `t` opcional: si se omite, el backend lo deriva
   *  del feed vivo (max(snapshot_timestamp) de waze_jams), coherente con getState(). */
  async getPrediction(t?: number): Promise<CongestionPredictionResponse> {
    const res = await httpClient.get<CongestionPredictionResponse>(
      '/congestion/prediction',
      t !== undefined ? { params: { t } } : undefined,
    );
    return res.data;
  },
};
