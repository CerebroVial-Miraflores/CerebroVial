// HU-22 — Cliente REST de los endpoints de congestión de red (Fase 1).
//
// Reusa el httpClient compartido (interceptor JWT de HU-01: inyecta
// Authorization: Bearer y maneja 401 → authBridge.onUnauthorized). Los endpoints
// /congestion/* están protegidos con require_role(OPERATOR, ADMIN), por lo que
// NO se usa fetch crudo — el token va por el interceptor.
//
// FASE 2 rediseño UI: firmas extendidas con `opts?: { signal? }` trailing para
// la cancelación desde hooks (AbortController en unmount). Backward-compatible:
// sin signal, la llamada a httpClient conserva su forma original exacta.
import { httpClient } from './httpClient';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
  CongestionSeriesResponse,
  CongestionPredictionResponse,
} from '../types/congestion';

export const congestionService = {
  /** Geometría de la red (estática, 1660 aristas). Se consume 1× al montar el mapa. */
  async getGeometry(opts?: { signal?: AbortSignal }): Promise<GeometryFeatureCollection> {
    const res = opts?.signal
      ? await httpClient.get<GeometryFeatureCollection>('/congestion/geometry', {
          signal: opts.signal,
        })
      : await httpClient.get<GeometryFeatureCollection>('/congestion/geometry');
    return res.data;
  },

  /** Último estado de congestión por arista. Se re-consulta por cada wake del SSE. */
  async getState(opts?: { signal?: AbortSignal }): Promise<CongestionStateResponse> {
    const res = opts?.signal
      ? await httpClient.get<CongestionStateResponse>('/congestion/state', {
          signal: opts.signal,
        })
      : await httpClient.get<CongestionStateResponse>('/congestion/state');
    return res.data;
  },

  /** Serie temporal de congestión de un día (recorrido HU-23). */
  async getSeries(
    day: string,
    opts?: { signal?: AbortSignal },
  ): Promise<CongestionSeriesResponse> {
    const res = await httpClient.get<CongestionSeriesResponse>(
      '/congestion/series',
      { params: { day }, ...(opts?.signal ? { signal: opts.signal } : {}) },
    );
    return res.data;
  },

  /** Predicción GRU por arista (Fase 3). `t` opcional: si se omite, el backend lo deriva
   *  del feed vivo (max(snapshot_timestamp) de waze_jams), coherente con getState(). */
  async getPrediction(
    t?: number,
    opts?: { signal?: AbortSignal },
  ): Promise<CongestionPredictionResponse> {
    const config = {
      ...(t !== undefined ? { params: { t } } : {}),
      ...(opts?.signal ? { signal: opts.signal } : {}),
    };
    const res = await httpClient.get<CongestionPredictionResponse>(
      '/congestion/prediction',
      Object.keys(config).length > 0 ? config : undefined,
    );
    return res.data;
  },
};
