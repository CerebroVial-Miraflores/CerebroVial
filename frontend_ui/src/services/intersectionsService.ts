// FASE 2 rediseño UI — Cliente REST de GET /api/intersections.
//
// Normaliza el fetch crudo de DashboardView (v1, que queda intacto hasta morir)
// a la política de la fase: todo acceso al core vía httpClient (interceptor JWT
// de HU-01 + 401 → authBridge.onUnauthorized).
import { httpClient } from './httpClient';
import type { IntersectionSummary } from '../types/intersections';

export const intersectionsService = {
  /** Lista de intersecciones con congestión agregada en `status`. Acepta signal
   *  para cancelación desde hooks (AbortController en unmount). */
  async getIntersections(opts?: { signal?: AbortSignal }): Promise<IntersectionSummary[]> {
    const res = opts?.signal
      ? await httpClient.get<IntersectionSummary[]>('/api/intersections', { signal: opts.signal })
      : await httpClient.get<IntersectionSummary[]>('/api/intersections');
    return res.data;
  },
};
