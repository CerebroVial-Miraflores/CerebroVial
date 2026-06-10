// FASE 2 rediseño UI — Cliente REST de GET /api/cameras.
//
// Lista liviana (solo CameraDB, sin agregado Waze). Normaliza el fetch crudo
// del carril de CameraDetailView (v1, intacto hasta morir) vía httpClient (JWT).
// El hook useCameras se difiere a la fase de cámaras; el service queda listo.
import { httpClient } from './httpClient';
import type { CameraSummary } from '../types/cameras';

export const camerasService = {
  async getCameras(opts?: { signal?: AbortSignal }): Promise<CameraSummary[]> {
    const res = opts?.signal
      ? await httpClient.get<CameraSummary[]>('/api/cameras', { signal: opts.signal })
      : await httpClient.get<CameraSummary[]>('/api/cameras');
    return res.data;
  },
};
