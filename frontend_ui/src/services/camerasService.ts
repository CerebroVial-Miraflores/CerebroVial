// FASE 2 rediseño UI — Cliente REST de GET /api/cameras.
//
// Lista liviana (solo CameraDB, sin agregado Waze) vía httpClient (JWT). La
// consume useCameras (detalle de cámara y su carril, F4).
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
