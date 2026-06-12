// FASE 4 rediseño UI — hook de datos de GET /api/cameras.
//
// Envuelve camerasService (httpClient, JWT). Lista liviana de cámaras (solo
// CameraDB, sin el agregado Waze de /api/intersections) para el detalle y su
// carril. Pendiente anotado en F2 (camerasService) que esta fase retoma.
import { useCallback } from 'react';
import { camerasService } from '../services/camerasService';
import { useRestResource } from './internal/useRestResource';
import type { RestResource } from './types';
import type { CameraSummary } from '../types/cameras';

export function useCameras(opts?: {
  intervalMs?: number;
}): RestResource<CameraSummary[]> {
  const fetcher = useCallback(
    (signal: AbortSignal) => camerasService.getCameras({ signal }),
    [],
  );
  return useRestResource(fetcher, { intervalMs: opts?.intervalMs });
}
