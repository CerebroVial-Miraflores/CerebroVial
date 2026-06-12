// FASE 2 rediseño UI — hook de datos de GET /predictions/history/{cameraId}.
//
// Sobre el service normalizado (httpClient, JWT). Cambio de cameraId o
// interval = nuevo recurso. `cameraId` vacío = disabled (id asíncrono).
// `intervalMs` opcional: el widget v1 hace polling de 60 s; Fase 3 decide.
import { useCallback } from 'react';
import { predictionService } from '../services/predictionService';
import { useRestResource } from './internal/useRestResource';
import type { RestResource } from './types';
import type {
  PredictionHistoryInterval,
  PredictionHistoryResponse,
} from '../types/predictionHistory';

export function usePredictionHistory(
  cameraId: string,
  interval: PredictionHistoryInterval,
  opts?: { intervalMs?: number },
): RestResource<PredictionHistoryResponse> {
  const fetcher = useCallback(
    (signal: AbortSignal) => predictionService.getHistory(cameraId, interval, { signal }),
    [cameraId, interval],
  );
  return useRestResource(fetcher, {
    enabled: cameraId !== '',
    intervalMs: opts?.intervalMs,
  });
}
