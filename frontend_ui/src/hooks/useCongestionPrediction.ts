// FASE 2 rediseño UI — hook de datos de GET /congestion/prediction?t=.
//
// REST simple para el modo predicción de Fase 3. `t` omitido: el backend
// deriva el timestep del feed vivo (max(snapshot_timestamp) de waze_jams),
// coherente con getState(). Cambio de `t` = nuevo recurso.
import { useCallback } from 'react';
import { congestionService } from '../services/congestionService';
import { useRestResource } from './internal/useRestResource';
import type { RestResource } from './types';
import type { CongestionPredictionResponse } from '../types/congestion';

export function useCongestionPrediction(t?: number): RestResource<CongestionPredictionResponse> {
  const fetcher = useCallback(
    (signal: AbortSignal) => congestionService.getPrediction(t, { signal }),
    [t],
  );
  return useRestResource(fetcher);
}
