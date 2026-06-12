// FASE 2 rediseño UI — hook de datos de GET /congestion/series?day= (HU-23).
//
// REST simple para el modo histórico de Fase 3. Cambio de `day` = nuevo
// recurso (reset + abort del vuelo anterior). `day` vacío = disabled (el
// consumidor puede esperar la selección de día sin gimnasia condicional).
import { useCallback } from 'react';
import { congestionService } from '../services/congestionService';
import { useRestResource } from './internal/useRestResource';
import type { RestResource } from './types';
import type { CongestionSeriesResponse } from '../types/congestion';

export function useCongestionSeries(day: string): RestResource<CongestionSeriesResponse> {
  const fetcher = useCallback(
    (signal: AbortSignal) => congestionService.getSeries(day, { signal }),
    [day],
  );
  return useRestResource(fetcher, { enabled: day !== '' });
}
