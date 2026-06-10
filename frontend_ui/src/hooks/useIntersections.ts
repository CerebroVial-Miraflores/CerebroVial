// FASE 2 rediseño UI — hook de datos de GET /api/intersections.
//
// Envuelve intersectionsService (httpClient, JWT). AbortController en unmount
// vía el signal de axios; polling opcional con intervalMs.
import { useCallback } from 'react';
import { intersectionsService } from '../services/intersectionsService';
import { useRestResource } from './internal/useRestResource';
import type { RestResource } from './types';
import type { IntersectionSummary } from '../types/intersections';

export function useIntersections(opts?: {
  intervalMs?: number;
}): RestResource<IntersectionSummary[]> {
  const fetcher = useCallback(
    (signal: AbortSignal) => intersectionsService.getIntersections({ signal }),
    [],
  );
  return useRestResource(fetcher, { intervalMs: opts?.intervalMs });
}
