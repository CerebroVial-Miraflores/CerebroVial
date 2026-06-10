// FASE 2 rediseño UI — hook de datos de GET /congestion/geometry.
//
// La geometría (1660 tramos) es estática: se consulta UNA vez por sesión con
// un cache de PROMISE a nivel módulo. Cachear la promise (no el valor) dedupea
// montajes concurrentes — el doble montaje de StrictMode dispara dos effects
// antes de que resuelva la primera y aun así hay UN solo GET.
//
// Ante fallo, el catch LIMPIA el cache y re-lanza: un error no envenena la
// sesión (refetch reintenta de verdad) y el hook ve el error por la superficie
// {data, error, loading} normal.
import { useCallback } from 'react';
import { congestionService } from '../services/congestionService';
import { useRestResource } from './internal/useRestResource';
import type { RestResource } from './types';
import type { GeometryFeatureCollection } from '../types/congestion';

let geometryPromise: Promise<GeometryFeatureCollection> | null = null;

function fetchGeometryOnce(): Promise<GeometryFeatureCollection> {
  if (!geometryPromise) {
    // Sin signal a propósito: la promise es compartida entre consumidores —
    // abortar a uno no debe matar el GET de los demás. La protección
    // anti-setState post-unmount la da el guard signal.aborted del helper.
    geometryPromise = congestionService.getGeometry().catch((err) => {
      geometryPromise = null;
      throw err;
    });
  }
  return geometryPromise;
}

/** Solo para tests: limpia el cache de sesión entre casos. */
export function resetGeometryCacheForTests(): void {
  geometryPromise = null;
}

export function useCongestionGeometry(): RestResource<GeometryFeatureCollection> {
  const fetcher = useCallback(() => fetchGeometryOnce(), []);
  return useRestResource(fetcher);
}
