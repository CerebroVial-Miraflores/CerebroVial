// FASE 2 rediseño UI — hook de datos de GET /congestion/state + SSE.
//
// GET inicial vía congestionService + suscripción a openCongestionStream()
// (cliente HU-22, se envuelve SIN modificarlo): el wake 'congestion-updated'
// no trae payload → re-fetch autoritativo del estado (patrón v1/DHU-021 #15).
// `connection` se deriva de los callbacks del client — ver el caveat 401 en
// el JSDoc de StreamConnectionState (hooks/types.ts).
//
// `isStale`: true cuando pasaron staleAfterMs sin un fetch exitoso (el timer
// se re-arma con cada avance de lastUpdated; un refetch fallido NO lo avanza,
// así que isStale mide la antigüedad real del dato).
import { useCallback, useEffect, useState } from 'react';
import { congestionService } from '../services/congestionService';
import { openCongestionStream } from '../services/congestionSseClient';
import { useRestResource } from './internal/useRestResource';
import type { RestResource, StreamConnectionState } from './types';
import type { CongestionStateResponse } from '../types/congestion';

export interface UseCongestionStateResult extends RestResource<CongestionStateResponse> {
  isStale: boolean;
  connection: StreamConnectionState;
}

export function useCongestionState(opts?: {
  staleAfterMs?: number;
}): UseCongestionStateResult {
  const staleAfterMs = opts?.staleAfterMs ?? 90_000;

  const fetcher = useCallback(
    (signal: AbortSignal) => congestionService.getState({ signal }),
    [],
  );
  const resource = useRestResource(fetcher);
  const { refetch, lastUpdated } = resource;

  const [connection, setConnection] = useState<StreamConnectionState>('connecting');

  // Stream SSE: vive mientras el hook esté montado. El cleanup aborta el
  // controller → cierra la conexión y desuscribe del broadcaster del backend.
  useEffect(() => {
    const controller = openCongestionStream({
      onWake: () => {
        void refetch();
      },
      onOpen: () => setConnection('open'),
      onError: () => setConnection('retrying'),
      onClose: () => setConnection('closed'),
    });
    return () => {
      controller.abort();
    };
  }, [refetch]);

  const [isStale, setIsStale] = useState(false);

  // Render-guard (patrón documentado "adjusting state when props change"):
  // un avance de lastUpdated = dato fresco → vuelve a FRESCO sin esperar al
  // próximo effect.
  const [prevUpdated, setPrevUpdated] = useState(lastUpdated);
  if (prevUpdated !== lastUpdated) {
    setPrevUpdated(lastUpdated);
    setIsStale(false);
  }

  // Timer de staleness: se (re)arma con cada fetch exitoso; cleanup lo limpia
  // (cero timers filtrados en unmount).
  useEffect(() => {
    if (lastUpdated === null) return;
    const id = window.setTimeout(() => {
      setIsStale(true);
    }, staleAfterMs);
    return () => {
      window.clearTimeout(id);
    };
  }, [lastUpdated, staleAfterMs]);

  return { ...resource, isStale, connection };
}
