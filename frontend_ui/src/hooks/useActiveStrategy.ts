// FASE 2 rediseño UI — hook de datos de la estrategia vigente (HU-05).
//
// Estado inicial por controlActiveStateService + suscripción al SSE
// 'active-state-changed' vía openControlActiveStateStream (cliente HU-05, se
// envuelve SIN modificarlo). El payload del evento se IGNORA: es solo señal
// de cambio y se re-lee REST como fuente autoritativa (DHU-021 #15, mismo
// patrón que ActiveStrategyView v1).
//
// `connection` se deriva de los callbacks del client — ver el caveat 401 en
// el JSDoc de StreamConnectionState (hooks/types.ts). Sin `isStale`: el
// umbral de antigüedad es política de vista (v1 usa 10 s atado a la cadencia
// del motor); la vista v2 lo deriva de `lastUpdated`.
import { useCallback, useEffect, useState } from 'react';
import { controlActiveStateService } from '../services/controlActiveStateService';
import type { ActiveStateResponse } from '../services/controlActiveStateService';
import { openControlActiveStateStream } from '../services/sseClient';
import { useRestResource } from './internal/useRestResource';
import type { RestResource, StreamConnectionState } from './types';

export interface UseActiveStrategyResult extends RestResource<ActiveStateResponse> {
  connection: StreamConnectionState;
}

export function useActiveStrategy(nodeId: string): UseActiveStrategyResult {
  const fetcher = useCallback(
    (signal: AbortSignal) => controlActiveStateService.getActiveState(nodeId, { signal }),
    [nodeId],
  );
  const resource = useRestResource(fetcher, { enabled: nodeId !== '' });
  const { refetch } = resource;

  const [connection, setConnection] = useState<StreamConnectionState>('connecting');

  // Render-guard: cambio de nodo = stream nuevo → la conexión vuelve a
  // 'connecting' en el mismo render (sin flash del estado del nodo anterior).
  const [prevNodeId, setPrevNodeId] = useState(nodeId);
  if (prevNodeId !== nodeId) {
    setPrevNodeId(nodeId);
    setConnection('connecting');
  }

  useEffect(() => {
    if (nodeId === '') return;
    const controller = openControlActiveStateStream(nodeId, {
      onMessage: () => {
        // Señal de cambio, payload ignorado: REST autoritativo (DHU-021 #15).
        void refetch();
      },
      onOpen: () => setConnection('open'),
      onError: () => setConnection('retrying'),
      onClose: () => setConnection('closed'),
    });
    return () => {
      controller.abort();
    };
  }, [nodeId, refetch]);

  return { ...resource, connection };
}
