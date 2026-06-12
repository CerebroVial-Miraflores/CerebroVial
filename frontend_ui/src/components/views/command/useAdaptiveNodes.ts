// FASE 3 rediseño UI — KPI "cruces en modo adaptativo" (REAL-PARCIAL).
//
// One-shot Promise.allSettled de GET /control/active-state/{node_id} sobre los
// 5 KNOWN_NODE_IDS (decisión D1 del plan): 5 GETs baratos al montar, sin SSE.
// Clasificación por respuesta-contrato:
// · fulfilled → 'active' (hay estrategia adaptativa vigente: webster/max_pressure)
// · rejected 404 → 'no-strategy' (contrato `no_active_state` de HU-05 — en los
//   nodos del seed un 404 solo puede significar eso)
// · otro rechazo → 'failed' ("sin dato": red/auth/5xx)
// Vive en command/ (no en src/hooks/): depende de KNOWN_NODE_IDS, que es del
// plano de control (su reemplazo por endpoint dedicado es deuda HU-05+).
import { useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';

import {
  controlActiveStateService,
  type ActiveStateResponse,
} from '../../../services/controlActiveStateService';
import { KNOWN_NODE_IDS } from '../control/controlTypes';

export interface AdaptiveNodeStatus {
  nodeId: string;
  state: ActiveStateResponse | null;
  kind: 'active' | 'no-strategy' | 'failed';
}

export interface UseAdaptiveNodesResult {
  /** null mientras el one-shot está en vuelo. */
  nodes: AdaptiveNodeStatus[] | null;
  /** Nº de nodos con estrategia activa; null mientras carga. */
  activeCount: number | null;
  loading: boolean;
  refetch: () => void;
}

function classify(
  nodeId: string,
  result: PromiseSettledResult<ActiveStateResponse>,
): AdaptiveNodeStatus {
  if (result.status === 'fulfilled') {
    return { nodeId, state: result.value, kind: 'active' };
  }
  const err = result.reason;
  if (axios.isAxiosError(err) && err.response?.status === 404) {
    return { nodeId, state: null, kind: 'no-strategy' };
  }
  return { nodeId, state: null, kind: 'failed' };
}

export function useAdaptiveNodes(): UseAdaptiveNodesResult {
  const [nodes, setNodes] = useState<AdaptiveNodeStatus[] | null>(null);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      const settled = await Promise.allSettled(
        KNOWN_NODE_IDS.map((id) =>
          controlActiveStateService.getActiveState(id, { signal: controller.signal }),
        ),
      );
      if (controller.signal.aborted) return;
      setNodes(KNOWN_NODE_IDS.map((id, i) => classify(id, settled[i])));
    })();
    return () => {
      controller.abort();
    };
  }, [version]);

  // Refetch silencioso: conserva la lista stale mientras el nuevo vuelo settlea.
  const refetch = useCallback(() => setVersion((v) => v + 1), []);

  const activeCount = useMemo(
    () => (nodes === null ? null : nodes.filter((n) => n.kind === 'active').length),
    [nodes],
  );

  return { nodes, activeCount, loading: nodes === null, refetch };
}
