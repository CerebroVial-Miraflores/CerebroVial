// FASE 2 rediseño UI — hook del SSE de visión del edge (TTH-08 §6.2).
//
// EventSource NATIVO a `${VITE_EDGE_API_URL}/stream/{cameraId}` escuchando
// 'traffic_update' con parse tipado a VisionStreamPayload. Lo consume el panel
// en vivo del detalle de cámara (CameraLivePanel, F4).
//
// DEUDA BACKEND (documentada en CLAUDE.md §Rediseño UI): el edge NO exige JWT
// y EventSource nativo no acepta headers custom — este hook NO resuelve esa
// auth, la envuelve tal cual. Cuando el edge la exija, el reemplazo natural
// es un client tipo @microsoft/fetch-event-source como los del core.
//
// `enabled` (default true): en el flujo real el stream se abre recién después
// del alta on-demand al edge (POST /cameras/{id}, gate `isActive` de v1) —
// sin alta el edge no emite y EventSource reintentaría en loop.
import { useEffect, useState } from 'react';
import type { VisionStreamPayload } from '../types/visionStream';
import type { StreamConnectionState } from './types';

const EDGE_API_URL: string =
  import.meta.env?.VITE_EDGE_API_URL || 'http://localhost:8000';

export interface UseVisionStreamResult {
  /** Último payload 'traffic_update' parseado. Uno malformado se ignora. */
  data: VisionStreamPayload | null;
  /** Epoch ms del último payload válido. */
  lastUpdated: number | null;
  /**
   * Derivado de onopen/onerror del EventSource nativo (que auto-reconecta
   * salvo cierre fatal): onerror con readyState CLOSED → 'closed', si no
   * 'retrying'. Sin el caveat 401 de los SSE del core: el edge no autentica.
   */
  connection: StreamConnectionState;
}

export function useVisionStream(
  cameraId: string,
  opts?: { enabled?: boolean },
): UseVisionStreamResult {
  const enabled = opts?.enabled ?? true;
  const active = enabled && cameraId !== '';

  const [data, setData] = useState<VisionStreamPayload | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [connection, setConnection] = useState<StreamConnectionState>(() =>
    active ? 'connecting' : 'closed',
  );

  // Render-guard: cambio de cámara o de enabled = stream nuevo → métricas de
  // la cámara anterior fuera y conexión re-derivada en el mismo render.
  const streamKey = `${active}:${cameraId}`;
  const [prevKey, setPrevKey] = useState(streamKey);
  if (prevKey !== streamKey) {
    setPrevKey(streamKey);
    setData(null);
    setLastUpdated(null);
    setConnection(active ? 'connecting' : 'closed');
  }

  useEffect(() => {
    if (!active) return;
    const source = new EventSource(`${EDGE_API_URL}/stream/${cameraId}`);

    const onTrafficUpdate = (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data as string) as VisionStreamPayload;
        setData(payload);
        setLastUpdated(Date.now());
      } catch {
        // Payload malformado: se ignora en silencio, la data previa queda.
      }
    };
    source.addEventListener('traffic_update', onTrafficUpdate);
    source.onopen = () => setConnection('open');
    source.onerror = () => {
      setConnection(
        source.readyState === EventSource.CLOSED ? 'closed' : 'retrying',
      );
    };

    return () => {
      source.close();
    };
  }, [active, cameraId]);

  return { data, lastUpdated, connection };
}
