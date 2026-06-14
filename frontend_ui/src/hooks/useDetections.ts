// FASE 4 Mitad A — hook del tap de detecciones del edge (D-019).
//
// Polling a 1 Hz contra GET /detections/{id}/latest (la cadencia del scheduler
// keyframe-only). Espeja el patrón de useVisionStream: gate `enabled` (se abre
// recién tras el alta on-demand, `isActive`), reset por cámara, y degradación
// silenciosa ante payload/red mala.
//
// Canal SEPARADO del SSE de métricas (criterio HU-22): las cajas tienen cadencia,
// payload y consumidor (overlay) distintos a los agregados por-ventana.
//
// FRESCURA: la edad de la caja se calcula en RELOJ DEL EDGE como
// `server_timestamp - frame_timestamp` (sin sesgo por skew con el reloj del
// browser). Si supera FRESHNESS_SECONDS, las cajas se consideran viejas y el hook
// devuelve `boxes: []` (mejor sin cajas que cajas viejas sobre autos que ya se
// movieron). Este umbral es INDEPENDIENTE del umbral de frescura del scheduler
// (Fase 1, 4.5 s): uno gobierna si el overlay dibuja; el otro si el edge infiere.
//
// DEUDA BACKEND (igual que useVisionStream): el edge no exige JWT.
import { useEffect, useState } from 'react';

import type { DetectionBox, DetectionFrame, DetectionsPayload } from '../types/detections';

const EDGE_API_URL: string =
  import.meta.env?.VITE_EDGE_API_URL || 'http://localhost:8000';

const DETECTIONS_POLL_MS = 1000; // 1 Hz: la cadencia del scheduler keyframe-only.
const FRESHNESS_SECONDS = 3; // edad máx de la caja antes de ocultar el overlay.

export interface UseDetectionsResult {
  /** Cajas a dibujar. Vacío si stale, sin datos, o error de red. */
  boxes: DetectionBox[];
  /** Dims del frame para el viewBox del overlay. null si sin datos. */
  frame: DetectionFrame | null;
  /** Edad de la caja en segundos (server - frame), en reloj del edge. */
  ageSeconds: number | null;
  /** True si hay cajas frescas (edad <= FRESHNESS_SECONDS) sin error. */
  fresh: boolean;
}

export function useDetections(
  cameraId: string,
  opts?: { enabled?: boolean },
): UseDetectionsResult {
  const enabled = opts?.enabled ?? true;
  const active = enabled && cameraId !== '';

  const [data, setData] = useState<DetectionsPayload | null>(null);
  const [hasError, setHasError] = useState(false);

  // Render-guard: cambio de cámara/enabled = stream nuevo → datos de la cámara
  // anterior fuera en el mismo render (espeja useVisionStream).
  const streamKey = `${active}:${cameraId}`;
  const [prevKey, setPrevKey] = useState(streamKey);
  if (prevKey !== streamKey) {
    setPrevKey(streamKey);
    setData(null);
    setHasError(false);
  }

  useEffect(() => {
    if (!active) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(`${EDGE_API_URL}/detections/${cameraId}/latest`);
        if (!res.ok) throw new Error(`Edge respondió ${res.status}`);
        const payload = (await res.json()) as DetectionsPayload;
        if (!cancelled) {
          setData(payload);
          setHasError(false);
        }
      } catch {
        // Red/payload malo: marcar error → el overlay se oculta (no congela
        // cajas viejas si el polling deja de responder).
        if (!cancelled) setHasError(true);
      }
    };

    void poll();
    const interval = window.setInterval(() => void poll(), DETECTIONS_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [active, cameraId]);

  const ageSeconds =
    data && data.frame_timestamp != null
      ? data.server_timestamp - data.frame_timestamp
      : null;
  const fresh =
    !hasError && ageSeconds != null && ageSeconds <= FRESHNESS_SECONDS;
  const boxes = fresh && data ? data.detections : [];
  const frame = data?.frame ?? null;

  return { boxes, frame, ageSeconds, fresh };
}
