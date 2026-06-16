// Cliente de las acciones de inferencia del edge (alta/baja on-demand del YOLO y
// estado del contenedor). El edge va CRUDO (fetch directo, sin httpClient/JWT —
// su auth es deuda backend). Contratos en
// edge_device/src/vision/presentation/api/routes/cameras.py:
//   GET    /cameras/inference-status → {inferring, count, cap, capacity_used}
//   POST   /cameras/{id}  body {source, source_type, zones} → 200 | 409 (tope)
//   DELETE /cameras/{id}  → {status:"removed"}
import { EDGE_API_URL } from '../config/edge';

export interface InferenceStatus {
  inferring: string[];
  count: number;
  cap: number | null;
  capacity_used: number | null;
}

export interface StartInferenceConfig {
  source: string;
  source_type: string;
  zones: Record<string, unknown>;
}

/** El contenedor de inferencia alcanzó su tope (HTTP 409). El handler del toggle
 *  la distingue de un fallo de red para revertir a BAJA sin romper. */
export class EdgeCapacityError extends Error {
  constructor(message = 'El edge está a capacidad de inferencia.') {
    super(message);
    this.name = 'EdgeCapacityError';
  }
}

export const edgeInferenceService = {
  async getInferenceStatus(opts?: { signal?: AbortSignal }): Promise<InferenceStatus> {
    const res = await fetch(`${EDGE_API_URL}/cameras/inference-status`, {
      ...(opts?.signal ? { signal: opts.signal } : {}),
    });
    if (!res.ok) throw new Error(`Edge respondió ${res.status} al consultar inference-status`);
    return (await res.json()) as InferenceStatus;
  },

  async startInference(cameraId: string, config: StartInferenceConfig): Promise<void> {
    const res = await fetch(`${EDGE_API_URL}/cameras/${cameraId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (res.status === 409) throw new EdgeCapacityError();
    if (!res.ok) throw new Error(`Edge respondió ${res.status} al iniciar la detección`);
  },

  async stopInference(cameraId: string): Promise<void> {
    const res = await fetch(`${EDGE_API_URL}/cameras/${cameraId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`Edge respondió ${res.status} al detener la detección`);
  },
};
