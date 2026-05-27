// HU-05 — Cliente REST del endpoint GET /control/active-state/{node_id}.
//
// Reusa el httpClient compartido (con el interceptor JWT de HU-01 que inyecta
// Authorization: Bearer y maneja 401 → authBridge.onUnauthorized).
import { httpClient } from './httpClient';
import type { ControlMode, PhaseTimings } from './controlService';

export interface ActiveStateResponse {
  node_id: string;
  strategy_mode: ControlMode;
  cycle_seconds: number;
  phase_timings: PhaseTimings[];
  decided_at: string;
  activated_at: string;
  activated_by: string | null;
}

export const controlActiveStateService = {
  async getActiveState(nodeId: string): Promise<ActiveStateResponse> {
    const res = await httpClient.get<ActiveStateResponse>(
      `/control/active-state/${nodeId}`,
    );
    return res.data;
  },
};
