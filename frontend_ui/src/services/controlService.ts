import axios from 'axios';
import type { AxiosError } from 'axios';
import { httpClient } from './httpClient';

export interface PhaseFlow {
    phase_id: string;
    flow: number;
    saturation_flow: number;
    queue?: number;
    has_pedestrian?: boolean;
}

export interface PredictedDemand {
    horizon_minutes: number;
    vehicles_by_phase: Record<string, number>;
}

export interface IntersectionState {
    intersection_id: string;
    timestamp: string;
    phases: PhaseFlow[];
    lost_time?: number;
    predicted_demand?: PredictedDemand | null;
}

export interface PhaseTimings {
    phase_id: string;
    green: number;
    yellow: number;
    all_red: number;
}

export type ControlMode = 'webster' | 'max_pressure';

export interface ControlRecommendation {
    intersection_id: string;
    mode: ControlMode;
    cycle_seconds: number;
    phase_timings: PhaseTimings[];
    next_phase?: string | null;
    reasoning: string;
    adjustments: string[];
    // Expuesto por el backend para que el playground de Admin pueda
    // disparar __internal/activate sobre esta decisión sin consulta
    // adicional. Optional para compat con respuestas previas a la
    // ampliación del schema (HU-05 fase de cierre).
    decision_id?: string | null;
}

export interface RecommendResponse {
    data: ControlRecommendation;
}

export interface ErrorDetail {
    code: string;
    message: string;
}

export interface InternalActivatePayload {
    node_id: string;
    decision_id: string;
    activated_by?: string | null;
}

export const controlService = {
    async recommend(state: IntersectionState): Promise<RecommendResponse> {
        const res = await httpClient.post<RecommendResponse>(
            '/control/recommend',
            state,
        );
        return res.data;
    },

    // Disparador de demo del Admin (DHU-020). El endpoint del backend
    // está gated por ENABLE_TEST_ACTIVATOR=true y requiere rol Admin.
    // Si la env var no está abierta, el backend devuelve 404 (la ruta
    // no se registra) — el caller lo trata como "no disponible en este
    // entorno" en vez de error feo.
    //
    // TODO(HU-07): este disparador es TEMPORAL para que la demo de DHU-020
    // pueda observar el SSE en vivo. HU-07 lo reemplaza por el activador
    // productivo (operador o automatización) sin gate de env var y con
    // la UI dedicada del operador. Visión: demo (HU-05) → producto manual
    // (HU-07) → automático (motor+predicción, futuro).
    async activateDecision(
        payload: InternalActivatePayload,
    ): Promise<{ ok: boolean }> {
        const res = await httpClient.post<{ ok: boolean }>(
            '/control/__internal/activate',
            payload,
        );
        return res.data;
    },
};

export function parseControlError(err: unknown): ErrorDetail | null {
    if (axios.isAxiosError(err) && err.response?.status === 422) {
        const detail = (err as AxiosError<{ detail: unknown }>).response?.data?.detail;
        if (
            detail &&
            typeof detail === 'object' &&
            !Array.isArray(detail) &&
            'code' in detail &&
            'message' in detail
        ) {
            return detail as ErrorDetail;
        }
    }
    return null;
}
