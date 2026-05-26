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
}

export interface RecommendResponse {
    data: ControlRecommendation;
}

export interface ErrorDetail {
    code: string;
    message: string;
}

export const controlService = {
    async recommend(state: IntersectionState): Promise<RecommendResponse> {
        const res = await httpClient.post<RecommendResponse>(
            '/control/recommend',
            state,
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
