import { httpClient } from './httpClient';

// Interfaces matching the backend
export interface PredictionInput {
    camera_id: string;
    total_vehicles: number;
    occupancy_rate: number;
    flow_rate_per_min: number;
    avg_speed: number;
    avg_density: number;
    hour?: number;
    day_of_week?: number;
}

export interface PredictionResult {
    data: {
        current_congestion_level: string;
        predicted_congestion_15min: string;
        predicted_congestion_30min: string;
        predicted_congestion_45min: string;
        confidence_score: number;
    };
    alert: boolean;
    message: string;
}

export const predictionService = {
    /**
     * Sends current traffic metrics to the prediction engine to get future congestion estimates.
     */
    async predictTraffic(input: PredictionInput): Promise<PredictionResult> {
        try {
            // Add time features if not present
            const now = new Date();
            const payload = {
                ...input,
                hour: input.hour ?? now.getHours(),
                day_of_week: input.day_of_week ?? now.getDay(), // 0=Sunday, 1=Monday...
            };

            const response = await httpClient.post<PredictionResult>('/predictions/predict', payload);
            return response.data;
        } catch (error) {
            console.error('Error calling prediction API:', error);
            throw error;
        }
    }
};
