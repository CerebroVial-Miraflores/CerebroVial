// FASE 2 rediseño UI — wire de GET /predictions/history/{cameraId}?interval=.
//
// SOLO el contrato del backend. Los campos derivados que usa el chart v1
// (type, congestion_score, vehicles_real/pred, congestion_real/pred) se
// computan client-side en la vista y NO pertenecen al wire.

/** Granularidad de agregación en minutos (selector del widget v1). */
export type PredictionHistoryInterval = 1 | 2 | 5 | 10 | 15;

export interface PredictionHistoryPoint {
  timestamp: string;
  total_vehicles: number;
  congestion_level: string;
  is_prediction: boolean;
}

export interface PredictionHistoryForecast {
  predicted_congestion_15min: string;
  predicted_congestion_30min: string;
  predicted_congestion_45min: string;
  predicted_vehicles_15min?: number;
  predicted_vehicles_30min?: number;
  predicted_vehicles_45min?: number;
}

export interface PredictionHistoryResponse {
  camera_id: string;
  history: PredictionHistoryPoint[];
  prediction?: PredictionHistoryForecast;
}
