/**
 * Tests de predictionService.getHistory (FASE 2 rediseño UI).
 *
 * Patrón A: mockea httpClient y verifica URL con path param, params de query,
 * shape wire devuelto tal cual y passthrough del signal. `predictTraffic`
 * (mutación v1) no se testea acá: no cambia en esta fase.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { predictionService } from '../predictionService';
import { httpClient } from '../httpClient';
import type { PredictionHistoryResponse } from '../../types/predictionHistory';

vi.mock('../httpClient', () => ({
  httpClient: { get: vi.fn(), post: vi.fn() },
}));

const getMock = httpClient.get as unknown as ReturnType<typeof vi.fn>;

const payload: PredictionHistoryResponse = {
  camera_id: 'cam_01',
  history: [
    {
      timestamp: '2026-06-10T14:00:00',
      total_vehicles: 42,
      congestion_level: 'Moderado',
      is_prediction: false,
    },
  ],
  prediction: {
    predicted_congestion_15min: 'Alto',
    predicted_congestion_30min: 'Alto',
    predicted_congestion_45min: 'Moderado',
    predicted_vehicles_15min: 55,
  },
};

describe('predictionService.getHistory', () => {
  beforeEach(() => {
    getMock.mockReset();
  });

  it('hace GET a /predictions/history/{cameraId} con interval y devuelve el wire tal cual', async () => {
    getMock.mockResolvedValue({ data: payload });

    const result = await predictionService.getHistory('cam_01', 5);

    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith('/predictions/history/cam_01', {
      params: { interval: 5 },
    });
    expect(result).toEqual(payload);
    expect(result.prediction?.predicted_congestion_15min).toBe('Alto');
  });

  it('pasa el signal junto a los params cuando se provee', async () => {
    getMock.mockResolvedValue({ data: payload });
    const controller = new AbortController();

    await predictionService.getHistory('cam_01', 15, { signal: controller.signal });

    expect(getMock).toHaveBeenCalledWith('/predictions/history/cam_01', {
      params: { interval: 15 },
      signal: controller.signal,
    });
  });
});
