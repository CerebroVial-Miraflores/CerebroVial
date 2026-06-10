/**
 * Tests de usePredictionHistory (FASE 2 rediseño UI).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { usePredictionHistory } from '../usePredictionHistory';
import { predictionService } from '../../services/predictionService';
import type {
  PredictionHistoryInterval,
  PredictionHistoryResponse,
} from '../../types/predictionHistory';

vi.mock('../../services/predictionService', () => ({
  predictionService: { getHistory: vi.fn() },
}));

const getHistoryMock = predictionService.getHistory as unknown as ReturnType<typeof vi.fn>;

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
};

beforeEach(() => {
  getHistoryMock.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('usePredictionHistory', () => {
  it('éxito: llama getHistory(cameraId, interval, {signal}) y expone el wire', async () => {
    getHistoryMock.mockResolvedValue(payload);

    const { result } = renderHook(() => usePredictionHistory('cam_01', 5));
    await act(async () => {});

    expect(result.current.data).toEqual(payload);
    const [cameraId, interval, opts] = getHistoryMock.mock.calls[0] as [
      string,
      PredictionHistoryInterval,
      { signal: AbortSignal },
    ];
    expect(cameraId).toBe('cam_01');
    expect(interval).toBe(5);
    expect(opts.signal).toBeInstanceOf(AbortSignal);
  });

  it('cambio de interval = nuevo recurso (reset + nueva llamada)', async () => {
    getHistoryMock.mockResolvedValue(payload);

    const { rerender } = renderHook(
      ({ interval }: { interval: PredictionHistoryInterval }) =>
        usePredictionHistory('cam_01', interval),
      { initialProps: { interval: 5 as PredictionHistoryInterval } },
    );
    await act(async () => {});

    rerender({ interval: 15 });
    await act(async () => {});

    expect(getHistoryMock).toHaveBeenCalledTimes(2);
    expect(getHistoryMock.mock.calls[1][1]).toBe(15);
  });

  it("cameraId '' = disabled (id asíncrono aún no disponible)", async () => {
    const { result } = renderHook(() => usePredictionHistory('', 5));
    await act(async () => {});

    expect(getHistoryMock).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);
  });

  it('polling opcional 60 s y unmount sin timers filtrados', async () => {
    vi.useFakeTimers();
    getHistoryMock.mockResolvedValue(payload);

    const { unmount } = renderHook(() =>
      usePredictionHistory('cam_01', 5, { intervalMs: 60_000 }),
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(getHistoryMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(getHistoryMock).toHaveBeenCalledTimes(2);

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('error: superficie {data:null, error} sin throw', async () => {
    getHistoryMock.mockRejectedValue(new Error('cámara desconocida'));

    const { result } = renderHook(() => usePredictionHistory('cam_x', 5));
    await act(async () => {});

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('cámara desconocida');
  });
});
