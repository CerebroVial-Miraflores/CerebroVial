/**
 * Tests de useCongestionPrediction (FASE 2 rediseño UI).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useCongestionPrediction } from '../useCongestionPrediction';
import { congestionService } from '../../services/congestionService';
import type { CongestionPredictionResponse } from '../../types/congestion';

vi.mock('../../services/congestionService', () => ({
  congestionService: { getPrediction: vi.fn() },
}));

const getPredictionMock = congestionService.getPrediction as unknown as ReturnType<typeof vi.fn>;

const payload: CongestionPredictionResponse = {
  base_timestep: 720,
  horizon: 30,
  source: 'seed051 (day_idx=9)',
  source_date: '2026-06-10',
  count: 1,
  edges: [{ edge_id: '-129822384#0', levels: [1, 2, 3] }],
};

beforeEach(() => {
  getPredictionMock.mockReset();
});

describe('useCongestionPrediction', () => {
  it('sin t: llama getPrediction(undefined, {signal}) — el backend deriva del feed vivo', async () => {
    getPredictionMock.mockResolvedValue(payload);

    const { result } = renderHook(() => useCongestionPrediction());
    await act(async () => {});

    expect(result.current.data).toEqual(payload);
    const [t, opts] = getPredictionMock.mock.calls[0] as [
      number | undefined,
      { signal: AbortSignal },
    ];
    expect(t).toBeUndefined();
    expect(opts.signal).toBeInstanceOf(AbortSignal);
  });

  it('con t: lo pasa al service; cambio de t = nueva consulta', async () => {
    getPredictionMock.mockResolvedValue(payload);

    const { rerender } = renderHook(({ t }: { t?: number }) => useCongestionPrediction(t), {
      initialProps: { t: 600 as number | undefined },
    });
    await act(async () => {});
    expect(getPredictionMock.mock.calls[0][0]).toBe(600);

    rerender({ t: 720 });
    await act(async () => {});
    expect(getPredictionMock).toHaveBeenCalledTimes(2);
    expect(getPredictionMock.mock.calls[1][0]).toBe(720);
  });

  it('error: superficie {data:null, error} y refetch recupera', async () => {
    getPredictionMock
      .mockRejectedValueOnce(new Error('modelo no disponible'))
      .mockResolvedValueOnce(payload);

    const { result } = renderHook(() => useCongestionPrediction());
    await act(async () => {});
    expect(result.current.error).toBe('modelo no disponible');

    await act(async () => {
      await result.current.refetch();
    });
    expect(result.current.error).toBeNull();
    expect(result.current.data).toEqual(payload);
  });
});
