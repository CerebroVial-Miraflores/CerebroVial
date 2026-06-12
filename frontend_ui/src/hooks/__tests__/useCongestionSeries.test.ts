/**
 * Tests de useCongestionSeries (FASE 2 rediseño UI).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useCongestionSeries } from '../useCongestionSeries';
import { congestionService } from '../../services/congestionService';
import type { CongestionSeriesResponse } from '../../types/congestion';

vi.mock('../../services/congestionService', () => ({
  congestionService: { getSeries: vi.fn() },
}));

const getSeriesMock = congestionService.getSeries as unknown as ReturnType<typeof vi.fn>;

const payload: CongestionSeriesResponse = {
  day: '2026-06-10',
  t0: '2026-06-10T00:00:00',
  step_s: 300,
  coverage_end: '2026-06-10T12:00:00',
  count: 1,
  edges: [{ edge_id: '-129822384#0', levels: [0, 1, 2] }],
};

beforeEach(() => {
  getSeriesMock.mockReset();
});

describe('useCongestionSeries', () => {
  it('éxito: llama al service con (day, {signal}) y expone la serie', async () => {
    getSeriesMock.mockResolvedValue(payload);

    const { result } = renderHook(() => useCongestionSeries('2026-06-10'));
    await act(async () => {});

    expect(result.current.data).toEqual(payload);
    const [day, opts] = getSeriesMock.mock.calls[0] as [string, { signal: AbortSignal }];
    expect(day).toBe('2026-06-10');
    expect(opts.signal).toBeInstanceOf(AbortSignal);
  });

  it('error: superficie {data:null, error}', async () => {
    getSeriesMock.mockRejectedValue(new Error('sin datos'));

    const { result } = renderHook(() => useCongestionSeries('2026-06-10'));
    await act(async () => {});

    expect(result.current.error).toBe('sin datos');
  });

  it('cambio de day = nuevo recurso: reset y nueva llamada', async () => {
    getSeriesMock.mockResolvedValue(payload);

    const { result, rerender } = renderHook(({ day }) => useCongestionSeries(day), {
      initialProps: { day: '2026-06-09' },
    });
    await act(async () => {});
    expect(result.current.data).toEqual(payload);

    rerender({ day: '2026-06-10' });
    await act(async () => {});

    expect(getSeriesMock).toHaveBeenCalledTimes(2);
    expect(getSeriesMock.mock.calls[1][0]).toBe('2026-06-10');
  });

  it("day '' = disabled: no consulta", async () => {
    const { result } = renderHook(() => useCongestionSeries(''));
    await act(async () => {});

    expect(getSeriesMock).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);
  });
});
