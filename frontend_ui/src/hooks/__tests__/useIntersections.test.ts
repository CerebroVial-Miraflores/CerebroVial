/**
 * Tests de useIntersections (FASE 2 rediseño UI).
 *
 * Service mockeado a nivel módulo. La máquina de estados completa la cubre
 * useRestResource.test.ts; acá: wiring del service (signal incluido),
 * éxito/error/refetch, polling y cleanup sin timers filtrados.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useIntersections } from '../useIntersections';
import { intersectionsService } from '../../services/intersectionsService';
import type { IntersectionSummary } from '../../types/intersections';

vi.mock('../../services/intersectionsService', () => ({
  intersectionsService: { getIntersections: vi.fn() },
}));

const getMock = intersectionsService.getIntersections as unknown as ReturnType<typeof vi.fn>;

const payload: IntersectionSummary[] = [
  {
    id: 'larco_schell',
    name: 'Av. Larco / Schell',
    speed: 22,
    flow: 540,
    status: 'Moderado',
    lat: -12.121,
    lng: -77.029,
    stream_url: null,
  },
];

beforeEach(() => {
  getMock.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useIntersections', () => {
  it('éxito: expone la lista y pasa un AbortSignal al service', async () => {
    getMock.mockResolvedValue(payload);

    const { result } = renderHook(() => useIntersections());
    expect(result.current.loading).toBe(true);
    await act(async () => {});

    expect(result.current.data).toEqual(payload);
    expect(result.current.error).toBeNull();
    const opts = getMock.mock.calls[0][0] as { signal: AbortSignal };
    expect(opts.signal).toBeInstanceOf(AbortSignal);
  });

  it('error: superficie {data:null, error} sin throw', async () => {
    getMock.mockRejectedValue(new Error('core caído'));

    const { result } = renderHook(() => useIntersections());
    await act(async () => {});

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('core caído');
  });

  it('refetch re-consulta el service', async () => {
    getMock.mockResolvedValue(payload);

    const { result } = renderHook(() => useIntersections());
    await act(async () => {});

    await act(async () => {
      await result.current.refetch();
    });
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it('polling opcional: re-fetch cada intervalMs y unmount sin timers filtrados', async () => {
    vi.useFakeTimers();
    getMock.mockResolvedValue(payload);

    const { unmount } = renderHook(() => useIntersections({ intervalMs: 30_000 }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(getMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(getMock).toHaveBeenCalledTimes(3);

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('unmount aborta el vuelo: el signal pasado al service queda aborted', async () => {
    let captured: AbortSignal | null = null;
    getMock.mockImplementation((opts: { signal: AbortSignal }) => {
      captured = opts.signal;
      return new Promise(() => {}); // nunca resuelve
    });

    const { unmount } = renderHook(() => useIntersections());
    await act(async () => {});

    unmount();
    expect(captured!.aborted).toBe(true);
  });
});
