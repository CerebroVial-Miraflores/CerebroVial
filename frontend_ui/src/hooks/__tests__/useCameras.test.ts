/**
 * Tests de useCameras (FASE 4 rediseño UI).
 *
 * Service mockeado a nivel módulo. La máquina de estados completa la cubre
 * useRestResource.test.ts; acá: wiring del service (signal incluido),
 * éxito/error/refetch y cleanup sin timers filtrados.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useCameras } from '../useCameras';
import { camerasService } from '../../services/camerasService';
import type { CameraSummary } from '../../types/cameras';

vi.mock('../../services/camerasService', () => ({
  camerasService: { getCameras: vi.fn() },
}));

const getMock = camerasService.getCameras as unknown as ReturnType<typeof vi.fn>;

const payload: CameraSummary[] = [
  { id: 'cam_larco_schell', name: 'Av. Larco / Schell', stream_url: 'https://x/a.m3u8' },
  { id: 'cam_larco_benavides', name: 'Av. Larco / Benavides', stream_url: null },
];

beforeEach(() => {
  getMock.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useCameras', () => {
  it('éxito: expone la lista y pasa un AbortSignal al service', async () => {
    getMock.mockResolvedValue(payload);

    const { result } = renderHook(() => useCameras());
    expect(result.current.loading).toBe(true);
    await act(async () => {});

    expect(result.current.data).toEqual(payload);
    expect(result.current.error).toBeNull();
    const opts = getMock.mock.calls[0][0] as { signal: AbortSignal };
    expect(opts.signal).toBeInstanceOf(AbortSignal);
  });

  it('error: superficie {data:null, error} sin throw', async () => {
    getMock.mockRejectedValue(new Error('core caído'));

    const { result } = renderHook(() => useCameras());
    await act(async () => {});

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('core caído');
  });

  it('refetch re-consulta el service', async () => {
    getMock.mockResolvedValue(payload);

    const { result } = renderHook(() => useCameras());
    await act(async () => {});

    await act(async () => {
      await result.current.refetch();
    });
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it('polling opcional: re-fetch cada intervalMs y unmount sin timers filtrados', async () => {
    vi.useFakeTimers();
    getMock.mockResolvedValue(payload);

    const { unmount } = renderHook(() => useCameras({ intervalMs: 30_000 }));
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
});
