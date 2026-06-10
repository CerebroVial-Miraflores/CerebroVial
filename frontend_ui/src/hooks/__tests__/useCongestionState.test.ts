/**
 * Tests de useCongestionState (FASE 2 rediseño UI).
 *
 * Mock del service + del SSE client capturando callbacks en closure (patrón
 * sseClient.test.ts). Cubre: GET inicial, mapping de connection, wake →
 * re-fetch autoritativo, isStale a los 90 s con fake timers, caso 401
 * silencioso (callbacks jamás disparados) y cleanup completo.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useCongestionState } from '../useCongestionState';
import { congestionService } from '../../services/congestionService';
import { openCongestionStream } from '../../services/congestionSseClient';
import type { CongestionStreamOptions } from '../../services/congestionSseClient';
import type { CongestionStateResponse } from '../../types/congestion';

vi.mock('../../services/congestionService', () => ({
  congestionService: { getState: vi.fn() },
}));
vi.mock('../../services/congestionSseClient', () => ({
  openCongestionStream: vi.fn(),
}));

const getStateMock = congestionService.getState as unknown as ReturnType<typeof vi.fn>;
const openStreamMock = vi.mocked(openCongestionStream);

let captured: CongestionStreamOptions | null = null;
let streamController: AbortController | null = null;

function statePayload(level: number): CongestionStateResponse {
  return {
    count: 1,
    edges: [
      {
        edge_id: '-129822384#0',
        congestion_level: level,
        snapshot_timestamp: '2026-06-10T14:00:00',
      },
    ],
  };
}

beforeEach(() => {
  captured = null;
  streamController = null;
  getStateMock.mockReset();
  openStreamMock.mockReset();
  openStreamMock.mockImplementation((opts: CongestionStreamOptions) => {
    captured = opts;
    streamController = new AbortController();
    return streamController;
  });
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useCongestionState', () => {
  it('GET inicial + stream abierto; connection arranca en connecting', async () => {
    getStateMock.mockResolvedValue(statePayload(2));

    const { result } = renderHook(() => useCongestionState());
    expect(result.current.connection).toBe('connecting');

    await act(async () => {});

    expect(getStateMock).toHaveBeenCalledTimes(1);
    expect(openStreamMock).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual(statePayload(2));
    expect(result.current.isStale).toBe(false);
  });

  it('mapping de connection: onOpen → open, onError → retrying, onClose → closed', async () => {
    getStateMock.mockResolvedValue(statePayload(2));
    const { result } = renderHook(() => useCongestionState());
    await act(async () => {});

    act(() => captured!.onOpen!());
    expect(result.current.connection).toBe('open');

    act(() => captured!.onError!(new Error('stream caído')));
    expect(result.current.connection).toBe('retrying');

    act(() => captured!.onOpen!()); // reconexión exitosa
    expect(result.current.connection).toBe('open');

    act(() => captured!.onClose!());
    expect(result.current.connection).toBe('closed');
  });

  it('wake → re-fetch autoritativo silencioso (sin loading, data nueva)', async () => {
    getStateMock
      .mockResolvedValueOnce(statePayload(1))
      .mockResolvedValueOnce(statePayload(4));

    const { result } = renderHook(() => useCongestionState());
    await act(async () => {});
    expect(result.current.data?.edges[0].congestion_level).toBe(1);

    await act(async () => {
      captured!.onWake();
    });

    expect(getStateMock).toHaveBeenCalledTimes(2);
    expect(result.current.loading).toBe(false);
    expect(result.current.data?.edges[0].congestion_level).toBe(4);
  });

  it('isStale: false al éxito, true a los 90 s, false con wake exitoso y true de nuevo a los 90 s', async () => {
    vi.useFakeTimers();
    getStateMock.mockImplementation(() => Promise.resolve(statePayload(2)));

    const { result } = renderHook(() => useCongestionState());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.isStale).toBe(false);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(90_000);
    });
    expect(result.current.isStale).toBe(true);

    // Wake con fetch exitoso → lastUpdated avanza → vuelve a FRESCO.
    await act(async () => {
      captured!.onWake();
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.isStale).toBe(false);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(90_000);
    });
    expect(result.current.isStale).toBe(true);
  });

  it('staleAfterMs configurable', async () => {
    vi.useFakeTimers();
    getStateMock.mockResolvedValue(statePayload(2));

    const { result } = renderHook(() => useCongestionState({ staleAfterMs: 5_000 }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.isStale).toBe(false);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(result.current.isStale).toBe(true);
  });

  it('error de refetch en wake: conserva data stale y expone error', async () => {
    getStateMock
      .mockResolvedValueOnce(statePayload(3))
      .mockRejectedValueOnce(new Error('core caído'));

    const { result } = renderHook(() => useCongestionState());
    await act(async () => {});

    await act(async () => {
      captured!.onWake();
    });

    expect(result.current.data?.edges[0].congestion_level).toBe(3); // stale
    expect(result.current.error).toBe('core caído');
  });

  it('caso 401 silenciado (FatalSSEError): sin callbacks, connection queda congelada y el unmount no rompe', async () => {
    getStateMock.mockResolvedValue(statePayload(2));

    const { result, unmount } = renderHook(() => useCongestionState());
    await act(async () => {});

    // El client tragó el FatalSSEError: ni onError ni onClose llegan jamás.
    expect(result.current.connection).toBe('connecting');
    unmount(); // no lanza
  });

  it('cleanup: unmount aborta el stream y no filtra timers', async () => {
    vi.useFakeTimers();
    getStateMock.mockResolvedValue(statePayload(2));

    const { unmount } = renderHook(() => useCongestionState());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(streamController!.signal.aborted).toBe(false);

    unmount();

    expect(streamController!.signal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
});
