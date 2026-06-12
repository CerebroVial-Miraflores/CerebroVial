/**
 * Tests de useActiveStrategy (FASE 2 rediseño UI).
 *
 * Mock del service + sseClient con captura de callbacks. Clave: el evento
 * SSE es solo señal — se re-lee REST como fuente autoritativa (DHU-021 #15)
 * y el payload del evento se ignora.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useActiveStrategy } from '../useActiveStrategy';
import { controlActiveStateService } from '../../services/controlActiveStateService';
import type { ActiveStateResponse } from '../../services/controlActiveStateService';
import { openControlActiveStateStream } from '../../services/sseClient';
import type { SSEClientOptions } from '../../services/sseClient';

vi.mock('../../services/controlActiveStateService', () => ({
  controlActiveStateService: { getActiveState: vi.fn() },
}));
vi.mock('../../services/sseClient', () => ({
  openControlActiveStateStream: vi.fn(),
}));

const getActiveStateMock = controlActiveStateService.getActiveState as unknown as ReturnType<
  typeof vi.fn
>;
const openStreamMock = vi.mocked(openControlActiveStateStream);

const openedStreams: Array<{
  nodeId: string;
  opts: SSEClientOptions;
  controller: AbortController;
}> = [];

function activeState(cycle: number): ActiveStateResponse {
  return {
    node_id: 'larco_schell',
    strategy_mode: 'webster',
    cycle_seconds: cycle,
    phase_timings: [],
    decided_at: '2026-06-10T14:00:00',
    activated_at: '2026-06-10T14:00:01',
    activated_by: null,
  };
}

beforeEach(() => {
  openedStreams.length = 0;
  getActiveStateMock.mockReset();
  openStreamMock.mockReset();
  openStreamMock.mockImplementation((nodeId: string, opts: SSEClientOptions) => {
    const controller = new AbortController();
    openedStreams.push({ nodeId, opts, controller });
    return controller;
  });
});

describe('useActiveStrategy', () => {
  it('GET inicial con (nodeId, {signal}) + stream del mismo nodo', async () => {
    getActiveStateMock.mockResolvedValue(activeState(90));

    const { result } = renderHook(() => useActiveStrategy('larco_schell'));
    expect(result.current.connection).toBe('connecting');
    await act(async () => {});

    expect(result.current.data?.cycle_seconds).toBe(90);
    const [nodeId, opts] = getActiveStateMock.mock.calls[0] as [
      string,
      { signal: AbortSignal },
    ];
    expect(nodeId).toBe('larco_schell');
    expect(opts.signal).toBeInstanceOf(AbortSignal);
    expect(openedStreams).toHaveLength(1);
    expect(openedStreams[0].nodeId).toBe('larco_schell');
  });

  it('evento SSE → re-fetch autoritativo (el payload del evento se IGNORA)', async () => {
    getActiveStateMock
      .mockResolvedValueOnce(activeState(90))
      .mockResolvedValueOnce(activeState(120));

    const { result } = renderHook(() => useActiveStrategy('larco_schell'));
    await act(async () => {});

    await act(async () => {
      // Payload deliberadamente basura: el hook NO debe usarlo.
      openedStreams[0].opts.onMessage({ type: 'active-state-changed', data: { garbage: true } });
    });

    expect(getActiveStateMock).toHaveBeenCalledTimes(2);
    expect(result.current.data?.cycle_seconds).toBe(120);
  });

  it('mapping de connection: open / retrying / closed', async () => {
    getActiveStateMock.mockResolvedValue(activeState(90));
    const { result } = renderHook(() => useActiveStrategy('larco_schell'));
    await act(async () => {});

    act(() => openedStreams[0].opts.onOpen!());
    expect(result.current.connection).toBe('open');
    act(() => openedStreams[0].opts.onError!(new Error('x')));
    expect(result.current.connection).toBe('retrying');
    act(() => openedStreams[0].opts.onClose!());
    expect(result.current.connection).toBe('closed');
  });

  it('cambio de nodeId: aborta el stream viejo, abre el nuevo y la conexión vuelve a connecting', async () => {
    getActiveStateMock.mockResolvedValue(activeState(90));

    const { result, rerender } = renderHook(({ nodeId }) => useActiveStrategy(nodeId), {
      initialProps: { nodeId: 'larco_schell' },
    });
    await act(async () => {});
    act(() => openedStreams[0].opts.onOpen!());
    expect(result.current.connection).toBe('open');

    rerender({ nodeId: 'pardo_espinar' });
    await act(async () => {});

    expect(openedStreams[0].controller.signal.aborted).toBe(true);
    expect(openedStreams).toHaveLength(2);
    expect(openedStreams[1].nodeId).toBe('pardo_espinar');
    expect(getActiveStateMock.mock.calls.at(-1)?.[0]).toBe('pardo_espinar');
  });

  it("nodeId '' = disabled: ni GET ni stream", async () => {
    const { result } = renderHook(() => useActiveStrategy(''));
    await act(async () => {});

    expect(getActiveStateMock).not.toHaveBeenCalled();
    expect(openStreamMock).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);
  });

  it('error inicial: superficie {data:null, error} sin throw', async () => {
    getActiveStateMock.mockRejectedValue(new Error('nodo desconocido'));

    const { result } = renderHook(() => useActiveStrategy('nodo_x'));
    await act(async () => {});

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('nodo desconocido');
  });

  it('cleanup: unmount aborta el stream', async () => {
    getActiveStateMock.mockResolvedValue(activeState(90));

    const { unmount } = renderHook(() => useActiveStrategy('larco_schell'));
    await act(async () => {});

    unmount();
    expect(openedStreams[0].controller.signal.aborted).toBe(true);
  });
});
