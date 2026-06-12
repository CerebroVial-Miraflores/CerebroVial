/**
 * Tests del one-shot del KPI "cruces en modo adaptativo" (FASE 3, D1).
 * Service mockeado a nivel módulo; clasificación por respuesta-contrato:
 * fulfilled → active, 404 → no-strategy (contrato no_active_state), resto → failed.
 */
import { describe, expect, it, beforeEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { AxiosError, AxiosHeaders } from 'axios';

import { useAdaptiveNodes } from '../useAdaptiveNodes';
import { controlActiveStateService } from '../../../../services/controlActiveStateService';
import { KNOWN_NODE_IDS } from '../../control/controlTypes';

vi.mock('../../../../services/controlActiveStateService', () => ({
  controlActiveStateService: { getActiveState: vi.fn() },
}));

const getMock = controlActiveStateService.getActiveState as unknown as ReturnType<typeof vi.fn>;

function http(status: number): AxiosError {
  const config = { headers: new AxiosHeaders() };
  return new AxiosError('falló', 'ERR_BAD_REQUEST', config, {}, {
    data: status === 404 ? { code: 'no_active_state' } : {},
    status,
    statusText: 'x',
    headers: {},
    config,
  } as never);
}

function activeState(nodeId: string) {
  return {
    node_id: nodeId,
    strategy_mode: 'webster' as const,
    cycle_seconds: 90,
    phase_timings: [{ phase_id: 'NS', green: 42, yellow: 4, all_red: 2 }],
    decided_at: '2026-06-10T12:00:00',
    activated_at: '2026-06-10T12:00:05',
    activated_by: null,
  };
}

beforeEach(() => {
  getMock.mockReset();
});

describe('useAdaptiveNodes', () => {
  it('consulta los 5 KNOWN_NODE_IDS una sola vez y clasifica active/no-strategy/failed', async () => {
    getMock.mockImplementation((nodeId: string) => {
      if (nodeId === KNOWN_NODE_IDS[0]) return Promise.resolve(activeState(nodeId));
      if (nodeId === KNOWN_NODE_IDS[1]) return Promise.resolve(activeState(nodeId));
      if (nodeId === KNOWN_NODE_IDS[2]) return Promise.reject(http(404));
      if (nodeId === KNOWN_NODE_IDS[3]) return Promise.reject(http(404));
      return Promise.reject(new Error('red caída'));
    });

    const { result } = renderHook(() => useAdaptiveNodes());
    expect(result.current.loading).toBe(true);
    expect(result.current.activeCount).toBeNull();

    await act(async () => {});

    expect(getMock).toHaveBeenCalledTimes(KNOWN_NODE_IDS.length);
    expect(result.current.loading).toBe(false);
    expect(result.current.activeCount).toBe(2);
    expect(result.current.nodes?.map((n) => n.kind)).toEqual([
      'active',
      'active',
      'no-strategy',
      'no-strategy',
      'failed',
    ]);
    expect(result.current.nodes?.[0].state?.strategy_mode).toBe('webster');
    expect(result.current.nodes?.[2].state).toBeNull();
  });

  it('refetch re-consulta los 5 nodos conservando la lista stale durante el vuelo', async () => {
    getMock.mockResolvedValue(activeState('x'));
    const { result } = renderHook(() => useAdaptiveNodes());
    await act(async () => {});
    expect(result.current.activeCount).toBe(5);

    act(() => {
      result.current.refetch();
    });
    // Stale visible mientras settlea (sin flash a null).
    expect(result.current.activeCount).toBe(5);
    await act(async () => {});
    expect(getMock).toHaveBeenCalledTimes(KNOWN_NODE_IDS.length * 2);
  });

  it('unmount con vuelo pendiente no produce setState (signal abortado)', async () => {
    let resolveAll: (() => void) | null = null;
    getMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveAll = () => resolve(activeState('x'));
        }),
    );
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const { unmount } = renderHook(() => useAdaptiveNodes());
    unmount();
    await act(async () => {
      resolveAll?.();
    });
    const actWarnings = errorSpy.mock.calls.filter((args) =>
      String(args[0]).includes('act'),
    );
    expect(actWarnings).toHaveLength(0);
    errorSpy.mockRestore();
  });
});
