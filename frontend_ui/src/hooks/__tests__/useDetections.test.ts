/**
 * Tests de useDetections (FASE 4 Mitad A, D-019).
 *
 * Polling GET /detections/{id}/latest con `fetch` mockeado. Cubre: gate `enabled`
 * (no pollea sin alta), payload fresco (cajas visibles), stale por edad de caja
 * (server - frame > 3 s → sin cajas), payload vacío y error de red.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';

import { useDetections } from '../useDetections';
import type { DetectionsPayload } from '../../types/detections';

function box(id = 'v1') {
  return { id, type: 'car', confidence: 0.9, bbox: [0.1, 0.1, 0.2, 0.2] as [number, number, number, number] };
}

function payload(over: Partial<DetectionsPayload>): DetectionsPayload {
  return {
    camera_id: 'cam1',
    frame: { width: 1280, height: 720 },
    frame_timestamp: 100,
    server_timestamp: 101, // edad 1 s → fresco
    detection_ran: true,
    detections: [box()],
    ...over,
  };
}

function mockFetch(p: DetectionsPayload, ok = true) {
  const fn = vi.fn(() =>
    Promise.resolve({ ok, status: ok ? 200 : 500, json: () => Promise.resolve(p) } as Response),
  );
  globalThis.fetch = fn as unknown as typeof fetch;
  return fn;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useDetections', () => {
  it('no pollea cuando enabled=false (sin alta)', async () => {
    const fn = mockFetch(payload({}));
    renderHook(() => useDetections('cam1', { enabled: false }));
    await new Promise((r) => setTimeout(r, 20));
    expect(fn).not.toHaveBeenCalled();
  });

  it('payload fresco: cajas visibles y fresh=true', async () => {
    mockFetch(payload({ detections: [box('v1'), box('v2')] }));
    const { result } = renderHook(() => useDetections('cam1', { enabled: true }));

    await waitFor(() => expect(result.current.boxes).toHaveLength(2));
    expect(result.current.fresh).toBe(true);
    expect(result.current.ageSeconds).toBe(1);
    expect(result.current.frame).toEqual({ width: 1280, height: 720 });
  });

  it('stale por edad (server - frame > 3 s): sin cajas, fresh=false', async () => {
    mockFetch(payload({ frame_timestamp: 100, server_timestamp: 110 })); // edad 10 s
    const { result } = renderHook(() => useDetections('cam1', { enabled: true }));

    await waitFor(() => expect(result.current.ageSeconds).toBe(10));
    expect(result.current.fresh).toBe(false);
    expect(result.current.boxes).toEqual([]);
  });

  it('payload vacío (cámara recién activada): sin cajas, frame null', async () => {
    mockFetch(payload({ frame: null, frame_timestamp: null, detections: [] }));
    const { result } = renderHook(() => useDetections('cam1', { enabled: true }));

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    expect(result.current.boxes).toEqual([]);
    expect(result.current.fresh).toBe(false);
    expect(result.current.frame).toBeNull();
  });

  it('error de red: sin cajas (no congela cajas viejas)', async () => {
    mockFetch(payload({}), false); // res.ok=false → throw → hasError
    const { result } = renderHook(() => useDetections('cam1', { enabled: true }));

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    expect(result.current.fresh).toBe(false);
    expect(result.current.boxes).toEqual([]);
  });
});
