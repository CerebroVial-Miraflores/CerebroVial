/**
 * Tests de useVisionAggregates (FASE 3 rediseño UI).
 *
 * El stub global de EventSource (setupTests.ts) es inerte; acá se pisa
 * per-file con un FakeEventSource (patrón useVisionStream.test). Semántica de
 * agregación: media de speeds no-null + suma de flows; sin muestras → nulls
 * (la card muestra "—" + "sin señal de visión").
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';

import {
  aggregateVisionSamples,
  useVisionAggregates,
} from '../useVisionAggregates';
import type { VisionStreamPayload } from '../../types/visionStream';

class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  static instances: FakeEventSource[] = [];

  url: string;
  readyState = 0;
  closed = false;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private listeners = new Map<string, Set<(ev: MessageEvent) => void>>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(cb);
  }

  removeEventListener(type: string, cb: (ev: MessageEvent) => void) {
    this.listeners.get(type)?.delete(cb);
  }

  close() {
    this.closed = true;
    this.readyState = FakeEventSource.CLOSED;
  }

  emit(type: string, data: string) {
    this.listeners.get(type)?.forEach((cb) => cb({ data } as MessageEvent));
  }
}

function payload(cameraId: string, speed: number | null, flow: number): string {
  const p: VisionStreamPayload = {
    schema_version: '1.0',
    event_type: 'traffic_update',
    server_timestamp: '2026-06-10T14:00:00Z',
    camera: { id: cameraId, street_monitored: null },
    zone: { id: 'zone-1' },
    window: {
      start: '2026-06-10T13:59:00Z',
      end: '2026-06-10T14:00:00Z',
      duration_seconds: 60,
    },
    metrics: {
      unique_vehicles: 5,
      vehicles_by_type: { car: 5 },
      mean_speed_kmh: speed,
      flow_vehicles_per_hour: flow,
      mean_occupancy: 0.4,
      density_vehicles_per_km: null,
    },
  };
  return JSON.stringify(p);
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('aggregateVisionSamples (pura)', () => {
  it('sin muestras → nulls y 0 cámaras', () => {
    expect(aggregateVisionSamples({})).toEqual({
      meanSpeedKmh: null,
      totalFlowVph: null,
      camerasWithSignal: 0,
    });
  });

  it('media de speeds no-null + suma de flows; speed null no promedia ni anula', () => {
    const agg = aggregateVisionSamples({
      a: { speed: 20, flow: 300, at: 1 },
      b: { speed: 30, flow: 500, at: 2 },
      c: { speed: null, flow: 200, at: 3 },
    });
    expect(agg.meanSpeedKmh).toBe(25);
    expect(agg.totalFlowVph).toBe(1000);
    expect(agg.camerasWithSignal).toBe(3);
  });

  it('todas las cámaras sin calibración (speed null) → meanSpeed null pero flow real', () => {
    const agg = aggregateVisionSamples({ a: { speed: null, flow: 120, at: 1 } });
    expect(agg.meanSpeedKmh).toBeNull();
    expect(agg.totalFlowVph).toBe(120);
  });
});

describe('useVisionAggregates', () => {
  it('abre un EventSource por cámara contra /stream/{id}', () => {
    renderHook(() => useVisionAggregates(['cam_a', 'cam_b']));
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(FakeEventSource.instances[0].url.endsWith('/stream/cam_a')).toBe(true);
    expect(FakeEventSource.instances[1].url.endsWith('/stream/cam_b')).toBe(true);
  });

  it('traffic_update por cámara puebla byCamera y el agregado', () => {
    const { result } = renderHook(() => useVisionAggregates(['cam_a', 'cam_b']));

    act(() => {
      FakeEventSource.instances[0].emit('traffic_update', payload('cam_a', 20, 300));
      FakeEventSource.instances[1].emit('traffic_update', payload('cam_b', 30, 500));
    });

    expect(result.current.byCamera.cam_a).toMatchObject({ speed: 20, flow: 300 });
    expect(result.current.aggregate.meanSpeedKmh).toBe(25);
    expect(result.current.aggregate.totalFlowVph).toBe(800);
    expect(result.current.aggregate.camerasWithSignal).toBe(2);
  });

  it('payload malformado se ignora: la muestra previa queda', () => {
    const { result } = renderHook(() => useVisionAggregates(['cam_a']));
    act(() => {
      FakeEventSource.instances[0].emit('traffic_update', payload('cam_a', 18, 200));
    });
    act(() => {
      FakeEventSource.instances[0].emit('traffic_update', '{no es json');
    });
    expect(result.current.byCamera.cam_a.flow).toBe(200);
  });

  it('sin muestras → agregado null (camino "sin señal de visión")', () => {
    const { result } = renderHook(() => useVisionAggregates(['cam_a']));
    expect(result.current.aggregate.meanSpeedKmh).toBeNull();
    expect(result.current.aggregate.totalFlowVph).toBeNull();
  });

  it('cambio del set de cámaras cierra los streams viejos, abre nuevos y resetea muestras', () => {
    const { result, rerender } = renderHook(({ ids }) => useVisionAggregates(ids), {
      initialProps: { ids: ['cam_a'] as readonly string[] },
    });
    act(() => {
      FakeEventSource.instances[0].emit('traffic_update', payload('cam_a', 20, 300));
    });
    expect(result.current.aggregate.camerasWithSignal).toBe(1);

    rerender({ ids: ['cam_b', 'cam_c'] });

    expect(FakeEventSource.instances[0].closed).toBe(true);
    expect(FakeEventSource.instances).toHaveLength(3);
    expect(result.current.aggregate.camerasWithSignal).toBe(0);
  });

  it('misma lista con identidad nueva NO reabre streams', () => {
    const { rerender } = renderHook(({ ids }) => useVisionAggregates(ids), {
      initialProps: { ids: ['cam_a', 'cam_b'] as readonly string[] },
    });
    rerender({ ids: ['cam_a', 'cam_b'] });
    expect(FakeEventSource.instances).toHaveLength(2);
  });

  it('unmount cierra todos los streams; lista vacía no abre ninguno', () => {
    const { unmount } = renderHook(() => useVisionAggregates(['cam_a', 'cam_b']));
    unmount();
    expect(FakeEventSource.instances.every((s) => s.closed)).toBe(true);

    FakeEventSource.instances = [];
    renderHook(() => useVisionAggregates([]));
    expect(FakeEventSource.instances).toHaveLength(0);
  });
});
