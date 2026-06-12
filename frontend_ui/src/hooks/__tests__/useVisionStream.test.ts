/**
 * Tests de useVisionStream (FASE 2 rediseño UI).
 *
 * El stub global de EventSource (setupTests.ts) es inerte a propósito; acá se
 * pisa per-file con un FakeEventSource que registra listeners y permite
 * despachar eventos manualmente (emit/open/fail) — el patrón previsto por el
 * propio setupTests ("configurable + writable para que los tests... puedan
 * seguir pisándolos").
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useVisionStream } from '../useVisionStream';
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

  open() {
    this.readyState = FakeEventSource.OPEN;
    this.onopen?.();
  }

  fail(opts?: { closed?: boolean }) {
    this.readyState = opts?.closed ? FakeEventSource.CLOSED : FakeEventSource.CONNECTING;
    this.onerror?.();
  }
}

function payload(vehicles: number): string {
  const p: VisionStreamPayload = {
    schema_version: '1.0',
    event_type: 'traffic_update',
    server_timestamp: '2026-06-10T14:00:00Z',
    camera: { id: 'cam1', street_monitored: null },
    zone: { id: 'zone-1' },
    window: {
      start: '2026-06-10T13:59:00Z',
      end: '2026-06-10T14:00:00Z',
      duration_seconds: 60,
    },
    metrics: {
      unique_vehicles: vehicles,
      vehicles_by_type: { car: vehicles },
      mean_speed_kmh: 24.5,
      flow_vehicles_per_hour: vehicles * 60,
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

describe('useVisionStream', () => {
  it('abre EventSource contra /stream/{cameraId} del edge y arranca connecting', () => {
    const { result } = renderHook(() => useVisionStream('cam1'));

    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.instances[0].url.endsWith('/stream/cam1')).toBe(true);
    expect(result.current.connection).toBe('connecting');
    expect(result.current.data).toBeNull();
  });

  it('traffic_update válido → payload tipado y lastUpdated', () => {
    const { result } = renderHook(() => useVisionStream('cam1'));

    act(() => {
      FakeEventSource.instances[0].emit('traffic_update', payload(12));
    });

    expect(result.current.data?.metrics.unique_vehicles).toBe(12);
    expect(result.current.data?.event_type).toBe('traffic_update');
    expect(result.current.lastUpdated).toBeTypeOf('number');
  });

  it('payload malformado se ignora: la data previa queda intacta, sin throw', () => {
    const { result } = renderHook(() => useVisionStream('cam1'));

    act(() => {
      FakeEventSource.instances[0].emit('traffic_update', payload(7));
    });
    act(() => {
      FakeEventSource.instances[0].emit('traffic_update', '{no es json');
    });

    expect(result.current.data?.metrics.unique_vehicles).toBe(7);
  });

  it('connection: onopen → open; onerror reintentando → retrying; onerror cerrado → closed', () => {
    const { result } = renderHook(() => useVisionStream('cam1'));

    act(() => FakeEventSource.instances[0].open());
    expect(result.current.connection).toBe('open');

    act(() => FakeEventSource.instances[0].fail());
    expect(result.current.connection).toBe('retrying');

    act(() => FakeEventSource.instances[0].fail({ closed: true }));
    expect(result.current.connection).toBe('closed');
  });

  it('cleanup: unmount cierra el EventSource', () => {
    const { unmount } = renderHook(() => useVisionStream('cam1'));

    unmount();
    expect(FakeEventSource.instances[0].closed).toBe(true);
  });

  it('cambio de cameraId: cierra el stream viejo, abre el nuevo y resetea data/conexión', () => {
    const { result, rerender } = renderHook(({ id }) => useVisionStream(id), {
      initialProps: { id: 'cam1' },
    });
    act(() => {
      FakeEventSource.instances[0].open();
      FakeEventSource.instances[0].emit('traffic_update', payload(5));
    });
    expect(result.current.data).not.toBeNull();

    rerender({ id: 'cam2' });

    expect(FakeEventSource.instances[0].closed).toBe(true);
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(FakeEventSource.instances[1].url.endsWith('/stream/cam2')).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.connection).toBe('connecting');
  });

  it("enabled: false o cameraId '' → cero streams y connection closed", () => {
    const disabled = renderHook(() => useVisionStream('cam1', { enabled: false }));
    const empty = renderHook(() => useVisionStream(''));

    expect(FakeEventSource.instances).toHaveLength(0);
    expect(disabled.result.current.connection).toBe('closed');
    expect(empty.result.current.connection).toBe('closed');
  });

  it('enabled false → true abre el stream recién ahí (gate del alta al edge)', () => {
    const { rerender } = renderHook(
      ({ enabled }) => useVisionStream('cam1', { enabled }),
      { initialProps: { enabled: false } },
    );
    expect(FakeEventSource.instances).toHaveLength(0);

    rerender({ enabled: true });
    expect(FakeEventSource.instances).toHaveLength(1);
  });
});
