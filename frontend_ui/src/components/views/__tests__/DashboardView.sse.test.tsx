import { render, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// react-leaflet: jsdom no monta Leaflet real. Pasamos los componentes como passthrough y
// useMap devuelve los métodos que tocan MapUpdater (flyTo) y FitBounds (fitBounds).
vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  TileLayer: () => null,
  Marker: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  Popup: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  Tooltip: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  useMap: () => ({ flyTo: vi.fn(), fitBounds: vi.fn() }),
}));

// leaflet: el módulo usa L.icon / L.Marker.prototype al importar y L.divIcon / L.latLngBounds
// en render. Stubs no-op bastan: no verificamos el mapa, sí el refresco por SSE.
vi.mock('leaflet', () => {
  const L = {
    icon: vi.fn(() => ({})),
    divIcon: vi.fn(() => ({})),
    latLngBounds: vi.fn(() => ({})),
    Marker: { prototype: { options: {} as Record<string, unknown> } },
  };
  return { default: L, ...L };
});

// Captura el onWake del SSE de congestión para dispararlo a mano.
const sse = vi.hoisted(() => ({ onWake: undefined as undefined | (() => void), abort: vi.fn() }));
vi.mock('../../../services/congestionSseClient', () => ({
  openCongestionStream: (opts: { onWake: () => void }) => {
    sse.onWake = opts.onWake;
    return { abort: sse.abort } as unknown as AbortController;
  },
}));

import { DashboardView } from '../DashboardView';

const INTERSECTIONS = [
  { id: 'larco_benavides', name: 'Larco Benavides', lat: -12.1, lng: -77.0, status: 'critical', stream_url: null, speed: 0, flow: 0 },
];

describe('DashboardView — refresco por SSE de congestión (B2)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sse.onWake = undefined;
    // EventSource de visión (edge): jsdom no lo trae; stub que no hace nada.
    class FakeEventSource {
      addEventListener() {}
      close() {}
      set onerror(_v: unknown) {}
    }
    // @ts-expect-error inyección del stub global
    globalThis.EventSource = FakeEventSource;
    // IntersectionObserver: jsdom no lo trae; la grilla de cámaras (tab default 'cameras')
    // lo usa para el lazy-por-viewport. Stub no-op: este test verifica el refresco SSE, no el lazy.
    class FakeIntersectionObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    // @ts-expect-error inyección del stub global
    globalThis.IntersectionObserver = FakeIntersectionObserver;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => INTERSECTIONS,
    })) as unknown as typeof fetch;
  });

  it('hace el fetch inicial y re-consulta /api/intersections en cada wake del SSE', async () => {
    render(<DashboardView onSelectCamera={vi.fn()} />);

    // Fetch inicial.
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    });
    expect((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0]).toContain(
      '/api/intersections',
    );
    expect(sse.onWake).toBeDefined();

    // Wake del feed de congestión → re-fetch.
    await act(async () => {
      sse.onWake!();
    });
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    });
  });
});
