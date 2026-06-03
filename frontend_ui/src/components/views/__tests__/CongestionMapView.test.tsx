/**
 * CongestionMapView (HU-22) — smoke de render (Fase 2) + feed vivo (Fase 3).
 *
 * Patrón de mock de Leaflet (precedente para tests de mapa):
 *  - NO se monta Leaflet real: jsdom no tiene canvas ni dimensiones. Se stubea
 *    `react-leaflet` vía `vi.mock` hoisted, devolviendo divs con `data-testid`.
 *  - Los props del <GeoJSON> (`data`, `style`) se capturan con un holder de
 *    `vi.hoisted`. El mock cuenta montajes (useEffect) para detectar el REMONTE
 *    por `key` — la vía de recolorización de Fase 3.
 *  - `congestionService` y `congestionSseClient` se mockean para no pegar a la red
 *    ni abrir SSE real; el holder captura el callback `onWake` para simular wakes.
 *
 * Fase 2 afirma: monta sin crashear; carga geometry+state 1× cada uno; pasa 375
 * features + un `style` callback cableado a `congestionStyle`.
 * Fase 3 afirma: mount abre el stream; un wake re-lee SOLO state (geometry sigue
 * 1×) y recolorea (remonte); stale tras 90 s atenúa + muestra banner, y un wake
 * posterior lo limpia.
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import { CongestionMapView } from '../CongestionMapView';
import { congestionService } from '../../../services/congestionService';
import { openCongestionStream } from '../../../services/congestionSseClient';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
} from '../../../types/congestion';

const captured = vi.hoisted(() => ({
  props: null as null | Record<string, unknown>,
  mounts: 0,
}));

// Holder del stream SSE: captura el callback de wake y un spy de abort.
const sse = vi.hoisted(() => ({
  onWake: null as null | (() => void),
  abort: vi.fn(),
}));

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
  GeoJSON: (props: Record<string, unknown>) => {
    captured.props = props;
    // useEffect de montaje: incrementa en cada mount. Un cambio de `key`
    // (renderSeq o stale) desmonta+monta → cuenta sube = recolor por remonte.
    React.useEffect(() => {
      captured.mounts += 1;
    }, []);
    return <div data-testid="geojson" />;
  },
}));

vi.mock('../../../services/congestionService', () => ({
  congestionService: { getGeometry: vi.fn(), getState: vi.fn() },
}));

vi.mock('../../../services/congestionSseClient', () => ({
  openCongestionStream: vi.fn((opts: { onWake: () => void }) => {
    sse.onWake = opts.onWake;
    return { abort: sse.abort };
  }),
}));

const getGeometryMock = congestionService.getGeometry as unknown as ReturnType<typeof vi.fn>;
const getStateMock = congestionService.getState as unknown as ReturnType<typeof vi.fn>;
const openStreamMock = openCongestionStream as unknown as ReturnType<typeof vi.fn>;

const COUNT = 375;

function buildGeometry(): GeometryFeatureCollection {
  return {
    type: 'FeatureCollection',
    count: COUNT,
    features: Array.from({ length: COUNT }, (_, i) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'LineString' as const,
        coordinates: [
          [-77.0335, -12.118],
          [-77.0334, -12.1179],
        ] as [number, number][],
      },
      properties: {
        edge_id: `edge-${i}`,
        source_node: `src-${i}`,
        target_node: `dst-${i}`,
        distance_m: 100,
        lanes: 1,
      },
    })),
  };
}

/** Estado determinista; `shift` desplaza los niveles para distinguir snapshots. */
function buildState(shift = 0): CongestionStateResponse {
  return {
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      congestion_level: (i + shift) % 6,
      snapshot_timestamp: '2025-01-06T23:59:00',
    })),
  };
}

type CapturedData = { data: { features: { properties: { congestion_level: number } }[] } };
type StyleFn = (f: unknown) => { color: string; weight: number; opacity?: number };

describe('CongestionMapView', () => {
  beforeEach(() => {
    getGeometryMock.mockReset();
    getStateMock.mockReset();
    openStreamMock.mockClear();
    sse.onWake = null;
    sse.abort.mockClear();
    captured.props = null;
    captured.mounts = 0;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('carga geometría+estado 1× cada uno y pinta 375 tramos por <GeoJSON> con style callback', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());

    render(<CongestionMapView />);

    await waitFor(() => expect(screen.getByTestId('geojson')).toBeInTheDocument());
    expect(getGeometryMock).toHaveBeenCalledTimes(1);
    expect(getStateMock).toHaveBeenCalledTimes(1);

    const props = captured.props as CapturedData & { style: unknown };
    expect(props.data.features).toHaveLength(COUNT);

    expect(typeof props.style).toBe('function');
    const style = props.style as StyleFn;
    expect(style({ properties: { congestion_level: 3 } })).toEqual({
      color: '#F08C1D',
      weight: 6,
    });
  });

  it('muestra el bloque de error si falla la carga (no monta el mapa)', async () => {
    getGeometryMock.mockRejectedValue(new Error('boom'));
    getStateMock.mockResolvedValue(buildState());

    render(<CongestionMapView />);

    await waitFor(() =>
      expect(screen.getByText(/No se pudo cargar el mapa/i)).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('geojson')).not.toBeInTheDocument();
  });

  it('al montar abre el stream SSE de congestión (1×)', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());

    render(<CongestionMapView />);

    await waitFor(() => expect(screen.getByTestId('geojson')).toBeInTheDocument());
    expect(openStreamMock).toHaveBeenCalledTimes(1);
    expect(sse.onWake).toBeTypeOf('function');
  });

  it('un wake re-lee SOLO el estado (geometry sigue 1×) y recolorea por remonte', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    // Inicial: edge-0 nivel 0. Wake: niveles desplazados +3 → edge-0 nivel 3.
    getStateMock
      .mockResolvedValueOnce(buildState(0))
      .mockResolvedValueOnce(buildState(3));

    render(<CongestionMapView />);
    await waitFor(() => expect(screen.getByTestId('geojson')).toBeInTheDocument());

    const initial = captured.props as CapturedData;
    expect(initial.data.features[0].properties.congestion_level).toBe(0);
    const mountsBefore = captured.mounts;

    // Dispara un wake del feed.
    await act(async () => {
      sse.onWake?.();
    });

    await waitFor(() => {
      const after = captured.props as CapturedData;
      expect(after.data.features[0].properties.congestion_level).toBe(3);
    });

    // Re-lee state (2×) pero NO re-pide geometry (sigue 1×).
    expect(getStateMock).toHaveBeenCalledTimes(2);
    expect(getGeometryMock).toHaveBeenCalledTimes(1);
    // El remonte (key cambió) recoloreó la capa: contador de montajes subió.
    expect(captured.mounts).toBeGreaterThan(mountsBefore);
  });

  it('tras 90 s sin wake entra en stale (banner + opacidad atenuada) y un wake lo limpia', async () => {
    vi.useFakeTimers();
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());

    render(<CongestionMapView />);

    // Avanza 90 s: resuelve la carga inicial y dispara el timer de stale.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(90_000);
    });

    // Banner visible y tramos atenuados (opacity 0.5 manteniendo color+grosor).
    expect(screen.getByText(/datos de hace/i)).toBeInTheDocument();
    const staleStyle = (captured.props as { style: StyleFn }).style;
    expect(staleStyle({ properties: { congestion_level: 2 } }).opacity).toBe(0.5);

    // Un wake posterior limpia el stale: banner desaparece, sin opacidad.
    await act(async () => {
      sse.onWake?.();
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.queryByText(/datos de hace/i)).not.toBeInTheDocument();
    const liveStyle = (captured.props as { style: StyleFn }).style;
    expect(liveStyle({ properties: { congestion_level: 2 } }).opacity).toBeUndefined();
  });

  it('no re-pide la geometría en re-renders (la carga inicial corre una sola vez)', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());

    const { rerender } = render(<CongestionMapView />);
    await waitFor(() => expect(screen.getByTestId('geojson')).toBeInTheDocument());
    expect(getGeometryMock).toHaveBeenCalledTimes(1);

    // Un re-render no debe re-disparar el effect de carga inicial: vive en deps
    // estables (applyState), separado del feed SSE/stale (que depende de mode).
    rerender(<CongestionMapView />);
    rerender(<CongestionMapView />);
    await waitFor(() => expect(screen.getByTestId('geojson')).toBeInTheDocument());

    expect(getGeometryMock).toHaveBeenCalledTimes(1);
  });
});
