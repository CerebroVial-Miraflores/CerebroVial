/**
 * CongestionMapView (HU-23, Fase 3) — modo histórico: toggle + slider + repintado.
 *
 * Archivo SEPARADO del smoke de modo vivo (`CongestionMapView.test.tsx`) a
 * propósito: el modo histórico NO debe afectar al vivo, así que sus tests viven
 * aparte y los de HU-22 quedan intactos. Se calca el patrón de mock de aquel
 * (react-leaflet stubeado con captura de `props`/`mounts`; `congestionService` y
 * `congestionSseClient` mockeados), sumando `getSeries` al service.
 *
 * Afirma: en histórico el SSE se silencia (abort al togglear) y no reabre; mover
 * el slider remonta y recolorea por el índice; un día sin datos (count 0) muestra
 * ausencia sin error y deshabilita el slider; volver a vivo reactiva el SSE y
 * re-lee estado; y el atenuado por stale NO aplica en histórico (gate por `mode`).
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor, act, fireEvent } from '@testing-library/react';
import { CongestionMapView } from '../CongestionMapView';
import { congestionService } from '../../../services/congestionService';
import { openCongestionStream } from '../../../services/congestionSseClient';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
  CongestionSeriesResponse,
} from '../../../types/congestion';

const captured = vi.hoisted(() => ({
  props: null as null | Record<string, unknown>,
  mounts: 0,
}));

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
    React.useEffect(() => {
      captured.mounts += 1;
    }, []);
    return <div data-testid="geojson" />;
  },
}));

vi.mock('../../../services/congestionService', () => ({
  congestionService: { getGeometry: vi.fn(), getState: vi.fn(), getSeries: vi.fn() },
}));

vi.mock('../../../services/congestionSseClient', () => ({
  openCongestionStream: vi.fn((opts: { onWake: () => void }) => {
    sse.onWake = opts.onWake;
    return { abort: sse.abort };
  }),
}));

const getGeometryMock = congestionService.getGeometry as unknown as ReturnType<typeof vi.fn>;
const getStateMock = congestionService.getState as unknown as ReturnType<typeof vi.fn>;
const getSeriesMock = congestionService.getSeries as unknown as ReturnType<typeof vi.fn>;
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

function buildState(): CongestionStateResponse {
  return {
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      congestion_level: i % 6,
      snapshot_timestamp: '2025-01-06T23:59:00',
    })),
  };
}

/** Serie de 4 muestras/arista, niveles distintos por índice (para distinguir el
 *  recolor del slider): edge-i → levels = [i, i+1, i+2, i+3] (mod 6). */
function buildSeries(): CongestionSeriesResponse {
  return {
    day: '2025-01-06',
    t0: '2025-01-06T00:00:00',
    step_s: 60,
    coverage_end: '2025-01-06T00:03:00',
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      levels: [i % 6, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6],
    })),
  };
}

/** Serie de día vacío (CA-23.7): sin muestras, sin t0/step_s. */
function buildEmptySeries(): CongestionSeriesResponse {
  return {
    day: '2026-06-02',
    t0: null,
    step_s: null,
    coverage_end: null,
    count: 0,
    edges: [],
  };
}

type CapturedData = { data: { features: { properties: { congestion_level: number } }[] } };
type StyleFn = (f: unknown) => { color: string; weight: number; opacity?: number };

async function renderAndLoad() {
  render(<CongestionMapView />);
  await waitFor(() => expect(screen.getByTestId('geojson')).toBeInTheDocument());
}

function clickHistoric() {
  fireEvent.click(screen.getByRole('button', { name: /histórico/i }));
}
function clickLive() {
  fireEvent.click(screen.getByRole('button', { name: /vivo/i }));
}

describe('CongestionMapView — modo histórico (HU-23)', () => {
  beforeEach(() => {
    getGeometryMock.mockReset();
    getStateMock.mockReset();
    getSeriesMock.mockReset();
    openStreamMock.mockClear();
    sse.onWake = null;
    sse.abort.mockClear();
    captured.props = null;
    captured.mounts = 0;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('al togglear a histórico abortea el stream SSE y no lo reabre', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());

    await renderAndLoad();
    expect(openStreamMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      clickHistoric();
    });

    // El cleanup del Effect B (vivo) abortea; en histórico NO se reabre stream.
    expect(sse.abort).toHaveBeenCalled();
    expect(openStreamMock).toHaveBeenCalledTimes(1);
  });

  it('mover el slider remonta la capa y recolorea por el índice de la serie', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());

    await renderAndLoad();

    await act(async () => {
      clickHistoric();
    });

    // Cargada la serie: índice 0 → edge-0 nivel levels[0] = 0.
    await waitFor(() => {
      const after = captured.props as CapturedData;
      expect(after.data.features[0].properties.congestion_level).toBe(0);
    });

    const mountsBefore = captured.mounts;
    const slider = screen.getByLabelText('Hora del día') as HTMLInputElement;
    expect(slider.disabled).toBe(false);

    // Mover el slider al índice 2 → edge-0 nivel levels[2] = 2, y remonte (key).
    await act(async () => {
      fireEvent.change(slider, { target: { value: '2' } });
    });

    await waitFor(() => {
      const after = captured.props as CapturedData;
      expect(after.data.features[0].properties.congestion_level).toBe(2);
    });
    expect(captured.mounts).toBeGreaterThan(mountsBefore);
  });

  it('día sin datos (count 0): slider deshabilitado + mensaje de ausencia, sin error', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildEmptySeries());

    await renderAndLoad();

    await act(async () => {
      clickHistoric();
    });

    await waitFor(() =>
      expect(screen.getByText(/no hay datos para este día/i)).toBeInTheDocument(),
    );
    const slider = screen.getByLabelText('Hora del día') as HTMLInputElement;
    expect(slider.disabled).toBe(true);
    // Día vacío NO es error de carga.
    expect(screen.queryByText(/no se pudo cargar/i)).not.toBeInTheDocument();
  });

  it("volver a 'live' reabre el stream SSE y re-lee el estado", async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());

    await renderAndLoad();
    expect(getStateMock).toHaveBeenCalledTimes(1);
    expect(openStreamMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      clickHistoric();
    });
    await act(async () => {
      clickLive();
    });

    // Reabre el stream (2×) y re-lee el estado al regresar a vivo.
    await waitFor(() => expect(openStreamMock).toHaveBeenCalledTimes(2));
    expect(getStateMock).toHaveBeenCalledTimes(2);
  });

  it('en histórico el atenuado por stale NO se aplica (gate de estilo por mode)', async () => {
    vi.useFakeTimers();
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());

    render(<CongestionMapView />);

    // 90 s sin wake → stale en modo vivo (atenúa con opacity 0.5).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(90_000);
    });
    const staleStyle = (captured.props as { style: StyleFn }).style;
    expect(staleStyle({ properties: { congestion_level: 2 } }).opacity).toBe(0.5);

    // Togglear a histórico: aunque `stale` siga en true, el gate por `mode` evita
    // el atenuado (en histórico no hay feed que envejezca).
    await act(async () => {
      clickHistoric();
      await vi.advanceTimersByTimeAsync(0);
    });

    const histStyle = (captured.props as { style: StyleFn }).style;
    expect(histStyle({ properties: { congestion_level: 2 } }).opacity).toBeUndefined();
  });
});
