/**
 * CommandView — modos del mapa por query params (FASE 3).
 *
 * Mock per-file de react-leaflet con vi.hoisted: captura props del GeoJSON y
 * CUENTA MONTAJES (el recolor por remonte-con-key es contrato — ver
 * edgeStyle.ts). Hooks de datos mockeados a nivel módulo; el "wake SSE" se
 * simula avanzando lastUpdated del estado y re-renderizando.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';
import { useEffect, type ReactNode } from 'react';

import { CommandView } from '../CommandView';
import { ToastProvider } from '../../../ui/Toast';
import { useIntersections } from '../../../../hooks/useIntersections';
import { useCongestionState } from '../../../../hooks/useCongestionState';
import { useCongestionGeometry } from '../../../../hooks/useCongestionGeometry';
import { useCongestionSeries } from '../../../../hooks/useCongestionSeries';
import { useCongestionPrediction } from '../../../../hooks/useCongestionPrediction';
import { useVisionAggregates } from '../../../../hooks/useVisionAggregates';
import { useAdaptiveNodes } from '../useAdaptiveNodes';

const captured = vi.hoisted(() => ({
  geo: null as Record<string, unknown> | null,
  mounts: 0,
  markerClicks: [] as Array<() => void>,
}));

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
  GeoJSON: (props: Record<string, unknown>) => {
    captured.geo = props;
    useEffect(() => {
      captured.mounts += 1;
    }, []);
    return <div data-testid="geojson-layer" />;
  },
  Marker: (props: { eventHandlers?: { click?: () => void } }) => {
    if (props.eventHandlers?.click) captured.markerClicks.push(props.eventHandlers.click);
    return <div data-testid="marker" />;
  },
}));
vi.mock('leaflet', () => ({ divIcon: vi.fn(() => ({})) }));

vi.mock('../../../../hooks/useIntersections', () => ({ useIntersections: vi.fn() }));
vi.mock('../../../../hooks/useCongestionState', () => ({ useCongestionState: vi.fn() }));
vi.mock('../../../../hooks/useCongestionGeometry', () => ({ useCongestionGeometry: vi.fn() }));
vi.mock('../../../../hooks/useCongestionSeries', () => ({ useCongestionSeries: vi.fn() }));
vi.mock('../../../../hooks/useCongestionPrediction', () => ({ useCongestionPrediction: vi.fn() }));
vi.mock('../../../../hooks/useVisionAggregates', () => ({ useVisionAggregates: vi.fn() }));
vi.mock('../useAdaptiveNodes', () => ({ useAdaptiveNodes: vi.fn() }));

const intersectionsMock = vi.mocked(useIntersections);
const stateMock = vi.mocked(useCongestionState);
const geometryMock = vi.mocked(useCongestionGeometry);
const seriesMock = vi.mocked(useCongestionSeries);
const predictionMock = vi.mocked(useCongestionPrediction);
const visionMock = vi.mocked(useVisionAggregates);
const adaptiveMock = vi.mocked(useAdaptiveNodes);

function res<T>(data: T | null, extra: Record<string, unknown> = {}) {
  return {
    data,
    loading: false,
    error: null as string | null,
    errorStatus: null as number | null,
    lastUpdated: data !== null ? 1_700_000_000_000 : null,
    refetch: vi.fn(async () => {}),
    ...extra,
  };
}

// Fixture chico (~2 aristas): la fidelidad del merge la cubren congestion.test
// y derive.test — acá solo el cableado y el remonte.
const GEO = {
  type: 'FeatureCollection' as const,
  count: 2,
  features: ['e1', 'e2'].map((edge_id) => ({
    type: 'Feature' as const,
    geometry: { type: 'LineString' as const, coordinates: [[0, 0], [1, 1]] as [number, number][] },
    properties: { edge_id, source_node: 'n1', target_node: 'n2', distance_m: 100, lanes: 2 },
  })),
};

const STATE = {
  edges: [
    { edge_id: 'e1', congestion_level: 4, snapshot_timestamp: '2026-06-08T12:00:00' },
    { edge_id: 'e2', congestion_level: 1, snapshot_timestamp: '2026-06-08T12:00:00' },
  ],
  count: 2,
};

const SERIES = {
  day: '2026-06-05',
  t0: '2026-06-05T08:00:00',
  step_s: 3600,
  coverage_end: '2026-06-05T10:00:00',
  count: 2,
  edges: [
    { edge_id: 'e1', levels: [0, 3, 5] },
    { edge_id: 'e2', levels: [1, 2, 4] },
  ],
};

const PREDICTION = {
  base_timestep: 720,
  horizon: 30,
  source: 'seed051 (day_idx=9)',
  source_date: '2026-06-05',
  count: 2,
  edges: [
    { edge_id: 'e1', levels: Array(30).fill(2) },
    { edge_id: 'e2', levels: Array(30).fill(4) },
  ],
};

function stateRes(extra: Record<string, unknown> = {}) {
  return { ...res(STATE), isStale: false, connection: 'open' as const, ...extra };
}

function setDefaults() {
  intersectionsMock.mockReturnValue(
    res([
      {
        id: 'cam_larco_benavides',
        name: 'Larco × Benavides',
        speed: 0,
        flow: 0,
        status: 'critical',
        lat: -12.13,
        lng: -77.02,
        stream_url: null,
      },
    ]),
  );
  stateMock.mockReturnValue(stateRes());
  geometryMock.mockReturnValue(res(GEO));
  seriesMock.mockReturnValue(res(null, { lastUpdated: null }));
  predictionMock.mockReturnValue(
    res(null, { error: 'El servidor respondió 503.', errorStatus: 503 }),
  );
  visionMock.mockReturnValue({
    byCamera: {},
    aggregate: { meanSpeedKmh: null, totalFlowVph: null, camerasWithSignal: 0 },
  });
  adaptiveMock.mockReturnValue({ nodes: [], activeCount: 0, loading: false, refetch: vi.fn() });
}

function mount(initialEntry = '/') {
  const router = createMemoryRouter(
    [
      {
        path: '/',
        element: (
          <ToastProvider>
            <CommandView />
          </ToastProvider>
        ),
      },
      { path: '/camara/:id', element: <div data-testid="puente-camara" /> },
    ],
    { initialEntries: [initialEntry] },
  );
  const view = render(<RouterProvider router={router} />);
  return { router, view };
}

function geoFeatures() {
  const data = captured.geo?.data as { features: { properties: Record<string, unknown> }[] };
  return data.features;
}

beforeEach(() => {
  vi.useFakeTimers();
  captured.geo = null;
  captured.mounts = 0;
  captured.markerClicks = [];
  setDefaults();
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('modo Ahora (default)', () => {
  it('pinta el merge geometry × state y monta la capa una vez', () => {
    mount();
    expect(captured.mounts).toBe(1);
    const features = geoFeatures();
    expect(features.find((f) => f.properties.edge_id === 'e1')?.properties.congestion_level).toBe(4);
    expect(screen.getByText('EN VIVO')).toBeInTheDocument();
    expect(screen.getByText('Observado')).toBeInTheDocument();
  });

  it('wake SSE (lastUpdated avanza) → remonta la capa (recolor por key)', () => {
    mount();
    expect(captured.mounts).toBe(1);

    // El wake real refetchea y avanza lastUpdated; acá se actualiza el mock y
    // se fuerza un re-render de la vista con una interacción inocua (toggle de
    // Semáforos, que NO toca la capa GeoJSON). La key live-<lastUpdated> nueva
    // debe remontar la capa.
    stateMock.mockReturnValue(stateRes({ lastUpdated: 1_700_000_060_000 }));
    fireEvent.click(screen.getByRole('button', { name: 'Semáforos' }));

    expect(captured.mounts).toBe(2);
  });

  it('isStale → aviso "DATOS VIEJOS" visible', () => {
    stateMock.mockReturnValue(stateRes({ isStale: true }));
    mount();
    expect(screen.getByText('DATOS VIEJOS')).toBeInTheDocument();
  });

  it('apagar la capa Tráfico desmonta el GeoJSON; Semáforos controla los markers', () => {
    mount();
    expect(screen.getByTestId('geojson-layer')).toBeInTheDocument();
    expect(screen.getAllByTestId('marker').length).toBe(1);

    fireEvent.click(screen.getByRole('button', { name: 'Tráfico' }));
    expect(screen.queryByTestId('geojson-layer')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Semáforos' }));
    expect(screen.queryByTestId('marker')).not.toBeInTheDocument();
  });

  it('click en un nodo navega al puente /camara/:id (costura 3A)', () => {
    const { router } = mount();
    act(() => {
      captured.markerClicks[0]();
    });
    expect(router.state.location.pathname).toBe('/camara/cam_larco_benavides');
  });
});

describe('modo Histórico (?modo=historico&dia=&t=)', () => {
  it('pinta la serie del día en el índice t y etiqueta la hora (contrato t0 + i*step_s)', () => {
    seriesMock.mockReturnValue(res(SERIES));
    mount('/?modo=historico&dia=2026-06-05&t=1');

    expect(seriesMock).toHaveBeenCalledWith('2026-06-05');
    expect(geoFeatures().find((f) => f.properties.edge_id === 'e1')?.properties.congestion_level).toBe(3);
    // t0 08:00 + 1*3600s = 09:00 (slider label + badge del modo)
    expect(screen.getAllByText(/09:00/).length).toBeGreaterThanOrEqual(1);
  });

  it('mover el slider escribe ?t= y remonta la capa con el nivel nuevo', () => {
    seriesMock.mockReturnValue(res(SERIES));
    const { router } = mount('/?modo=historico&dia=2026-06-05');
    const mountsBefore = captured.mounts;

    fireEvent.change(screen.getByLabelText('Paso temporal'), { target: { value: '2' } });

    expect(new URLSearchParams(router.state.location.search).get('t')).toBe('2');
    expect(captured.mounts).toBeGreaterThan(mountsBefore);
    expect(geoFeatures().find((f) => f.properties.edge_id === 'e1')?.properties.congestion_level).toBe(5);
  });

  it('?t= fuera de rango se clampea al último índice de la serie', () => {
    seriesMock.mockReturnValue(res(SERIES));
    mount('/?modo=historico&dia=2026-06-05&t=999');
    expect(screen.getByLabelText('Paso temporal')).toHaveValue('2');
    expect(geoFeatures().find((f) => f.properties.edge_id === 'e1')?.properties.congestion_level).toBe(5);
  });

  it('día sin datos (t0 null) → empty honesto, sin capa', () => {
    seriesMock.mockReturnValue(res({ ...SERIES, t0: null, step_s: null, edges: [] }));
    mount('/?modo=historico&dia=2026-06-09');
    expect(screen.getByText('El día 2026-06-09 no tiene datos de congestión.')).toBeInTheDocument();
    expect(screen.queryByTestId('geojson-layer')).not.toBeInTheDocument();
  });
});

describe('modo Predicción (?modo=prediccion)', () => {
  it('hoy 503 → error elegante + Reintentar, sin capa y sin mock', () => {
    mount('/?modo=prediccion');
    expect(screen.getByText('Servicio de predicción no disponible.')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Reintentar' }).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByTestId('geojson-layer')).not.toBeInTheDocument();
  });

  it('con dato pinta el horizonte +15 y el hchip +30 remonta la capa', () => {
    predictionMock.mockReturnValue(res(PREDICTION));
    mount('/?modo=prediccion');
    expect(screen.getByText('PREDICCIÓN · +15 MIN')).toBeInTheDocument();
    expect(screen.getByText('Demora prevista')).toBeInTheDocument();
    const mountsBefore = captured.mounts;

    fireEvent.click(screen.getByRole('button', { name: '+30' }));

    expect(captured.mounts).toBeGreaterThan(mountsBefore);
    expect(screen.getByText('PREDICCIÓN · +30 MIN')).toBeInTheDocument();
  });
});

describe('SegmentedControl de modos', () => {
  it('cambia el modo por la URL y de vuelta a Ahora limpia dia/t', () => {
    seriesMock.mockReturnValue(res(SERIES));
    const { router } = mount('/?modo=historico&dia=2026-06-05&t=2');

    fireEvent.click(screen.getByRole('button', { name: '● Ahora' }));

    expect(router.state.location.search).toBe('');
    expect(screen.getByText('EN VIVO')).toBeInTheDocument();
  });
});
