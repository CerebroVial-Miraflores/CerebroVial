/**
 * CommandView — KPI strip (FASE 3): política de paridad por card con hooks de
 * datos mockeados a nivel módulo (real / real-con-caveat / mock / degradado).
 *
 * Mock per-file de react-leaflet (regla vigente) + leaflet (divIcon del
 * NodeMarker). ToastProvider en el harness (el provider real vive en AppShell).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';
import type { ReactNode } from 'react';

import { CommandView } from '../CommandView';
import { ToastProvider } from '../../../ui/Toast';
import { useIntersections } from '../../../../hooks/useIntersections';
import { useCongestionState } from '../../../../hooks/useCongestionState';
import { useCongestionGeometry } from '../../../../hooks/useCongestionGeometry';
import { useCongestionSeries } from '../../../../hooks/useCongestionSeries';
import { useCongestionPrediction } from '../../../../hooks/useCongestionPrediction';
import { useVisionAggregates } from '../../../../hooks/useVisionAggregates';
import { useAdaptiveNodes } from '../useAdaptiveNodes';

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
  GeoJSON: () => <div data-testid="geojson-layer" />,
  Marker: () => <div data-testid="marker" />,
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

const GEO = {
  type: 'FeatureCollection' as const,
  count: 2,
  features: ['e1', 'e2'].map((edge_id) => ({
    type: 'Feature' as const,
    geometry: { type: 'LineString' as const, coordinates: [[0, 0], [1, 1]] as [number, number][] },
    properties: { edge_id, source_node: 'n1', target_node: 'n2', distance_m: 100, lanes: 2 },
  })),
};

const INTERSECTIONS = [
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
];

function setDefaults() {
  intersectionsMock.mockReturnValue(res(INTERSECTIONS));
  stateMock.mockReturnValue({
    ...res({ edges: [], count: 0 }),
    isStale: false,
    connection: 'open' as const,
  });
  geometryMock.mockReturnValue(res(GEO));
  seriesMock.mockReturnValue(res(null, { lastUpdated: null }));
  predictionMock.mockReturnValue(
    res(null, { error: 'El servidor respondió 503.', errorStatus: 503 }),
  );
  visionMock.mockReturnValue({
    byCamera: {},
    aggregate: { meanSpeedKmh: null, totalFlowVph: null, camerasWithSignal: 0 },
  });
  adaptiveMock.mockReturnValue({
    nodes: [],
    activeCount: 2,
    loading: false,
    refetch: vi.fn(),
  });
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
  render(<RouterProvider router={router} />);
  return router;
}

beforeEach(() => {
  vi.useFakeTimers();
  setDefaults();
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('CommandView — KPIs (política de paridad)', () => {
  it('idx REAL: media de niveles del estado → /100 con countUp', () => {
    stateMock.mockReturnValue({
      ...res({
        edges: [
          { edge_id: 'e1', congestion_level: 2, snapshot_timestamp: '2026-06-08T12:00:00' },
          { edge_id: 'e2', congestion_level: 3, snapshot_timestamp: '2026-06-08T12:00:00' },
        ],
        count: 2,
      }),
      isStale: false,
      connection: 'open' as const,
    });
    mount();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    // (2+3)/2 / 5 * 100 = 50
    expect(screen.getByText('50')).toBeInTheDocument();
    expect(screen.getByText('/100')).toBeInTheDocument();
  });

  it('idx con ventana vacía (0 aristas) → "—" + nota honesta, y el mapa avisa', () => {
    mount();
    expect(screen.getByText('Índice de congestión · red')).toBeInTheDocument();
    expect(screen.getByText('sin datos en ventana')).toBeInTheDocument();
    expect(
      screen.getByText('Sin datos de congestión en la ventana actual — la red pinta neutra.'),
    ).toBeInTheDocument();
  });

  it('vel/flu sin señal → "—" + caveat colapsado a UNA línea con title (B0)', () => {
    mount();
    // Empty deliberado: una sola línea corta por card; el texto completo va en
    // el title (tooltip). Caveats de limitación SIN DemoBadge (no es mock).
    const velCaveat = screen.getByText('sin señal · sin calibrar');
    expect(velCaveat).toHaveAttribute('title', expect.stringContaining('DEUDA-SPEED-CALIB'));
    const fluCaveat = screen.getByText('sin señal · presencia extrapolada');
    expect(fluCaveat).toHaveAttribute('title', expect.stringContaining('line-crossing'));
  });

  it('vel/flu con señal → media y suma del agregado de visión', () => {
    visionMock.mockReturnValue({
      byCamera: {
        cam_a: { speed: 20, flow: 300, at: 1 },
        cam_b: { speed: 30, flow: 700, at: 2 },
      },
      aggregate: { meanSpeedKmh: 25, totalFlowVph: 1000, camerasWithSignal: 2 },
    });
    mount();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByText('25.0')).toBeInTheDocument();
    expect(screen.getByText('km/h')).toBeInTheDocument();
    expect(screen.getByText('1,000')).toBeInTheDocument();
    expect(screen.getByText('veh/h')).toBeInTheDocument();
    // Los caveats real-con-caveat no desaparecen con dato presente.
    expect(screen.getByText('visión · sin calibrar')).toBeInTheDocument();
    expect(screen.getByText('visión · presencia extrapolada')).toBeInTheDocument();
  });

  it('dem MOCK lleva DemoBadge además del DemoBadge del panel de alertas', () => {
    mount();
    // 1 en la card de demora + 1 en el header del panel de alertas.
    expect(screen.getAllByText('Demo · datos simulados').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('Demora media · críticos')).toBeInTheDocument();
    expect(screen.getByText('s/veh')).toBeInTheDocument();
  });

  it('sem REAL-PARCIAL: N del one-shot sobre nodos instrumentados', () => {
    mount();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByText('Cruces en modo adaptativo')).toBeInTheDocument();
    expect(screen.getByText('/5')).toBeInTheDocument();
    expect(screen.getByText('nodos instrumentados')).toBeInTheDocument();
  });

  it('pred degradado (503) → "Servicio no disponible" + Reintentar dispara refetch sin cambiar el modo', () => {
    const refetch = vi.fn(async () => {});
    predictionMock.mockReturnValue(
      res(null, { error: 'El servidor respondió 503.', errorStatus: 503, refetch }),
    );
    const router = mount();
    expect(screen.getByText('Servicio no disponible')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Reintentar' }));
    expect(refetch).toHaveBeenCalledTimes(1);
    // stopPropagation: el retry NO navega al modo predicción.
    expect(router.state.location.search).toBe('');
  });

  it('pred con dato → etiqueta semántica de la media al horizonte +15', () => {
    predictionMock.mockReturnValue(
      res({
        base_timestep: 720,
        horizon: 30,
        source: 'seed051 (day_idx=9)',
        source_date: '2026-06-05',
        count: 1,
        edges: [{ edge_id: 'e1', levels: Array(30).fill(3) }],
      }),
    );
    mount();
    expect(screen.getByText('Demora alta')).toBeInTheDocument();
  });

  it('click en la card pred (con dato) cambia el mapa a modo predicción; degradada NO es clickeable', () => {
    predictionMock.mockReturnValue(
      res({
        base_timestep: 720,
        horizon: 30,
        source: 'seed051 (day_idx=9)',
        source_date: '2026-06-05',
        count: 1,
        edges: [{ edge_id: 'e1', levels: Array(30).fill(3) }],
      }),
    );
    const router = mount();
    fireEvent.click(screen.getByText('Predicción 15 min · red'));
    expect(new URLSearchParams(router.state.location.search).get('modo')).toBe('prediccion');
  });

  it('la card pred degradada no es botón (sin nesting con Reintentar)', () => {
    mount();
    const label = screen.getByText('Predicción 15 min · red');
    expect(label.closest('button')).toBeNull();
  });
});
