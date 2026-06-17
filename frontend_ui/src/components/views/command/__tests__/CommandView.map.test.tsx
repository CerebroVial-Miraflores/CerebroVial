/**
 * CommandView — modos del mapa por query params (FASE 4 migración MapLibre).
 *
 * Mock per-file de react-map-gl/maplibre con vi.hoisted: captura el `data` del
 * <Source> (FeatureCollection mergeada). El recolor ya NO es por remonte-con-key
 * (murió geoJsonKeyFor): es por setData → el contrato acá es que el data del
 * Source se actualiza con el merge correcto en cada cambio (wake, slider, modo).
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

const captured = vi.hoisted(() => ({
  sourceData: null as Record<string, unknown> | null,
  markerClicks: [] as Array<(e: { originalEvent: { stopPropagation: () => void } }) => void>,
}));

vi.mock('react-map-gl/maplibre', () => ({
  Map: ({ children }: { children?: ReactNode }) => <div data-testid="map">{children}</div>,
  Source: ({ data, children }: { data?: Record<string, unknown>; children?: ReactNode }) => {
    captured.sourceData = data ?? null;
    return <div data-testid="edge-source">{children}</div>;
  },
  Layer: () => <div data-testid="edge-layer" />,
  Marker: ({
    onClick,
    children,
  }: {
    onClick?: (e: { originalEvent: { stopPropagation: () => void } }) => void;
    children?: ReactNode;
  }) => {
    if (onClick) captured.markerClicks.push(onClick);
    return <div data-testid="marker">{children}</div>;
  },
  Popup: ({ children }: { children?: ReactNode }) => <div data-testid="popup">{children}</div>,
}));

// Overlays stubeados: el drawer real monta HLS + useActiveStrategy (servicios
// reales) y tiene test propio (IntersectionDrawer.test); acá solo el cableado.
vi.mock('../IntersectionDrawer', () => ({
  IntersectionDrawer: (props: { cameraId: string | null }) =>
    props.cameraId !== null ? <div data-testid="drawer-stub">{props.cameraId}</div> : null,
}));
vi.mock('../KpiModal', () => ({
  KpiModal: (props: { kind: string | null }) =>
    props.kind !== null ? <div data-testid="kpi-modal-stub">{props.kind}</div> : null,
}));

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
// y derive.test — acá solo el cableado y el setData.
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

// Estado posterior a un wake: e1 baja de 4 a 2 (para verificar el setData).
const STATE_WAKE = {
  edges: [
    { edge_id: 'e1', congestion_level: 2, snapshot_timestamp: '2026-06-08T12:01:00' },
    { edge_id: 'e2', congestion_level: 1, snapshot_timestamp: '2026-06-08T12:01:00' },
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
  const data = captured.sourceData as { features: { properties: Record<string, unknown> }[] } | null;
  return data?.features ?? [];
}

function levelOf(edgeId: string): unknown {
  return geoFeatures().find((f) => f.properties.edge_id === edgeId)?.properties.congestion_level;
}

beforeEach(() => {
  vi.useFakeTimers();
  captured.sourceData = null;
  captured.markerClicks = [];
  setDefaults();
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('modo Ahora (default)', () => {
  it('pinta el merge geometry × state en el data del Source', () => {
    mount();
    expect(levelOf('e1')).toBe(4);
    expect(screen.getByText('EN VIVO')).toBeInTheDocument();
    expect(screen.getByText('Observado')).toBeInTheDocument();
  });

  it('wake SSE (estado nuevo) → el Source recibe el merge actualizado (setData, sin remonte)', () => {
    mount();
    expect(levelOf('e1')).toBe(4);

    // El wake real refetchea y trae un estado nuevo; acá se actualiza el mock y
    // se fuerza un re-render con una interacción inocua (toggle de Semáforos, que
    // NO toca la capa de aristas). El data del Source debe reflejar el nivel nuevo.
    stateMock.mockReturnValue(stateRes({ data: STATE_WAKE, lastUpdated: 1_700_000_060_000 }));
    fireEvent.click(screen.getByRole('button', { name: 'Semáforos' }));

    expect(levelOf('e1')).toBe(2);
  });

  it('isStale → aviso "DATOS VIEJOS" visible', () => {
    stateMock.mockReturnValue(stateRes({ isStale: true }));
    mount();
    expect(screen.getByText('DATOS VIEJOS')).toBeInTheDocument();
  });

  it('apagar la capa Tráfico desmonta la capa de aristas; Semáforos controla los markers', () => {
    mount();
    expect(screen.getByTestId('edge-layer')).toBeInTheDocument();
    expect(screen.getAllByTestId('marker').length).toBe(1);

    fireEvent.click(screen.getByRole('button', { name: 'Tráfico' }));
    expect(screen.queryByTestId('edge-layer')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Semáforos' }));
    expect(screen.queryByTestId('marker')).not.toBeInTheDocument();
  });

  it('click en un nodo abre el drawer por ?nodo= (B3 — reemplaza la costura 3A)', () => {
    const { router } = mount();
    act(() => {
      captured.markerClicks[0]({ originalEvent: { stopPropagation: () => {} } });
    });
    expect(router.state.location.pathname).toBe('/');
    expect(new URLSearchParams(router.state.location.search).get('nodo')).toBe(
      'cam_larco_benavides',
    );
    expect(screen.getByTestId('drawer-stub')).toHaveTextContent('cam_larco_benavides');
  });
});

describe('modo Histórico (?modo=historico&dia=&t=)', () => {
  it('pinta la serie del día en el índice t y etiqueta la hora (contrato t0 + i*step_s)', () => {
    seriesMock.mockReturnValue(res(SERIES));
    mount('/?modo=historico&dia=2026-06-05&t=1');

    expect(seriesMock).toHaveBeenCalledWith('2026-06-05');
    expect(levelOf('e1')).toBe(3);
    // t0 08:00 + 1*3600s = 09:00 (slider label + badge del modo)
    expect(screen.getAllByText(/09:00/).length).toBeGreaterThanOrEqual(1);
  });

  it('mover el slider escribe ?t= y el Source recibe el nivel nuevo', () => {
    seriesMock.mockReturnValue(res(SERIES));
    const { router } = mount('/?modo=historico&dia=2026-06-05');

    fireEvent.change(screen.getByLabelText('Paso temporal'), { target: { value: '2' } });

    expect(new URLSearchParams(router.state.location.search).get('t')).toBe('2');
    expect(levelOf('e1')).toBe(5);
  });

  it('?t= fuera de rango se clampea al último índice de la serie', () => {
    seriesMock.mockReturnValue(res(SERIES));
    mount('/?modo=historico&dia=2026-06-05&t=999');
    expect(screen.getByLabelText('Paso temporal')).toHaveValue('2');
    expect(levelOf('e1')).toBe(5);
  });

  it('día sin datos (t0 null) → empty honesto, sin capa', () => {
    seriesMock.mockReturnValue(res({ ...SERIES, t0: null, step_s: null, edges: [] }));
    mount('/?modo=historico&dia=2026-06-09');
    expect(screen.getByText('El día 2026-06-09 no tiene datos de congestión.')).toBeInTheDocument();
    expect(screen.queryByTestId('edge-layer')).not.toBeInTheDocument();
  });
});

describe('modo Predicción (?modo=prediccion)', () => {
  it('hoy 503 → error elegante + Reintentar, sin capa y sin mock', () => {
    mount('/?modo=prediccion');
    expect(screen.getByText('Servicio de predicción no disponible.')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Reintentar' }).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByTestId('edge-layer')).not.toBeInTheDocument();
  });

  it('con dato pinta el horizonte +15 con la leyenda de predicción; +30 cambia el badge', () => {
    predictionMock.mockReturnValue(res(PREDICTION));
    mount('/?modo=prediccion');
    expect(screen.getByText('PREDICCIÓN · +15 MIN')).toBeInTheDocument();
    // Leyenda propia del modo predicción (PREDICTION_LEGEND), no la del estado vivo.
    expect(screen.getByText('Demora prevista')).toBeInTheDocument();
    expect(levelOf('e1')).toBe(2);

    fireEvent.click(screen.getByRole('button', { name: '+30' }));

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
