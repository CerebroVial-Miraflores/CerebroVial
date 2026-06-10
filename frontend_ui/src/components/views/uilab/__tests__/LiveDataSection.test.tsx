/**
 * Tests de LiveDataSection (FASE 2 rediseño UI).
 *
 * Los 8 hooks de datos van mockeados a nivel módulo con retornos controlados:
 * variante feliz (conteos, chips de connection/staleness) y variante
 * backend-apagado (cada tarjeta muestra su error, ninguna tumba a las demás).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { LiveDataSection } from '../LiveDataSection';
import { useIntersections } from '../../../../hooks/useIntersections';
import { useCongestionGeometry } from '../../../../hooks/useCongestionGeometry';
import { useCongestionState } from '../../../../hooks/useCongestionState';
import { useCongestionSeries } from '../../../../hooks/useCongestionSeries';
import { useCongestionPrediction } from '../../../../hooks/useCongestionPrediction';
import { useActiveStrategy } from '../../../../hooks/useActiveStrategy';
import { useVisionStream } from '../../../../hooks/useVisionStream';
import { usePredictionHistory } from '../../../../hooks/usePredictionHistory';
import type { RestResource } from '../../../../hooks/types';

vi.mock('../../../../hooks/useIntersections', () => ({ useIntersections: vi.fn() }));
vi.mock('../../../../hooks/useCongestionGeometry', () => ({ useCongestionGeometry: vi.fn() }));
vi.mock('../../../../hooks/useCongestionState', () => ({ useCongestionState: vi.fn() }));
vi.mock('../../../../hooks/useCongestionSeries', () => ({ useCongestionSeries: vi.fn() }));
vi.mock('../../../../hooks/useCongestionPrediction', () => ({ useCongestionPrediction: vi.fn() }));
vi.mock('../../../../hooks/useActiveStrategy', () => ({ useActiveStrategy: vi.fn() }));
vi.mock('../../../../hooks/useVisionStream', () => ({ useVisionStream: vi.fn() }));
vi.mock('../../../../hooks/usePredictionHistory', () => ({ usePredictionHistory: vi.fn() }));

function resource<T>(over: Partial<RestResource<T>> = {}): RestResource<T> {
  return {
    data: null,
    loading: false,
    error: null,
    lastUpdated: null,
    refetch: vi.fn().mockResolvedValue(undefined),
    ...over,
  };
}

const intersectionsMock = vi.mocked(useIntersections);
const geometryMock = vi.mocked(useCongestionGeometry);
const stateMock = vi.mocked(useCongestionState);
const seriesMock = vi.mocked(useCongestionSeries);
const predictionMock = vi.mocked(useCongestionPrediction);
const strategyMock = vi.mocked(useActiveStrategy);
const visionMock = vi.mocked(useVisionStream);
const historyMock = vi.mocked(usePredictionHistory);

function arrangeHappy() {
  intersectionsMock.mockReturnValue(
    resource({
      data: [
        {
          id: 'cam_01',
          name: 'Av. Larco / Schell',
          speed: 22,
          flow: 540,
          status: 'Moderado',
          lat: -12.121,
          lng: -77.029,
          stream_url: null,
        },
      ],
      lastUpdated: 1760000000000,
    }),
  );
  geometryMock.mockReturnValue(
    resource({
      data: { type: 'FeatureCollection', count: 1660, features: new Array(1660).fill(null) },
    } as never),
  );
  stateMock.mockReturnValue({
    ...resource({
      data: { count: 980, edges: new Array(980).fill(null) } as never,
      lastUpdated: 1760000000000,
    }),
    isStale: false,
    connection: 'open',
  });
  seriesMock.mockReturnValue(
    resource({
      data: {
        day: '2026-06-10',
        t0: '2026-06-10T00:00:00',
        step_s: 300,
        coverage_end: null,
        count: 288,
        edges: [],
      },
    }),
  );
  predictionMock.mockReturnValue(
    resource({
      data: {
        base_timestep: 720,
        horizon: 30,
        source: 'seed051 (day_idx=9)',
        source_date: '2026-06-10',
        count: 375,
        edges: [],
      },
    }),
  );
  strategyMock.mockReturnValue({
    ...resource({
      data: {
        node_id: 'larco_schell',
        strategy_mode: 'webster',
        cycle_seconds: 90,
        phase_timings: [],
        decided_at: '2026-06-10T14:00:00',
        activated_at: '2026-06-10T14:00:01',
        activated_by: null,
      },
    }),
    connection: 'open',
  });
  historyMock.mockReturnValue(
    resource({
      data: { camera_id: 'cam_01', history: [], prediction: undefined },
    }),
  );
  visionMock.mockReturnValue({ data: null, lastUpdated: null, connection: 'closed' });
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('LiveDataSection — variante feliz', () => {
  it('muestra conteos, chips de conexión/staleness y la guía de rol operator', () => {
    arrangeHappy();
    render(<LiveDataSection />);

    expect(screen.getByText('1660')).toBeInTheDocument(); // tramos de geometría
    expect(screen.getByText('980')).toBeInTheDocument(); // aristas con estado
    expect(screen.getByText('FRESCO')).toBeInTheDocument();
    expect(screen.getAllByText('SSE OPEN').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('webster')).toBeInTheDocument();
    expect(screen.getByText(/rol operator/i)).toBeInTheDocument();
    expect(screen.queryByText('Demo · datos simulados')).not.toBeInTheDocument();
  });

  it('history y visión reusan el primer id de intersections', () => {
    arrangeHappy();
    render(<LiveDataSection />);

    expect(historyMock).toHaveBeenCalledWith('cam_01', 5);
    expect(visionMock).toHaveBeenCalledWith('cam_01', { enabled: false });
  });

  it('"Actualizar" llama al refetch de intersections', () => {
    arrangeHappy();
    const refetch = vi.fn().mockResolvedValue(undefined);
    intersectionsMock.mockReturnValue(resource({ data: [], refetch }));
    render(<LiveDataSection />);

    fireEvent.click(screen.getByRole('button', { name: 'Actualizar' }));
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  it('el toggle "Conectar al edge" habilita useVisionStream', () => {
    arrangeHappy();
    render(<LiveDataSection />);

    fireEvent.click(screen.getByRole('button', { name: 'Conectar al edge' }));

    const lastCall = visionMock.mock.calls.at(-1);
    expect(lastCall?.[1]).toEqual({ enabled: true });
  });
});

describe('LiveDataSection — backend apagado', () => {
  it('cada tarjeta muestra su error con Reintentar; ninguna tumba a las demás', () => {
    const failed = () =>
      resource({ error: 'Error de red al conectar con el servidor.' });
    intersectionsMock.mockReturnValue(failed());
    geometryMock.mockReturnValue(failed());
    stateMock.mockReturnValue({ ...failed(), isStale: false, connection: 'retrying' });
    seriesMock.mockReturnValue(failed());
    predictionMock.mockReturnValue(failed());
    strategyMock.mockReturnValue({ ...failed(), connection: 'retrying' });
    historyMock.mockReturnValue(resource()); // disabled: sin id de cámara
    visionMock.mockReturnValue({ data: null, lastUpdated: null, connection: 'closed' });

    render(<LiveDataSection />);

    expect(screen.getAllByText('Error de red al conectar con el servidor.')).toHaveLength(6);
    expect(screen.getAllByRole('button', { name: 'Reintentar' })).toHaveLength(6);
    expect(screen.getAllByText('SSE RETRYING').length).toBe(2);
    expect(screen.getByText(/Esperando la lista de cámaras/)).toBeInTheDocument();
  });
});
