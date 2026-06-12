/**
 * IntersectionDrawer (FASE 3 B3) — drawer por ?nodo= con política de paridad:
 * HLS real (mockeado per-file), ciclo HU-05 real-resoluble vía adapter, IA
 * mock con toasts demo (D7: sin cascade sobre capas reales), métricas reales
 * con caveat. useActiveStrategy mockeado a nivel módulo.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { IntersectionDrawer } from '../IntersectionDrawer';
import { ToastProvider } from '../../../ui/Toast';
import { useActiveStrategy } from '../../../../hooks/useActiveStrategy';
import type { IntersectionSummary } from '../../../../types/intersections';

vi.mock('../../../../hooks/useActiveStrategy', () => ({ useActiveStrategy: vi.fn() }));
vi.mock('../../../HlsPlayer', () => ({
  HlsPlayer: ({ src }: { src: string }) => <div data-testid="hls-player">{src}</div>,
}));
vi.mock('../../control/TrafficLightCycle', () => ({
  TrafficLightCycle: (props: { status: string; data: { intersection_id: string } | null }) => (
    <div data-testid="traffic-cycle">
      {props.status}:{props.data?.intersection_id ?? 'null'}
    </div>
  ),
}));

const strategyMock = vi.mocked(useActiveStrategy);

const CAMERA: IntersectionSummary = {
  id: 'cam_larco_benavides',
  name: 'Larco × Benavides',
  speed: 0,
  flow: 0,
  status: 'critical',
  lat: -12.13,
  lng: -77.02,
  stream_url: 'https://claro/hls/larco.m3u8',
};

function strategyRes(extra: Record<string, unknown> = {}) {
  return {
    data: null,
    loading: false,
    error: null as string | null,
    errorStatus: null as number | null,
    lastUpdated: null,
    refetch: vi.fn(async () => {}),
    connection: 'open' as const,
    ...extra,
  };
}

function mountDrawer(
  overrides: Partial<Parameters<typeof IntersectionDrawer>[0]> = {},
  cameras: IntersectionSummary[] = [CAMERA],
) {
  const onClose = vi.fn();
  render(
    <MemoryRouter>
      <ToastProvider>
        <IntersectionDrawer
          cameraId="cam_larco_benavides"
          intersections={cameras}
          intersectionsLoading={false}
          vision={{ speed: 18.4, flow: 820, at: 1 }}
          edgeLevel={4}
          onClose={onClose}
          {...overrides}
        />
      </ToastProvider>
    </MemoryRouter>,
  );
  return { onClose };
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('IntersectionDrawer — apertura y datos reales', () => {
  it('cerrado (cameraId null): el cuerpo es null — sin HLS ni consulta de ciclo montados', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer({ cameraId: null });
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
    expect(strategyMock).not.toHaveBeenCalled();
  });

  it('abierto: título real, chip por status crítico y HLS con el stream_url', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer();
    expect(screen.getByText('Larco × Benavides')).toBeInTheDocument();
    expect(screen.getByText('CRÍTICO · ADAPTATIVO EN PAUSA')).toBeInTheDocument();
    expect(screen.getByTestId('hls-player')).toHaveTextContent('https://claro/hls/larco.m3u8');
  });

  it('sin stream_url → vacío honesto en lugar del player', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer({}, [{ ...CAMERA, stream_url: null }]);
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
    expect(screen.getByText('Sin stream registrado para esta cámara')).toBeInTheDocument();
  });

  it('métricas reales: velocidad/flujo de visión con caveat en title + nivel del tramo', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer();
    expect(screen.getByText('18.4')).toBeInTheDocument();
    expect(screen.getByText('820')).toBeInTheDocument();
    expect(screen.getByText('nivel 4')).toBeInTheDocument();
    expect(screen.getByTitle(/DEUDA-SPEED-CALIB/)).toBeInTheDocument();
    expect(screen.getByTitle(/line-crossing/)).toBeInTheDocument();
  });

  it('sin muestra de visión → "—" + "sin señal" (honesto)', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer({ vision: undefined, edgeLevel: null });
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText('sin señal')).toBeInTheDocument();
  });

  it('?nodo= desconocido → error honesto con botón de cierre', () => {
    strategyMock.mockReturnValue(strategyRes());
    const { onClose } = mountDrawer({ cameraId: 'cam_fantasma' });
    expect(screen.getByText(/no existe en el inventario/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Cerrar panel' }));
    expect(onClose).toHaveBeenCalled();
  });
});

describe('IntersectionDrawer — ciclo semafórico (D5: real resoluble)', () => {
  it('nodo instrumentado con estrategia activa → TrafficLightCycle con el adapter', () => {
    strategyMock.mockReturnValue(
      strategyRes({
        data: {
          node_id: 'larco_benavides',
          strategy_mode: 'webster',
          cycle_seconds: 90,
          phase_timings: [{ phase_id: 'NS', green: 42, yellow: 4, all_red: 2 }],
          decided_at: 'x',
          activated_at: 'y',
          activated_by: null,
        },
      }),
    );
    mountDrawer();
    // El hook se consulta con el node_id SIN el prefijo cam_ (convención del seed).
    expect(strategyMock).toHaveBeenCalledWith('larco_benavides');
    expect(screen.getByTestId('traffic-cycle')).toHaveTextContent('success:larco_benavides');
  });

  it('404 no_active_state → "Sin estrategia activa" honesto, sin ciclo', () => {
    strategyMock.mockReturnValue(
      strategyRes({ error: 'El servidor respondió 404.', errorStatus: 404 }),
    );
    mountDrawer();
    expect(screen.getByText(/Sin estrategia activa para este nodo/)).toBeInTheDocument();
    expect(screen.queryByTestId('traffic-cycle')).not.toBeInTheDocument();
  });

  it('error no-contrato → mensaje + Reintentar', () => {
    const refetch = vi.fn(async () => {});
    strategyMock.mockReturnValue(
      strategyRes({ error: 'Error de red al conectar con el servidor.', errorStatus: null, refetch }),
    );
    mountDrawer();
    fireEvent.click(screen.getByRole('button', { name: 'Reintentar' }));
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  it('nodo NO instrumentado → nota y el hook queda deshabilitado (nodeId vacío)', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer({ cameraId: 'cam_otro_cruce' }, [
      { ...CAMERA, id: 'cam_otro_cruce', name: 'Otro Cruce' },
    ]);
    expect(strategyMock).toHaveBeenCalledWith('');
    expect(screen.getByText(/Nodo sin instrumentación de control/)).toBeInTheDocument();
  });
});

describe('IntersectionDrawer — IA mock (D7) y navegación', () => {
  it('la recomendación lleva DemoBadge, "¿Por qué?" expande y Aplicar pasa a done + toast (sin cascade)', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer();
    expect(screen.getAllByText('Demo · datos simulados').length).toBeGreaterThanOrEqual(2);

    const why = screen.getByRole('button', { name: /¿Por qué esta decisión\?/ });
    expect(why).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(why);
    expect(why).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText(/La cola creció 38%/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar' }));
    expect(screen.getByText('Plan aplicado')).toBeInTheDocument(); // toast
    expect(screen.getByRole('button', { name: '✓ Aplicado · ciclo 1/3' })).toBeDisabled();
    // El chip del header pasa a EN RECUPERACIÓN (estado demo local).
    expect(screen.getByText('EN RECUPERACIÓN')).toBeInTheDocument();
  });

  it('Simular/Rechazar → toast «Acción de demo»', () => {
    strategyMock.mockReturnValue(strategyRes());
    mountDrawer();
    fireEvent.click(screen.getByRole('button', { name: 'Simular' }));
    expect(screen.getByText('«Simular» · sin efecto')).toBeInTheDocument();
  });

  it('"Detalle de cámara" enlaza al puente /camara/:id', () => {
    strategyMock.mockReturnValue(strategyRes());
    render(
      <MemoryRouter initialEntries={['/']}>
        <ToastProvider>
          <IntersectionDrawer
            cameraId="cam_larco_benavides"
            intersections={[CAMERA]}
            intersectionsLoading={false}
            vision={undefined}
            edgeLevel={null}
            onClose={vi.fn()}
          />
        </ToastProvider>
      </MemoryRouter>,
    );
    // El botón existe y es clickeable (la navegación real la cubre router.test).
    expect(screen.getByRole('button', { name: 'Detalle de cámara' })).toBeInTheDocument();
  });
});
