/**
 * CameraDetailRoute — puente temporal /camara/:id (FASE 3, muere en F4).
 *
 * CameraDetailView se mockea per-file (acá se valida la resolución id →
 * {name, stream_url} vía useIntersections y los estados honestos; el detalle
 * tiene sus propios tests).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';

import { CameraDetailRoute } from '../CameraDetailRoute';
import { useIntersections } from '../../../hooks/useIntersections';

vi.mock('../../../hooks/useIntersections', () => ({ useIntersections: vi.fn() }));
vi.mock('../CameraDetailView', () => ({
  CameraDetailView: (props: {
    cameraId: string;
    cameraName: string;
    streamUrl: string | null;
    onBack: () => void;
  }) => (
    <div data-testid="camera-detail">
      <span>{props.cameraId}</span>
      <span>{props.cameraName}</span>
      <span>{props.streamUrl ?? 'sin-stream'}</span>
      <button onClick={props.onBack}>volver</button>
    </div>
  ),
}));

const intersectionsMock = vi.mocked(useIntersections);

const CAMERA = {
  id: 'cam_larco_benavides',
  name: 'Larco × Benavides',
  speed: 0,
  flow: 0,
  status: 'critical',
  lat: -12.13,
  lng: -77.02,
  stream_url: 'https://claro/hls/larco.m3u8',
};

function res(data: typeof CAMERA[] | null, extra: Record<string, unknown> = {}) {
  return {
    data,
    loading: false,
    error: null as string | null,
    errorStatus: null as number | null,
    lastUpdated: data !== null ? 1 : null,
    refetch: vi.fn(async () => {}),
    ...extra,
  };
}

function mountAt(id: string) {
  const router = createMemoryRouter(
    [
      { path: '/', element: <div data-testid="comando" /> },
      { path: '/camara/:id', element: <CameraDetailRoute /> },
    ],
    { initialEntries: [`/camara/${id}`] },
  );
  render(<RouterProvider router={router} />);
  return router;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('CameraDetailRoute (puente /camara/:id)', () => {
  it('id válido → monta CameraDetailView con name/stream resueltos del inventario', () => {
    intersectionsMock.mockReturnValue(res([CAMERA]));
    mountAt('cam_larco_benavides');
    expect(screen.getByTestId('camera-detail')).toBeInTheDocument();
    expect(screen.getByText('Larco × Benavides')).toBeInTheDocument();
    expect(screen.getByText('https://claro/hls/larco.m3u8')).toBeInTheDocument();
  });

  it('onBack vuelve al comando ("/")', () => {
    intersectionsMock.mockReturnValue(res([CAMERA]));
    const router = mountAt('cam_larco_benavides');
    fireEvent.click(screen.getByRole('button', { name: 'volver' }));
    expect(router.state.location.pathname).toBe('/');
  });

  it('id desconocido → error honesto con vuelta al comando (sin montar el detalle)', () => {
    intersectionsMock.mockReturnValue(res([CAMERA]));
    const router = mountAt('cam_fantasma');
    expect(screen.queryByTestId('camera-detail')).not.toBeInTheDocument();
    expect(screen.getByText(/no existe en el inventario/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Volver al comando' }));
    expect(router.state.location.pathname).toBe('/');
  });

  it('loading → estado de carga; error → mensaje + Reintentar', () => {
    intersectionsMock.mockReturnValue(res(null, { loading: true }));
    mountAt('cam_larco_benavides');
    expect(screen.getByRole('status')).toHaveTextContent('Cargando cámara…');

    const refetch = vi.fn(async () => {});
    intersectionsMock.mockReturnValue(res(null, { error: 'core caído', refetch }));
    mountAt('cam_larco_benavides');
    expect(screen.getByText('core caído')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reintentar' }));
    expect(refetch).toHaveBeenCalledTimes(1);
  });
});
