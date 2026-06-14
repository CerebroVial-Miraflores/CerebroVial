import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { CameraDetailV2 } from '../CameraDetailV2';
import { useCameras } from '../../../../hooks/useCameras';
import { usePredictionHistory } from '../../../../hooks/usePredictionHistory';
import type { CameraSummary } from '../../../../types/cameras';
import type { RestResource } from '../../../../hooks/types';

vi.mock('../../../../hooks/useCameras', () => ({ useCameras: vi.fn() }));
vi.mock('../../../../hooks/usePredictionHistory', () => ({ usePredictionHistory: vi.fn() }));
vi.mock('../../../HlsPlayer', () => ({
  HlsPlayer: ({ src }: { src: string }) => <div data-testid="hls-player">{src}</div>,
}));
// El panel en vivo monta AnnotatedCameraStream, que abre un fetch streaming a
// /video/{id}?type=processed y lee response.body. El fetch stub de este test no trae
// body → se mockea el componente (igual que HlsPlayer) para no disparar reconexión.
vi.mock('../AnnotatedCameraStream', () => ({
  AnnotatedCameraStream: ({ cameraId }: { cameraId: string }) => (
    <div data-testid="annotated-stream">{cameraId}</div>
  ),
}));
vi.mock('../../../../services/predictionService', () => ({
  predictionService: { predictTraffic: vi.fn(() => new Promise(() => {})) },
}));

const useCamerasMock = useCameras as unknown as ReturnType<typeof vi.fn>;
const useHistoryMock = usePredictionHistory as unknown as ReturnType<typeof vi.fn>;

const EDGE = 'http://localhost:8000';

const cameras: CameraSummary[] = [
  { id: 'cam_a', name: 'Cámara A', stream_url: 'https://x/a.m3u8' },
  { id: 'cam_b', name: 'Cámara B', stream_url: 'https://x/b.m3u8' },
];

function camerasResource(over: Partial<RestResource<CameraSummary[]>>): RestResource<CameraSummary[]> {
  return {
    data: cameras,
    loading: false,
    error: null,
    errorStatus: null,
    lastUpdated: 1,
    refetch: vi.fn(),
    ...over,
  };
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.clearAllMocks();
  fetchMock = vi.fn(() => Promise.resolve({ ok: true, status: 200 } as Response));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  useHistoryMock.mockReturnValue({
    data: null,
    loading: true,
    error: null,
    errorStatus: null,
    lastUpdated: null,
    refetch: vi.fn(),
  });
});

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/camara/:id" element={<CameraDetailV2 />} />
        <Route path="/" element={<div data-testid="comando" />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('CameraDetailV2', () => {
  it('loading sin data muestra "Cargando cámara…"', () => {
    useCamerasMock.mockReturnValue(camerasResource({ data: null, loading: true }));
    renderAt('/camara/cam_a');
    expect(screen.getByText(/cargando cámara/i)).toBeInTheDocument();
  });

  it('error de la lista expone mensaje y reintento', () => {
    useCamerasMock.mockReturnValue(camerasResource({ data: null, error: 'core caído' }));
    renderAt('/camara/cam_a');
    expect(screen.getByText('core caído')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reintentar/i })).toBeInTheDocument();
  });

  it('id desconocido es honesto y no da de alta nada en el edge', () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_inexistente');
    expect(screen.getByText(/no existe en el inventario/i)).toBeInTheDocument();
    const posted = fetchMock.mock.calls.some((c) => (c[1] as RequestInit)?.method === 'POST');
    expect(posted).toBe(false);
  });

  it('da de alta la cámara en el edge (POST con el stream de Claro como source)', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        `${EDGE}/cameras/cam_a`,
        expect.objectContaining({ method: 'POST' }),
      );
    });
    const postCall = fetchMock.mock.calls.find((c) => (c[1] as RequestInit)?.method === 'POST');
    const body = JSON.parse((postCall![1] as RequestInit).body as string);
    expect(body).toEqual({ source: 'https://x/a.m3u8', source_type: 'hls', zones: {} });
  });

  it('NO da de baja en unmount (la baja la posee el edge)', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    const { unmount } = renderAt('/camara/cam_a');
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    unmount();
    const deleted = fetchMock.mock.calls.some((c) => (c[1] as RequestInit)?.method === 'DELETE');
    expect(deleted).toBe(false);
  });

  it('cámara sin stream: estado honesto y sin POST', () => {
    useCamerasMock.mockReturnValue(
      camerasResource({ data: [{ id: 'cam_a', name: 'Cámara A', stream_url: null }] }),
    );
    renderAt('/camara/cam_a');
    expect(screen.getByText(/no tiene un stream configurado/i)).toBeInTheDocument();
    const posted = fetchMock.mock.calls.some((c) => (c[1] as RequestInit)?.method === 'POST');
    expect(posted).toBe(false);
  });

  it('señaliza los caveats real-con-caveat de las métricas de visión', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');
    // Velocidad: caveat visible "aprox" + title "sin calibrar".
    expect(screen.getByText('aprox')).toBeInTheDocument();
    expect(screen.getByTitle(/sin calibrar/i)).toBeInTheDocument();
    // Pie de la card de estado: presencia extrapolada (no aforo calibrado).
    expect(screen.getByText(/presencia extrapolada/i)).toBeInTheDocument();
    // Flush del alta on-demand (setState async del POST) dentro de act.
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  });

  it('navegar por el carril cambia la cámara activa y re-da de alta', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    expect(screen.getByRole('heading', { name: 'Cámara A' })).toBeInTheDocument();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(`${EDGE}/cameras/cam_a`, expect.objectContaining({ method: 'POST' })),
    );

    // El carril muestra la OTRA cámara (cam_b); click → navega y re-resuelve.
    fireEvent.click(screen.getByTitle('Ver Cámara B'));

    expect(screen.getByRole('heading', { name: 'Cámara B' })).toBeInTheDocument();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(`${EDGE}/cameras/cam_b`, expect.objectContaining({ method: 'POST' })),
    );
  });

  it('toggle Histórico monta la historia de predicción', () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    expect(screen.getByText('Métricas en vivo')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Histórico' }));

    expect(screen.getByText('Historial y predicción')).toBeInTheDocument();
    expect(screen.queryByText('Métricas en vivo')).not.toBeInTheDocument();
  });
});
