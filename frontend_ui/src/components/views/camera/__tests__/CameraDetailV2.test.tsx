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
// useVisionStream abre un EventSource real al edge en ALTA; se mockea para que el
// test del toggle no dependa del SSE (no es lo que se está ejercitando acá).
vi.mock('../../../../hooks/useVisionStream', () => ({
  useVisionStream: () => ({ data: null, lastUpdated: null, connection: 'idle' }),
}));
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

/** fetch stub del edge: GET inference-status devuelve `inferring`; POST/DELETE 200.
 *  `inferring` y `postStatus` se parametrizan por test. */
function makeFetch(opts: { inferring?: string[]; postStatus?: number } = {}) {
  const { inferring = [], postStatus = 200 } = opts;
  return vi.fn((url: unknown, init?: RequestInit) => {
    const u = String(url);
    if (u.includes('/cameras/inference-status')) {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({ inferring, count: inferring.length, cap: null, capacity_used: null }),
      } as unknown as Response);
    }
    // POST (alta) / DELETE (baja) a /cameras/{id}
    const method = init?.method ?? 'GET';
    if (method === 'POST') {
      return Promise.resolve({ ok: postStatus < 400, status: postStatus } as Response);
    }
    return Promise.resolve({ ok: true, status: 200 } as Response);
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  fetchMock = makeFetch();
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

const postCalls = () =>
  fetchMock.mock.calls.filter((c) => (c[1] as RequestInit)?.method === 'POST');
const deleteCalls = () =>
  fetchMock.mock.calls.filter((c) => (c[1] as RequestInit)?.method === 'DELETE');

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

  it('id desconocido es honesto y no consulta ni da de alta nada en el edge', () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_inexistente');
    expect(screen.getByText(/no existe en el inventario/i)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('al montar SOLO consulta inference-status (GET) — no postea ni borra', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(`${EDGE}/cameras/inference-status`, {}),
    );
    expect(postCalls()).toHaveLength(0);
    expect(deleteCalls()).toHaveLength(0);
  });

  it('estado inicial refleja el GET: cámara fresca → BAJA (HlsPlayer directo)', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    expect(await screen.findByTestId('hls-player')).toHaveTextContent('https://x/a.m3u8');
    expect(screen.queryByTestId('annotated-stream')).not.toBeInTheDocument();
  });

  it('estado inicial refleja el GET: cámara ya infiriendo → ALTA (stream anotado)', async () => {
    fetchMock = makeFetch({ inferring: ['cam_a'] });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    expect(await screen.findByTestId('annotated-stream')).toHaveTextContent('cam_a');
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
    expect(postCalls()).toHaveLength(0); // no se re-postea lo que ya infiere
  });

  it('check (Detección) dispara POST y cambia a stream anotado', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    await screen.findByTestId('hls-player');
    fireEvent.click(screen.getByRole('button', { name: 'Detección' }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        `${EDGE}/cameras/cam_a`,
        expect.objectContaining({ method: 'POST' }),
      ),
    );
    const body = JSON.parse((postCalls()[0][1] as RequestInit).body as string);
    expect(body).toEqual({ source: 'https://x/a.m3u8', source_type: 'hls', zones: {} });
    expect(await screen.findByTestId('annotated-stream')).toBeInTheDocument();
  });

  it('uncheck (Directo) dispara DELETE y vuelve al HlsPlayer directo', async () => {
    fetchMock = makeFetch({ inferring: ['cam_a'] });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    await screen.findByTestId('annotated-stream');
    fireEvent.click(screen.getByRole('button', { name: 'Directo' }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        `${EDGE}/cameras/cam_a`,
        expect.objectContaining({ method: 'DELETE' }),
      ),
    );
    expect(await screen.findByTestId('hls-player')).toBeInTheDocument();
  });

  it('409 al activar revierte a BAJA con aviso y sin tapar el video', async () => {
    fetchMock = makeFetch({ postStatus: 409 });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    await screen.findByTestId('hls-player');
    fireEvent.click(screen.getByRole('button', { name: 'Detección' }));

    expect(await screen.findByText(/a capacidad/i)).toBeInTheDocument();
    // Sigue en BAJA: HlsPlayer visible, sin stream anotado.
    expect(screen.getByTestId('hls-player')).toBeInTheDocument();
    expect(screen.queryByTestId('annotated-stream')).not.toBeInTheDocument();
  });

  it('NO da de baja en unmount (la baja solo la dispara el uncheck del toggle)', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    const { unmount } = renderAt('/camara/cam_a');
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    unmount();
    expect(deleteCalls()).toHaveLength(0);
  });

  it('cámara sin stream: estado honesto y sin tocar el edge', () => {
    useCamerasMock.mockReturnValue(
      camerasResource({ data: [{ id: 'cam_a', name: 'Cámara A', stream_url: null }] }),
    );
    renderAt('/camara/cam_a');
    expect(screen.getByText(/no tiene un stream configurado/i)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('señaliza los caveats real-con-caveat de las métricas de visión', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');
    // Velocidad: caveat visible "aprox" + title "sin calibrar".
    expect(screen.getByText('aprox')).toBeInTheDocument();
    expect(screen.getByTitle(/sin calibrar/i)).toBeInTheDocument();
    // Pie de la card de estado: presencia extrapolada (no aforo calibrado).
    expect(screen.getByText(/presencia extrapolada/i)).toBeInTheDocument();
    // Flush del GET de estado (setState async) dentro de act.
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  });

  it('navegar por el carril cambia la cámara y re-consulta el estado (sin auto-POST)', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    expect(screen.getByRole('heading', { name: 'Cámara A' })).toBeInTheDocument();
    await screen.findByTestId('hls-player');

    // El carril muestra la OTRA cámara (cam_b); click → navega y re-resuelve.
    fireEvent.click(screen.getByTitle('Ver Cámara B'));

    expect(screen.getByRole('heading', { name: 'Cámara B' })).toBeInTheDocument();
    expect(await screen.findByTestId('hls-player')).toHaveTextContent('https://x/b.m3u8');
    expect(postCalls()).toHaveLength(0); // navegar no da de alta solo
  });

  it('toggle Histórico monta la historia de predicción', async () => {
    useCamerasMock.mockReturnValue(camerasResource({}));
    renderAt('/camara/cam_a');

    expect(screen.getByText('Métricas en vivo')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Histórico' }));

    expect(screen.getByText('Historial y predicción')).toBeInTheDocument();
    expect(screen.queryByText('Métricas en vivo')).not.toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  });
});
