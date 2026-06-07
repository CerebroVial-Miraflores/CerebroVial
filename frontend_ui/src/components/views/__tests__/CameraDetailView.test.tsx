
import { StrictMode } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CameraDetailView } from '../CameraDetailView';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { predictionService } from '../../../services/predictionService';

// Mock Lucide icons
vi.mock('lucide-react', () => ({
    X: () => <div data-testid="icon-x" />,
    Users: () => <div data-testid="icon-users" />,
    Activity: () => <div data-testid="icon-activity" />,
    Car: () => <div data-testid="icon-car" />,
    AlertTriangle: () => <div data-testid="icon-alert-triangle" />,
    Zap: () => <div data-testid="icon-zap" />,
    ArrowLeft: () => <div data-testid="icon-arrow-left" />,
    Clock: () => <div data-testid="icon-clock" />
}));

// Mock TrafficHistoryWidget
vi.mock('../../widgets/TrafficHistoryWidget', () => ({
    TrafficHistoryWidget: () => <div data-testid="traffic-history-widget">History Chart</div>
}));

// Mock HlsPlayer (lo usa el carril de cámaras). No corre hls en jsdom.
vi.mock('../../HlsPlayer', () => ({
    HlsPlayer: ({ src }: { src: string }) => <div data-testid="hls-player">{src}</div>,
}));

// Mock de IntersectionObserver (jsdom no lo trae): los tiles del carril lo usan para el
// lazy. Acá no simulamos visibilidad — basta con que el constructor exista.
class MockIntersectionObserver {
    observe = vi.fn();
    disconnect = vi.fn();
    unobserve = vi.fn();
    takeRecords = vi.fn();
}
Object.defineProperty(globalThis, 'IntersectionObserver', {
    writable: true,
    value: MockIntersectionObserver,
});

// Mock Prediction Service
vi.mock('../../../services/predictionService', () => ({
    predictionService: {
        predictTraffic: vi.fn()
    }
}));

// Mock EventSource construction
class MockEventSource {
    url: string;
    constructor(url: string) {
        this.url = url;
    }
    addEventListener = vi.fn();
    removeEventListener = vi.fn();
    close = vi.fn();
}

Object.defineProperty(globalThis, 'EventSource', {
    writable: true,
    value: MockEventSource,
});

describe('CameraDetailView', () => {
    const mockOnBack = vi.fn();
    const mockCameraId = 'cam_larco_benavides';
    const mockCameraName = 'Larco Benavides';
    const mockStreamUrl = 'https://video.claro.com.pe/live/cam.m3u8';
    const EDGE = 'http://localhost:8000';

    let fetchMock: ReturnType<typeof vi.fn>;

    const renderView = (streamUrl: string | null = mockStreamUrl) =>
        render(
            <CameraDetailView
                cameraId={mockCameraId}
                cameraName={mockCameraName}
                streamUrl={streamUrl}
                onBack={mockOnBack}
            />,
        );

    beforeEach(() => {
        vi.clearAllMocks();
        mockOnBack.mockClear();
        // Alta/baja en el edge: por defecto OK. Cada test puede sobreescribirlo.
        fetchMock = vi.fn(() => Promise.resolve({ ok: true, status: 200 } as Response));
        globalThis.fetch = fetchMock as unknown as typeof fetch;
    });

    it('renders correctly with default live view', () => {
        renderView();

        // Check header title (el nombre viene por prop desde /api/intersections, no hardcodeado)
        expect(screen.getByText(mockCameraName)).toBeInTheDocument();

        // Check default buttons
        expect(screen.getByText('Analítica en tiempo Real')).toBeInTheDocument();
        expect(screen.getByText('Histórico')).toBeInTheDocument();

        // Check "Live" indicator
        expect(screen.getByText('En vivo')).toBeInTheDocument();

        // Check metrics cards
        expect(screen.getByText('Métricas en Tiempo Real')).toBeInTheDocument();
    });

    it('calls onBack when back button is clicked', () => {
        renderView();

        const backButtons = screen.getAllByTestId('icon-arrow-left');
        const backBtn = backButtons[0].parentElement;
        fireEvent.click(backBtn!);

        expect(mockOnBack).toHaveBeenCalledTimes(1);
    });

    it('switches to history view when tab is clicked', async () => {
        renderView();

        const historyTab = screen.getByText('Histórico');
        fireEvent.click(historyTab);

        // Should render TrafficHistoryWidget
        await waitFor(() => {
            expect(screen.getByTestId('traffic-history-widget')).toBeInTheDocument();
        });

        // Should NOT render Live Stream elements (checked via stream type toggle absence)
        expect(screen.queryByText('PROCESADO')).not.toBeInTheDocument();
    });

    it('does not call prediction service initially (vehicles = 0)', () => {
        renderView();
        expect(predictionService.predictTraffic).not.toHaveBeenCalled();
    });

    // ---- C1/F1-F2: orquestación on-demand del YOLO en el edge --------------

    it('POSTs the camera to the edge on mount with the Claro stream_url', async () => {
        renderView();

        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledWith(
                `${EDGE}/cameras/${mockCameraId}`,
                expect.objectContaining({ method: 'POST' }),
            );
        });

        // El body lleva la URL de Claro como `source` y source_type "hls".
        const postCall = fetchMock.mock.calls.find(
            (c) => (c[1] as RequestInit)?.method === 'POST',
        );
        const body = JSON.parse((postCall![1] as RequestInit).body as string);
        expect(body).toEqual({ source: mockStreamUrl, source_type: 'hls', zones: {} });
    });

    it('does NOT DELETE on unmount (teardown is owned by the edge watchdog)', async () => {
        const { unmount } = renderView();
        await waitFor(() => expect(fetchMock).toHaveBeenCalled());

        unmount();

        // El front nunca da de baja: la libera el edge (single-slot + watchdog).
        const deleted = fetchMock.mock.calls.some(
            (c) => (c[1] as RequestInit)?.method === 'DELETE',
        );
        expect(deleted).toBe(false);
    });

    it('survives the StrictMode mount→unmount→remount cycle without DELETE', async () => {
        // Bajo <StrictMode> React monta, desmonta y remonta el efecto. Con el código
        // viejo, el cleanup de la 1ª pasada disparaba un DELETE que removía la cámara
        // recién dada de alta → "Cargando detección…" eterno. Ahora no hay DELETE.
        render(
            <StrictMode>
                <CameraDetailView
                    cameraId={mockCameraId}
                    cameraName={mockCameraName}
                    streamUrl={mockStreamUrl}
                    onBack={mockOnBack}
                />
            </StrictMode>,
        );

        await waitFor(() => expect(fetchMock).toHaveBeenCalled());

        // Se dio de alta (POST) pero NUNCA se mandó DELETE pese al doble-invoke.
        const methods = fetchMock.mock.calls.map((c) => (c[1] as RequestInit)?.method);
        expect(methods).toContain('POST');
        expect(methods).not.toContain('DELETE');
    });

    it('shows a loading overlay until the edge responds', () => {
        // fetch nunca resuelve → queda en "Cargando detección…".
        fetchMock.mockReturnValue(new Promise(() => {}));
        renderView();
        expect(screen.getByText(/Cargando detección/)).toBeInTheDocument();
    });

    it('shows a clear error when the edge fails (Claro down)', async () => {
        fetchMock.mockResolvedValue({ ok: false, status: 404 } as Response);
        renderView();

        await waitFor(() => {
            expect(screen.getByText(/no se pudo iniciar la detección/i)).toBeInTheDocument();
        });
    });

    it('shows an error and does not POST when there is no stream_url', () => {
        renderView(null);

        expect(screen.getByText(/no tiene un stream configurado/i)).toBeInTheDocument();
        // Sin stream no se intenta el alta en el edge.
        const posted = fetchMock.mock.calls.some(
            (c) => (c[1] as RequestInit)?.method === 'POST',
        );
        expect(posted).toBe(false);
    });

    // ---- Carril de otras cámaras + reemplazo del detalle ------------------

    // fetch que sirve la lista de intersecciones (para el carril) y OK para el resto.
    const fetchWithIntersections = (cameras: Array<{ id: string; name: string; stream_url: string | null }>) =>
        vi.fn((url: RequestInfo | URL) => {
            if (typeof url === 'string' && url.includes('/api/intersections')) {
                return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(cameras) } as Response);
            }
            return Promise.resolve({ ok: true, status: 200 } as Response);
        });

    it('el carril muestra las OTRAS cámaras (excluye la activa)', async () => {
        fetchMock = fetchWithIntersections([
            { id: mockCameraId, name: mockCameraName, stream_url: mockStreamUrl },
            { id: 'cam_otra', name: 'Otra Cámara', stream_url: 'http://x/otra.m3u8' },
        ]);
        globalThis.fetch = fetchMock as unknown as typeof fetch;

        renderView();

        // La activa NO aparece como tile; la otra sí.
        await waitFor(() => {
            expect(screen.getByText('Otra Cámara')).toBeInTheDocument();
        });
        const tiles = screen.getAllByTestId('strip-tile');
        expect(tiles).toHaveLength(1);
    });

    it('al clickear otra cámara en el carril, el detalle se reemplaza (nueva alta en el edge)', async () => {
        fetchMock = fetchWithIntersections([
            { id: mockCameraId, name: mockCameraName, stream_url: mockStreamUrl },
            { id: 'cam_otra', name: 'Otra Cámara', stream_url: 'http://x/otra.m3u8' },
        ]);
        globalThis.fetch = fetchMock as unknown as typeof fetch;

        renderView();

        // Alta inicial sobre la cámara activa.
        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledWith(
                `${EDGE}/cameras/${mockCameraId}`,
                expect.objectContaining({ method: 'POST' }),
            );
        });

        // Click en el tile de la otra cámara.
        await waitFor(() => expect(screen.getByText('Otra Cámara')).toBeInTheDocument());
        fireEvent.click(screen.getByTestId('strip-tile'));

        // El panel se re-monta (key) → nueva alta on-demand para la cámara elegida.
        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledWith(
                `${EDGE}/cameras/cam_otra`,
                expect.objectContaining({ method: 'POST' }),
            );
        });

        // Y el header pasa a mostrar el nombre de la cámara elegida.
        expect(screen.getByRole('heading', { name: 'Otra Cámara' })).toBeInTheDocument();
    });
});
