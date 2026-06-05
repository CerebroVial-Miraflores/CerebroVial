
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

    it('DELETEs the camera on unmount', async () => {
        const { unmount } = renderView();
        await waitFor(() => expect(fetchMock).toHaveBeenCalled());

        unmount();

        expect(fetchMock).toHaveBeenCalledWith(
            `${EDGE}/cameras/${mockCameraId}`,
            expect.objectContaining({ method: 'DELETE' }),
        );
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
});
