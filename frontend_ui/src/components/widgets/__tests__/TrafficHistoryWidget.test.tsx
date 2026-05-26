import { render, screen, waitFor } from '@testing-library/react';
import { TrafficHistoryWidget } from '../TrafficHistoryWidget';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock recharts as it relies on DOM APIs not fully supported in jsdom or for simplicity
vi.mock('recharts', async (importOriginal) => {
    const original = await importOriginal<typeof import('recharts')>();
    return {
        ...original,
        ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container" style={{ width: 800, height: 300 }}>{children}</div>,
        // We can just render a div for the chart to verify it mounts
        ComposedChart: ({ children }: { children: React.ReactNode }) => <svg data-testid="composed-chart">{children}</svg>,
        Area: () => <div data-testid="area-chart" />,
        Line: () => <div data-testid="line-chart" />,
        XAxis: () => <div data-testid="x-axis" />,
        YAxis: () => <div data-testid="y-axis" />,
        CartesianGrid: () => <div data-testid="cartesian-grid" />,
        Tooltip: () => <div data-testid="tooltip" />,
        Legend: () => <div data-testid="legend" />,
        ReferenceLine: () => <div data-testid="reference-line" />,
    };
});

describe('TrafficHistoryWidget', () => {
    const mockCameraId = 'CAM_001';

    beforeEach(() => {
        vi.clearAllMocks();
        // Mock fetch
        globalThis.fetch = vi.fn();
    });

    it('renders loading state initially', async () => {
        (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
            ok: true,
            json: async () => ({ history: [] }),
        });
        
        render(<TrafficHistoryWidget cameraId={mockCameraId} />);
        
        // Wait for the component to finish loading to avoid 'act' warnings
        await waitFor(() => {
            expect(screen.queryByText('Cargando datos...')).not.toBeInTheDocument();
        });
    });

    it('fetches data on mount and renders chart', async () => {
        const mockData = {
            history: [
                { timestamp: '2023-10-27T10:00:00', total_vehicles: 50, occupancy_rate: 0.1 },
                { timestamp: '2023-10-27T10:01:00', total_vehicles: 55, occupancy_rate: 0.2 },
            ],
            prediction: {
                predicted_congestion_15min: 'Low',
                predicted_vehicles_15min: 60,
                predicted_congestion_30min: 'Medium',
                predicted_vehicles_30min: 70,
                predicted_congestion_45min: 'High',
                predicted_vehicles_45min: 80,
            }
        };

        (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
            ok: true,
            json: async () => mockData,
        });

        render(<TrafficHistoryWidget cameraId={mockCameraId} />);

        await waitFor(() => {
            expect(globalThis.fetch).toHaveBeenCalledWith(`http://localhost:8001/predictions/history/${mockCameraId}?interval=5`);
        });

        // Check if chart renders
        await waitFor(() => {
            expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
        });
    });

    it('handles fetch error gracefully', async () => {
        const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => { });
        (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('API Error'));

        render(<TrafficHistoryWidget cameraId={mockCameraId} />);

        await waitFor(() => {
            expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch history', expect.any(Error));
        });
        consoleSpy.mockRestore();
    });
});
