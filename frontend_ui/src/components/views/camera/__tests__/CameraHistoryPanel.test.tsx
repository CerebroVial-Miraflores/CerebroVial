import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { CameraHistoryPanel } from '../CameraHistoryPanel';
import { usePredictionHistory } from '../../../../hooks/usePredictionHistory';
import type { PredictionHistoryResponse } from '../../../../types/predictionHistory';
import type { RestResource } from '../../../../hooks/types';

vi.mock('../../../../hooks/usePredictionHistory', () => ({
  usePredictionHistory: vi.fn(),
}));

const useHistoryMock = usePredictionHistory as unknown as ReturnType<typeof vi.fn>;

function resource(
  over: Partial<RestResource<PredictionHistoryResponse>>,
): RestResource<PredictionHistoryResponse> {
  return {
    data: null,
    loading: false,
    error: null,
    errorStatus: null,
    lastUpdated: null,
    refetch: vi.fn(),
    ...over,
  };
}

const payload: PredictionHistoryResponse = {
  camera_id: 'cam_a',
  history: [
    { timestamp: '14:00', total_vehicles: 30, congestion_level: 'Bajo', is_prediction: false },
    { timestamp: '14:05', total_vehicles: 48, congestion_level: 'Moderado', is_prediction: false },
    { timestamp: '14:10', total_vehicles: 62, congestion_level: 'Alto', is_prediction: false },
  ],
  prediction: {
    predicted_congestion_15min: 'Heavy',
    predicted_congestion_30min: 'Moderado',
    predicted_congestion_45min: 'Low',
    predicted_vehicles_15min: 70,
    predicted_vehicles_30min: 55,
    predicted_vehicles_45min: 40,
  },
};

beforeEach(() => {
  useHistoryMock.mockReset();
});

describe('CameraHistoryPanel', () => {
  const render4A = () =>
    render(
      <CameraHistoryPanel cameraId="cam_a" interval={5} onIntervalChange={vi.fn()} />,
    );

  it('con historial real + predicción dibuja el BigChart (serie + forecast)', () => {
    useHistoryMock.mockReturnValue(resource({ data: payload }));
    render4A();
    // El forecast punteado de BigChart se marca con su testid.
    expect(screen.getByTestId('bigchart-forecast')).toBeInTheDocument();
  });

  it('muestra los chips de congestión por horizonte +15/+30/+45 normalizados', () => {
    useHistoryMock.mockReturnValue(resource({ data: payload }));
    render4A();
    expect(screen.getByText('+15 min')).toBeInTheDocument();
    expect(screen.getByText('+30 min')).toBeInTheDocument();
    expect(screen.getByText('+45 min')).toBeInTheDocument();
    // 'Heavy' → ALTO; 'Low' → BAJO (normalización ES del backend mezclado).
    expect(screen.getByText('ALTO')).toBeInTheDocument();
    expect(screen.getByText('BAJO')).toBeInTheDocument();
  });

  it('loading inicial sin data muestra "Cargando datos…"', () => {
    useHistoryMock.mockReturnValue(resource({ loading: true }));
    render4A();
    expect(screen.getByText(/cargando datos/i)).toBeInTheDocument();
  });

  it('historial insuficiente (<2 puntos reales) es honesto', () => {
    useHistoryMock.mockReturnValue(
      resource({
        data: {
          camera_id: 'cam_a',
          history: [
            { timestamp: '14:00', total_vehicles: 30, congestion_level: 'Bajo', is_prediction: false },
          ],
        },
      }),
    );
    render4A();
    expect(screen.getByText(/no hay suficiente historial/i)).toBeInTheDocument();
  });

  it('error expone el mensaje y un botón de reintento', () => {
    useHistoryMock.mockReturnValue(resource({ error: 'cámara desconocida' }));
    render4A();
    expect(screen.getByText('cámara desconocida')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reintentar/i })).toBeInTheDocument();
  });
});
