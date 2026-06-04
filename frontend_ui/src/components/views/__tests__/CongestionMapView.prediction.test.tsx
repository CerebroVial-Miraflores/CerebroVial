/**
 * CongestionMapView (Fase 4, Gate 3a) — tercer modo + segunda capa de predicción.
 *
 * Archivo SEPARADO del smoke vivo (`*.test.tsx`) y del histórico (`*.historic.test.tsx`)
 * a propósito: predicción no debe afectar a esos modos. Calca su patrón de mock de
 * react-leaflet, pero captura DOS capas `<GeoJSON>` (la base observada y la de
 * predicción superpuesta). Como React no pasa `key` como prop, se distinguen por la
 * presencia de `onEachFeature`: solo la base lo lleva (CA-22.5); la predicción no
 * (es `interactive: false`).
 *
 * Afirma: en predicción monta la capa de predicción pintando el índice de prueba con
 * `predictionStyle`; las dos capas COEXISTEN (un wake SSE remonta la base sin remontar
 * la predicción — el punto crítico de 3a); y los errores 503/409/422 del endpoint
 * muestran su mensaje sin romper la vista ni la capa base.
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor, act, fireEvent } from '@testing-library/react';
import { CongestionMapView } from '../CongestionMapView';
import { congestionService } from '../../../services/congestionService';
import { openCongestionStream } from '../../../services/congestionSseClient';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
  CongestionPredictionResponse,
} from '../../../types/congestion';

// Dos slots: la base (con onEachFeature) y la predicción (sin él), cada una con su
// contador de montajes para detectar remontes independientes por `key`.
const captured = vi.hoisted(() => ({
  base: null as null | Record<string, unknown>,
  baseMounts: 0,
  prediction: null as null | Record<string, unknown>,
  predictionMounts: 0,
}));

const sse = vi.hoisted(() => ({
  onWake: null as null | (() => void),
  abort: vi.fn(),
}));

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
  GeoJSON: (props: Record<string, unknown>) => {
    const isBase = typeof props.onEachFeature === 'function';
    if (isBase) captured.base = props;
    else captured.prediction = props;
    React.useEffect(() => {
      if (isBase) captured.baseMounts += 1;
      else captured.predictionMounts += 1;
    }, [isBase]);
    return <div data-testid={isBase ? 'geojson-base' : 'geojson-prediction'} />;
  },
}));

vi.mock('../../../services/congestionService', () => ({
  congestionService: {
    getGeometry: vi.fn(),
    getState: vi.fn(),
    getPrediction: vi.fn(),
  },
}));

vi.mock('../../../services/congestionSseClient', () => ({
  openCongestionStream: vi.fn((opts: { onWake: () => void }) => {
    sse.onWake = opts.onWake;
    return { abort: sse.abort };
  }),
}));

const getGeometryMock = congestionService.getGeometry as unknown as ReturnType<typeof vi.fn>;
const getStateMock = congestionService.getState as unknown as ReturnType<typeof vi.fn>;
const getPredictionMock = congestionService.getPrediction as unknown as ReturnType<typeof vi.fn>;
const openStreamMock = openCongestionStream as unknown as ReturnType<typeof vi.fn>;

const COUNT = 375;
const HORIZON = 30;
// El componente pinta el índice de prueba fijo PREDICTION_TEST_INDEX = 15.

function buildGeometry(): GeometryFeatureCollection {
  return {
    type: 'FeatureCollection',
    count: COUNT,
    features: Array.from({ length: COUNT }, (_, i) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'LineString' as const,
        coordinates: [
          [-77.0335, -12.118],
          [-77.0334, -12.1179],
        ] as [number, number][],
      },
      properties: {
        edge_id: `edge-${i}`,
        source_node: `src-${i}`,
        target_node: `dst-${i}`,
        distance_m: 100,
        lanes: 1,
      },
    })),
  };
}

function buildState(): CongestionStateResponse {
  return {
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      congestion_level: i % 6,
      snapshot_timestamp: '2025-01-06T23:59:00',
    })),
  };
}

/** Predicción a 30 pasos: edge-i → levels[k] = (i+k) % 5 (escala 0-4). En el índice
 *  de prueba 15, edge-0 → (0+15)%5 = 0; edge-2 → (2+15)%5 = 2 (niveles conocidos). */
function buildPrediction(): CongestionPredictionResponse {
  return {
    base_timestep: 600,
    horizon: HORIZON,
    source: 'seed051 (day_idx=9)',
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      levels: Array.from({ length: HORIZON }, (_, k) => (i + k) % 5),
    })),
  };
}

type CapturedData = { data: { features: { properties: { congestion_level: number } }[] } };
type StyleFn = (f: unknown) => {
  color: string;
  weight: number;
  opacity?: number;
  interactive?: boolean;
};

async function renderAndLoad() {
  render(<CongestionMapView />);
  await waitFor(() => expect(screen.getByTestId('geojson-base')).toBeInTheDocument());
}

function clickPrediction() {
  fireEvent.click(screen.getByRole('button', { name: /predicción/i }));
}

describe('CongestionMapView — modo predicción (Fase 4 Gate 3a)', () => {
  beforeEach(() => {
    getGeometryMock.mockReset();
    getStateMock.mockReset();
    getPredictionMock.mockReset();
    openStreamMock.mockClear();
    sse.onWake = null;
    sse.abort.mockClear();
    captured.base = null;
    captured.baseMounts = 0;
    captured.prediction = null;
    captured.predictionMounts = 0;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('al entrar en predicción monta la capa de predicción pintando el índice de prueba', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });

    // La capa de predicción aparece y cruza el índice 15: edge-0 → nivel 0.
    await waitFor(() =>
      expect(screen.getByTestId('geojson-prediction')).toBeInTheDocument(),
    );
    const pred = captured.prediction as CapturedData;
    expect(pred.data.features[0].properties.congestion_level).toBe(0);
    // edge-2 en índice 15 → (2+15)%5 = 2.
    expect(pred.data.features[2].properties.congestion_level).toBe(2);

    // Estilo de la capa de predicción: paleta fría (predictionStyle) + no interactiva.
    const style = (captured.prediction as { style: StyleFn }).style;
    const s2 = style({ properties: { congestion_level: 2 } });
    expect(s2.color).toBe('#818CF8'); // nivel 2 de la escala de predicción
    expect(s2.opacity).toBe(0.55);
    expect(s2.interactive).toBe(false);
    // Nivel fuera de 0-4 → neutro transparente (no pinta).
    expect(style({ properties: { congestion_level: null } }).opacity).toBe(0);
  });

  it('las dos capas coexisten: un wake SSE remonta la BASE sin remontar la predicción', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    await waitFor(() =>
      expect(screen.getByTestId('geojson-prediction')).toBeInTheDocument(),
    );

    // Conteos tras estabilizar la entrada en predicción.
    const baseBefore = captured.baseMounts;
    const predBefore = captured.predictionMounts;

    // Un wake del feed re-lee estado y recolorea la base (bump de renderSeq → remonta
    // la base). La capa de predicción NO debe remontarse: su `key` es independiente.
    await act(async () => {
      sse.onWake?.();
    });
    await waitFor(() => expect(captured.baseMounts).toBeGreaterThan(baseBefore));
    expect(captured.predictionMounts).toBe(predBefore);
  });

  it.each([
    [503, /no está disponible/i],
    [409, /suficiente historia/i],
    [422, /fuera de rango/i],
  ])('error %i del endpoint: muestra su mensaje sin romper la vista ni la base', async (status, re) => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockRejectedValue({ response: { status } });

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });

    await waitFor(() => expect(screen.getByText(re)).toBeInTheDocument());
    // La capa base sigue presente (no se rompió la vista); la predicción no pintó.
    expect(screen.getByTestId('geojson-base')).toBeInTheDocument();
    expect(screen.queryByTestId('geojson-prediction')).not.toBeInTheDocument();
  });
});
