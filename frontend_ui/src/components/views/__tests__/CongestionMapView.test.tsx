/**
 * Smoke + cableado de CongestionMapView (HU-22, Fase 2) — PRIMER test de mapa del repo.
 *
 * Patrón de mock de Leaflet (precedente para Fase 3 y futuros tests de mapa):
 *  - NO se monta Leaflet real: jsdom no tiene canvas ni dimensiones, el render real
 *    de tiles/SVG falla. Se stubea `react-leaflet` vía `vi.mock` hoisted, devolviendo
 *    componentes div con `data-testid` (idiom de la casa: ver CameraDetailView/App tests).
 *  - Los props que nos importan (`data` y `style` del <GeoJSON>) se capturan con un
 *    holder creado por `vi.hoisted` —referenciable dentro del factory de vi.mock—,
 *    espejo del `captureFetchEventSourceCallbacks` de sseClient.test.ts.
 *  - `congestionService` se mockea para no pegar a la red.
 *
 * El test NO valida pixeles. Afirma: (a) monta sin crashear; (b) llama getGeometry y
 * getState 1× cada uno al montar; (c) tras la carga pasa a <GeoJSON> un `data` con 375
 * features y un `style` callback que devuelve {color, weight} (cableado a congestionStyle).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { CongestionMapView } from '../CongestionMapView';
import { congestionService } from '../../../services/congestionService';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
} from '../../../types/congestion';

const captured = vi.hoisted(() => ({ props: null as null | Record<string, unknown> }));

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
  GeoJSON: (props: Record<string, unknown>) => {
    captured.props = props;
    return <div data-testid="geojson" />;
  },
}));

vi.mock('../../../services/congestionService', () => ({
  congestionService: { getGeometry: vi.fn(), getState: vi.fn() },
}));

const getGeometryMock = congestionService.getGeometry as unknown as ReturnType<typeof vi.fn>;
const getStateMock = congestionService.getState as unknown as ReturnType<typeof vi.fn>;

const COUNT = 375;

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
  // Un nivel determinista por arista (0-5) para tener estado en todas.
  return {
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      congestion_level: i % 6,
      snapshot_timestamp: '2025-01-06T23:59:00',
    })),
  };
}

describe('CongestionMapView', () => {
  beforeEach(() => {
    getGeometryMock.mockReset();
    getStateMock.mockReset();
    captured.props = null;
  });

  it('carga geometría+estado 1× cada uno y pinta 375 tramos por <GeoJSON> con style callback', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());

    render(<CongestionMapView />);

    // (b) consulta cada endpoint una sola vez al montar
    await waitFor(() => expect(screen.getByTestId('geojson')).toBeInTheDocument());
    expect(getGeometryMock).toHaveBeenCalledTimes(1);
    expect(getStateMock).toHaveBeenCalledTimes(1);

    // (c) data cruzado con 375 features
    const props = captured.props as { data: { features: unknown[] }; style: unknown };
    expect(props.data.features).toHaveLength(COUNT);

    // style es un callback y delega en congestionStyle (nivel 3 → naranja, grosor 6)
    expect(typeof props.style).toBe('function');
    const style = props.style as (f: unknown) => { color: string; weight: number };
    expect(style({ properties: { congestion_level: 3 } })).toEqual({
      color: '#F08C1D',
      weight: 6,
    });
  });

  it('muestra el bloque de error si falla la carga (no monta el mapa)', async () => {
    getGeometryMock.mockRejectedValue(new Error('boom'));
    getStateMock.mockResolvedValue(buildState());

    render(<CongestionMapView />);

    await waitFor(() =>
      expect(screen.getByText(/No se pudo cargar el mapa/i)).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('geojson')).not.toBeInTheDocument();
  });
});
