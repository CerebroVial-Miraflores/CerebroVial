/**
 * Tests del cliente REST de congestión (HU-22, Fase 1).
 *
 * Mockea httpClient (patrón A, vi.mock hoisted) y verifica que cada método pega a
 * su URL protegida y devuelve el shape parseado tal cual del cuerpo de la respuesta.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { congestionService } from '../congestionService';
import { httpClient } from '../httpClient';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
} from '../../types/congestion';

vi.mock('../httpClient', () => ({
  httpClient: { get: vi.fn() },
}));

const getMock = httpClient.get as unknown as ReturnType<typeof vi.fn>;

describe('congestionService.getGeometry', () => {
  beforeEach(() => {
    getMock.mockReset();
  });

  it('hace GET a /congestion/geometry y devuelve el FeatureCollection', async () => {
    const payload: GeometryFeatureCollection = {
      type: 'FeatureCollection',
      count: 1,
      features: [
        {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: [
              [-77.0335, -12.118],
              [-77.0334, -12.1179],
            ],
          },
          properties: {
            edge_id: '-129822384#0',
            source_node: 'sumo_138854736',
            target_node: 'sumo_262576671',
            distance_m: 241.2,
            lanes: 1,
          },
        },
      ],
    };
    getMock.mockResolvedValue({ data: payload });

    const result = await congestionService.getGeometry();

    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith('/congestion/geometry');
    expect(result).toEqual(payload);
    expect(result.features[0].properties.edge_id).toBe('-129822384#0');
  });
});

describe('congestionService.getState', () => {
  beforeEach(() => {
    getMock.mockReset();
  });

  it('hace GET a /congestion/state y devuelve el CongestionStateResponse', async () => {
    const payload: CongestionStateResponse = {
      count: 2,
      edges: [
        { edge_id: '-129822384#0', congestion_level: 0, snapshot_timestamp: '2025-01-06T23:59:00' },
        { edge_id: '-129822384#1', congestion_level: 3, snapshot_timestamp: '2025-01-06T23:59:00' },
      ],
    };
    getMock.mockResolvedValue({ data: payload });

    const result = await congestionService.getState();

    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith('/congestion/state');
    expect(result).toEqual(payload);
    expect(result.edges[1].congestion_level).toBe(3);
  });
});
