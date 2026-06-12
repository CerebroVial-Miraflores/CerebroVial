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
  CongestionSeriesResponse,
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

describe('congestionService.getSeries', () => {
  beforeEach(() => {
    getMock.mockReset();
  });

  it('hace GET a /congestion/series con el día y devuelve el CongestionSeriesResponse', async () => {
    const payload: CongestionSeriesResponse = {
      day: '2025-01-06',
      t0: '2025-01-06T00:00:00',
      step_s: 300,
      coverage_end: '2025-01-06T23:59:00',
      count: 2,
      edges: [
        { edge_id: '-129822384#0', levels: [0, 1, 2] },
        { edge_id: '-129822384#1', levels: [3, 4, 5] },
      ],
    };
    getMock.mockResolvedValue({ data: payload });

    const result = await congestionService.getSeries('2025-01-06');

    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith('/congestion/series', {
      params: { day: '2025-01-06' },
    });
    expect(result).toEqual(payload);
    expect(result.edges[0].levels).toEqual([0, 1, 2]);
  });
});

// FASE 2 rediseño UI — passthrough del signal de cancelación. Las firmas se
// extendieron con `opts?: { signal? }` trailing; sin signal, las llamadas de
// arriba conservan su forma original (lo garantizan los asserts existentes).
describe('congestionService — passthrough de signal (FASE 2)', () => {
  beforeEach(() => {
    getMock.mockReset();
    getMock.mockResolvedValue({ data: {} });
  });

  it('getGeometry pasa el signal a httpClient', async () => {
    const controller = new AbortController();
    await congestionService.getGeometry({ signal: controller.signal });
    expect(getMock).toHaveBeenCalledWith('/congestion/geometry', {
      signal: controller.signal,
    });
  });

  it('getState pasa el signal a httpClient', async () => {
    const controller = new AbortController();
    await congestionService.getState({ signal: controller.signal });
    expect(getMock).toHaveBeenCalledWith('/congestion/state', {
      signal: controller.signal,
    });
  });

  it('getSeries combina params y signal en el mismo config', async () => {
    const controller = new AbortController();
    await congestionService.getSeries('2026-06-10', { signal: controller.signal });
    expect(getMock).toHaveBeenCalledWith('/congestion/series', {
      params: { day: '2026-06-10' },
      signal: controller.signal,
    });
  });

  it('getPrediction combina t y signal; sin ninguno conserva la forma original', async () => {
    const controller = new AbortController();
    await congestionService.getPrediction(720, { signal: controller.signal });
    expect(getMock).toHaveBeenCalledWith('/congestion/prediction', {
      params: { t: 720 },
      signal: controller.signal,
    });

    getMock.mockClear();
    await congestionService.getPrediction();
    // Forma original preservada: config undefined explícito cuando no hay nada.
    expect(getMock).toHaveBeenCalledWith('/congestion/prediction', undefined);
  });
});
