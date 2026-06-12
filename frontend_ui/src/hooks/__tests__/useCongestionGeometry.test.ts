/**
 * Tests de useCongestionGeometry (FASE 2 rediseño UI).
 *
 * Foco: el cache de sesión a nivel módulo — un solo GET para múltiples
 * montajes (incluidos concurrentes, caso StrictMode), fallo que no envenena
 * el cache, y unmount durante el fetch compartido sin act warnings.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { resetGeometryCacheForTests, useCongestionGeometry } from '../useCongestionGeometry';
import { congestionService } from '../../services/congestionService';
import type { GeometryFeatureCollection } from '../../types/congestion';

vi.mock('../../services/congestionService', () => ({
  congestionService: { getGeometry: vi.fn() },
}));

const getGeometryMock = congestionService.getGeometry as unknown as ReturnType<typeof vi.fn>;

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

beforeEach(() => {
  getGeometryMock.mockReset();
  resetGeometryCacheForTests();
});

describe('useCongestionGeometry — cache de sesión', () => {
  it('dos montajes secuenciales → UN solo GET (cache de sesión)', async () => {
    getGeometryMock.mockResolvedValue(payload);

    const first = renderHook(() => useCongestionGeometry());
    await act(async () => {});
    expect(first.result.current.data).toEqual(payload);
    first.unmount();

    const second = renderHook(() => useCongestionGeometry());
    await act(async () => {});

    expect(second.result.current.data).toEqual(payload);
    expect(getGeometryMock).toHaveBeenCalledTimes(1);
  });

  it('dos montajes ANTES de resolver → UN solo GET (dedupe de promise, caso StrictMode)', async () => {
    let resolveGet!: (v: GeometryFeatureCollection) => void;
    getGeometryMock.mockImplementation(
      () => new Promise<GeometryFeatureCollection>((res) => (resolveGet = res)),
    );

    const a = renderHook(() => useCongestionGeometry());
    const b = renderHook(() => useCongestionGeometry());
    await act(async () => {});
    expect(getGeometryMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveGet(payload);
    });

    expect(a.result.current.data).toEqual(payload);
    expect(b.result.current.data).toEqual(payload);
  });

  it('fallo NO envenena el cache: refetch reintenta de verdad y recupera', async () => {
    getGeometryMock
      .mockRejectedValueOnce(new Error('core caído'))
      .mockResolvedValueOnce(payload);

    const { result } = renderHook(() => useCongestionGeometry());
    await act(async () => {});
    expect(result.current.error).toBe('core caído');
    expect(result.current.data).toBeNull();

    await act(async () => {
      await result.current.refetch();
    });

    expect(getGeometryMock).toHaveBeenCalledTimes(2);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toEqual(payload);
  });

  it('refetch tras éxito devuelve el cache sin red nueva', async () => {
    getGeometryMock.mockResolvedValue(payload);

    const { result } = renderHook(() => useCongestionGeometry());
    await act(async () => {});

    await act(async () => {
      await result.current.refetch();
    });

    expect(getGeometryMock).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual(payload);
  });

  it('unmount durante el fetch compartido: sin setState posterior ni act warnings', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    let resolveGet!: (v: GeometryFeatureCollection) => void;
    getGeometryMock.mockImplementation(
      () => new Promise<GeometryFeatureCollection>((res) => (resolveGet = res)),
    );

    const { unmount } = renderHook(() => useCongestionGeometry());
    await act(async () => {});
    unmount();

    resolveGet(payload);
    await Promise.resolve();
    await Promise.resolve();

    const actWarnings = errorSpy.mock.calls.filter((args) =>
      args.some((a) => typeof a === 'string' && a.includes('not wrapped in act')),
    );
    expect(actWarnings).toHaveLength(0);
    errorSpy.mockRestore();
  });
});
