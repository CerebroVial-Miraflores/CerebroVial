/**
 * Tests del cliente REST de intersecciones (FASE 2 rediseño UI).
 *
 * Patrón A (congestionService.test.ts): mockea httpClient y verifica URL,
 * shape devuelto tal cual, y passthrough del signal de cancelación.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { intersectionsService } from '../intersectionsService';
import { httpClient } from '../httpClient';
import type { IntersectionSummary } from '../../types/intersections';

vi.mock('../httpClient', () => ({
  httpClient: { get: vi.fn() },
}));

const getMock = httpClient.get as unknown as ReturnType<typeof vi.fn>;

const payload: IntersectionSummary[] = [
  {
    id: 'larco_schell',
    name: 'Av. Larco / Schell',
    speed: 22,
    flow: 540,
    status: 'Moderado',
    lat: -12.121,
    lng: -77.029,
    stream_url: 'https://claro.example/larco.m3u8',
  },
];

describe('intersectionsService.getIntersections', () => {
  beforeEach(() => {
    getMock.mockReset();
  });

  it('hace GET a /api/intersections (sin config cuando no hay signal) y devuelve la lista', async () => {
    getMock.mockResolvedValue({ data: payload });

    const result = await intersectionsService.getIntersections();

    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith('/api/intersections');
    expect(result).toEqual(payload);
    expect(result[0].stream_url).toBe('https://claro.example/larco.m3u8');
  });

  it('pasa el signal a httpClient cuando se provee', async () => {
    getMock.mockResolvedValue({ data: payload });
    const controller = new AbortController();

    await intersectionsService.getIntersections({ signal: controller.signal });

    expect(getMock).toHaveBeenCalledWith('/api/intersections', {
      signal: controller.signal,
    });
  });
});
