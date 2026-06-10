/**
 * Tests del cliente REST de cámaras (FASE 2 rediseño UI).
 *
 * Patrón A: mockea httpClient y verifica URL, shape y passthrough del signal.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { camerasService } from '../camerasService';
import { httpClient } from '../httpClient';
import type { CameraSummary } from '../../types/cameras';

vi.mock('../httpClient', () => ({
  httpClient: { get: vi.fn() },
}));

const getMock = httpClient.get as unknown as ReturnType<typeof vi.fn>;

const payload: CameraSummary[] = [
  { id: 'cam_01', name: 'Av. Larco / Schell', stream_url: 'https://claro.example/larco.m3u8' },
  { id: 'cam_02', name: 'Av. Pardo / Comandante Espinar', stream_url: null },
];

describe('camerasService.getCameras', () => {
  beforeEach(() => {
    getMock.mockReset();
  });

  it('hace GET a /api/cameras (sin config cuando no hay signal) y devuelve la lista liviana', async () => {
    getMock.mockResolvedValue({ data: payload });

    const result = await camerasService.getCameras();

    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith('/api/cameras');
    expect(result).toEqual(payload);
    expect(result[1].stream_url).toBeNull();
  });

  it('pasa el signal a httpClient cuando se provee', async () => {
    getMock.mockResolvedValue({ data: payload });
    const controller = new AbortController();

    await camerasService.getCameras({ signal: controller.signal });

    expect(getMock).toHaveBeenCalledWith('/api/cameras', {
      signal: controller.signal,
    });
  });
});
