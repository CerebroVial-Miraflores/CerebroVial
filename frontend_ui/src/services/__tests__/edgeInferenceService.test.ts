/**
 * Tests del cliente de inferencia del edge (toggle alta/baja on-demand).
 *
 * El edge va CRUDO: fetch directo, sin httpClient/JWT. Se stubea global.fetch y se
 * verifica URL, método, body, parseo de la respuesta y el mapeo del 409 →
 * EdgeCapacityError (que el toggle usa para revertir a BAJA sin romper).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  edgeInferenceService,
  EdgeCapacityError,
  type InferenceStatus,
} from '../edgeInferenceService';

const EDGE = 'http://localhost:8000';

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.clearAllMocks();
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

describe('edgeInferenceService.getInferenceStatus', () => {
  it('hace GET a /cameras/inference-status y devuelve el shape parseado', async () => {
    const payload: InferenceStatus = {
      inferring: ['cam_a'],
      count: 1,
      cap: null,
      capacity_used: null,
    };
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve(payload) });

    const result = await edgeInferenceService.getInferenceStatus();

    expect(fetchMock).toHaveBeenCalledWith(`${EDGE}/cameras/inference-status`, {});
    expect(result).toEqual(payload);
  });

  it('lanza si el edge responde no-ok', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 503 });
    await expect(edgeInferenceService.getInferenceStatus()).rejects.toThrow(/503/);
  });

  it('pasa el signal cuando se provee', async () => {
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    const controller = new AbortController();
    await edgeInferenceService.getInferenceStatus({ signal: controller.signal });
    expect(fetchMock).toHaveBeenCalledWith(`${EDGE}/cameras/inference-status`, {
      signal: controller.signal,
    });
  });
});

describe('edgeInferenceService.startInference', () => {
  it('hace POST con el body de alta y content-type JSON', async () => {
    fetchMock.mockResolvedValue({ ok: true, status: 200 });

    await edgeInferenceService.startInference('cam_a', {
      source: 'https://x/a.m3u8',
      source_type: 'hls',
      zones: {},
    });

    expect(fetchMock).toHaveBeenCalledWith(
      `${EDGE}/cameras/cam_a`,
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(body).toEqual({ source: 'https://x/a.m3u8', source_type: 'hls', zones: {} });
  });

  it('mapea el 409 a EdgeCapacityError', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 409 });
    await expect(
      edgeInferenceService.startInference('cam_a', { source: 's', source_type: 'hls', zones: {} }),
    ).rejects.toBeInstanceOf(EdgeCapacityError);
  });

  it('lanza un error genérico ante otro no-ok', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 500 });
    await expect(
      edgeInferenceService.startInference('cam_a', { source: 's', source_type: 'hls', zones: {} }),
    ).rejects.toThrow(/500/);
  });
});

describe('edgeInferenceService.stopInference', () => {
  it('hace DELETE a /cameras/{id}', async () => {
    fetchMock.mockResolvedValue({ ok: true, status: 200 });
    await edgeInferenceService.stopInference('cam_a');
    expect(fetchMock).toHaveBeenCalledWith(`${EDGE}/cameras/cam_a`, { method: 'DELETE' });
  });

  it('lanza si el edge responde no-ok', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 404 });
    await expect(edgeInferenceService.stopInference('cam_a')).rejects.toThrow(/404/);
  });
});
