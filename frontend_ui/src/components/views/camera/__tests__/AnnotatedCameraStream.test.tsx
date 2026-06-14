import { cleanup, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AnnotatedCameraStream } from '../AnnotatedCameraStream';

// FASE 4 (Opción B2) — test de ciclo de vida del stream anotado. jsdom no implementa
// canvas 2d ni createImageBitmap → se mockean. Las costuras de timing (freeze/initial/
// watchdog/backoff) van con valores chicos + timers REALES (sin fake timers) para
// testear reconexión de forma determinística.

const EDGE = 'http://localhost:8000';

function bytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}
function concat(...parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}
/** JPEG mínimo válido (SOI + payload + EOI). */
function jpeg(tag = 0): Uint8Array {
  return Uint8Array.from([0xff, 0xd8, tag, 0xff, 0xd9]);
}
function part(j: Uint8Array): Uint8Array {
  return concat(bytes('--frame\r\nContent-Type: image/jpeg\r\n\r\n'), j, bytes('\r\n'));
}
/** Bytes multipart byte-exactos con N frames + el `--frame` de cierre del último. */
function multipart(...jpegs: Uint8Array[]): Uint8Array {
  return concat(...jpegs.map(part), bytes('--frame'));
}

/** ReadableStream que encola `chunks`; cierra (done) si `close`, o queda abierto. */
function makeStream(chunks: Uint8Array[], close: boolean): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(c);
      if (close) controller.close();
    },
  });
}

let drawSpy: ReturnType<typeof vi.fn>;
let bitmapCloseSpy: ReturnType<typeof vi.fn>;
let createImageBitmapMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  drawSpy = vi.fn();
  bitmapCloseSpy = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage: drawSpy,
  } as unknown as CanvasRenderingContext2D);

  createImageBitmapMock = vi.fn(async () => ({
    close: bitmapCloseSpy,
    width: 1280,
    height: 720,
  }));
  globalThis.createImageBitmap = createImageBitmapMock as unknown as typeof createImageBitmap;
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('AnnotatedCameraStream', () => {
  it('pega a /video/{id}?type=processed y dibuja el frame (estado streaming)', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      body: makeStream([multipart(jpeg(1))], false),
    }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const onStatus = vi.fn();

    render(<AnnotatedCameraStream cameraId="cam_x" enabled onStatusChange={onStatus} />);

    await waitFor(() => expect(drawSpy).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledWith(
      `${EDGE}/video/cam_x?type=processed`,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(onStatus).toHaveBeenLastCalledWith('streaming');
    expect(bitmapCloseSpy).toHaveBeenCalled();
  });

  it('no abre el stream si enabled=false', async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<AnnotatedCameraStream cameraId="cam_x" enabled={false} />);
    await new Promise((r) => setTimeout(r, 30));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('al desmontar aborta el fetch (cierra el stream → backend decrementa el consumidor)', async () => {
    let captured: AbortSignal | null = null;
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      captured = init?.signal ?? null;
      return { ok: true, body: makeStream([multipart(jpeg(1))], false) };
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { unmount } = render(<AnnotatedCameraStream cameraId="cam_x" enabled />);
    await waitFor(() => expect(drawSpy).toHaveBeenCalled());
    expect(captured!.aborted).toBe(false);
    unmount();
    expect(captured!.aborted).toBe(true);
  });

  it('reconecta cuando el stream cierra (done), sin esperar el watchdog', async () => {
    const fetchMock = vi
      .fn()
      // 1ª conexión: un frame y CIERRA (done) → debe reconectar ya.
      .mockResolvedValueOnce({ ok: true, body: makeStream([multipart(jpeg(1))], true) })
      // 2ª conexión: queda abierta para no loopear.
      .mockResolvedValue({ ok: true, body: makeStream([multipart(jpeg(2))], false) });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const onStatus = vi.fn();

    render(
      <AnnotatedCameraStream
        cameraId="cam_x"
        enabled
        onStatusChange={onStatus}
        backoffBaseMs={5}
        backoffCapMs={20}
      />,
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(onStatus).toHaveBeenCalledWith('reconnecting');
  });

  it('reconecta por freeze SILENCIOSO (stream abierto sin bytes nuevos) vía watchdog', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      // Un frame y luego NUNCA más bytes ni close → freeze silencioso.
      body: makeStream([multipart(jpeg(1))], false),
    }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(
      <AnnotatedCameraStream
        cameraId="cam_x"
        enabled
        // Timeouts chicos + timers reales: el watchdog aborta tras el 1er frame.
        freezeTimeoutMs={40}
        initialTimeoutMs={40}
        watchdogIntervalMs={10}
        backoffBaseMs={5}
        backoffCapMs={20}
      />,
    );

    await waitFor(() => expect(drawSpy).toHaveBeenCalled());
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2), {
      timeout: 1000,
    });
  });

  it('descarta un frame corrupto (decode rechaza) sin matar la lectura ni reconectar', async () => {
    // createImageBitmap rechaza el 1er frame, acepta el 2º.
    createImageBitmapMock
      .mockRejectedValueOnce(new Error('decode falló'))
      .mockResolvedValue({ close: bitmapCloseSpy, width: 1280, height: 720 });

    const fetchMock = vi.fn(async () => ({
      ok: true,
      // Dos frames en un stream que queda abierto (sin reconexión esperada).
      body: makeStream([multipart(jpeg(1)), multipart(jpeg(2))], false),
    }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<AnnotatedCameraStream cameraId="cam_x" enabled backoffBaseMs={5} />);

    // El 2º frame (bueno) dibuja; el 1º (corrupto) no.
    await waitFor(() => expect(drawSpy).toHaveBeenCalledTimes(1));
    // Margen para descartar que haya reconectado por el frame malo.
    await new Promise((r) => setTimeout(r, 50));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(createImageBitmapMock).toHaveBeenCalledTimes(2);
  });
});
