import { useEffect, useRef, useState } from 'react';

import { EDGE_API_URL } from '../../../config/edge';
import { MjpegParser } from './mjpegStreamParser';

// FASE 4 (Opción B2) — cámara activa con frame ANOTADO server-side. Consume el MJPEG
// processed del edge (GET /video/{id}?type=processed) vía fetch + ReadableStream (NO
// vía <img>: el <img> MJPEG es la causa del freeze, DEUDA-MJPEG-BROWSER) y lo pinta en
// un <canvas>. Las cajas YA vienen dibujadas en el frame por el server → cero
// alineación cliente: solo se escala el frame entero con letterbox (object-fit:contain
// del canvas, backing store 1280×720, equivalente al HlsPlayer objectFit:'contain').
//
// Reconexión propia, dos rutas que comparten backoff:
//   1) fin/error del stream (reader.read() done o throw) → reconecta YA — la ruta más
//      común en la vida real (red, restart del edge);
//   2) freeze SILENCIOSO (stream abierto, sin bytes) → lo cubre el watchdog por timeout.
// Decode tolerante: un frame corrupto se descarta sin matar la lectura ni reconectar.
// Estado visible 'reconnecting' (badge): nunca pantalla negra, nunca frame congelado
// sin rótulo. Al desmontar / cambiar de cámara se aborta el fetch → el backend
// decrementa el consumidor MJPEG y su watchdog apaga el render de esa cámara.

export type StreamStatus = 'connecting' | 'streaming' | 'reconnecting';

// El server emite a ~25 fps (sleep 0.04 en video.py) → inter-frame ~40 ms. 2.5 s ≈ 60+
// frames perdidos: por encima de hipos normales, pero un freeze real se nota en ≤2.5 s.
const FREEZE_TIMEOUT_MS = 2500;
// El primer frame tarda más (alta + carga lazy del YOLO + 1ª inferencia en CPU): timeout
// laxo hasta el 1er drawImage; recién ahí se arma el de 2.5 s. Evita reconexión espuria
// en el arranque.
const INITIAL_TIMEOUT_MS = 10000;
const WATCHDOG_INTERVAL_MS = 500;
const BACKOFF_BASE_MS = 500;
const BACKOFF_CAP_MS = 5000;
// Stall de CONTENIDO (P1): el edge sigue mandando bytes durante un corte de red río
// arriba, pero reenvía el MISMO frame (latest_frame_processed congelado mientras el
// decode reconecta). El watchdog de "sin bytes" no lo caza (los bytes fluyen); este
// umbral detecta "sin frame NUEVO" comparando la firma del jpeg → overlay "Reconectando…".
const STALE_CONTENT_MS = 2000;

interface AnnotatedCameraStreamProps {
  cameraId: string;
  /** Gate: abre el stream recién tras el alta OK (espejo del SSE/overlay previos). */
  enabled: boolean;
  onStatusChange?: (status: StreamStatus) => void;
  className?: string;
  // Costuras de test (timing): defaults = constantes de producción. Permiten valores
  // chicos con timers reales en los tests, sin fake timers.
  freezeTimeoutMs?: number;
  initialTimeoutMs?: number;
  watchdogIntervalMs?: number;
  backoffBaseMs?: number;
  backoffCapMs?: number;
  staleContentMs?: number;
}

export function AnnotatedCameraStream({
  cameraId,
  enabled,
  onStatusChange,
  className,
  freezeTimeoutMs = FREEZE_TIMEOUT_MS,
  initialTimeoutMs = INITIAL_TIMEOUT_MS,
  watchdogIntervalMs = WATCHDOG_INTERVAL_MS,
  backoffBaseMs = BACKOFF_BASE_MS,
  backoffCapMs = BACKOFF_CAP_MS,
  staleContentMs = STALE_CONTENT_MS,
}: AnnotatedCameraStreamProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<StreamStatus>('connecting');

  // onStatusChange por ref → no entra en las deps del effect (evita reconectar al
  // re-renderizar el padre con una función nueva).
  const onStatusChangeRef = useRef(onStatusChange);
  onStatusChangeRef.current = onStatusChange;

  useEffect(() => {
    if (!enabled) return;

    const url = `${EDGE_API_URL}/video/${cameraId}?type=processed`;

    // Estado mutable de la conexión (refs: no disparan render).
    let closed = false; // true solo en el cleanup deliberado (distingue de un abort del watchdog)
    let gotFirstFrame = false;
    let attempt = 0;
    let lastFrameAt = Date.now();      // último frame con BYTES (para el watchdog de stream muerto)
    let lastNovelAt = Date.now();      // último frame con CONTENIDO nuevo (para el stall de decode)
    let lastSig = '';                   // firma del último jpeg (detecta frame reenviado idéntico)
    let controller: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    // Firma barata del jpeg: largo + 3 bytes muestreados. Un frame reenviado es
    // byte-idéntico (misma firma); uno nuevo casi siempre difiere. La ventana de 2s
    // (~50 frames) hace despreciable una colisión de firma puntual.
    const sigOf = (b: Uint8Array): string =>
      `${b.length}:${b[0]}:${b[b.length >> 1]}:${b[b.length - 1]}`;

    const emit = (s: StreamStatus) => {
      if (closed) return;
      setStatus(s);
      onStatusChangeRef.current?.(s);
    };

    const drawLatest = async (jpegBytes: Uint8Array) => {
      let bitmap: ImageBitmap;
      try {
        // Copia a un ArrayBuffer concreto: el slice del parser es genérico sobre
        // ArrayBufferLike y no satisface BlobPart (podría ser SharedArrayBuffer).
        const blob = new Blob([new Uint8Array(jpegBytes)], { type: 'image/jpeg' });
        bitmap = await createImageBitmap(blob);
      } catch {
        // Frame corrupto: descartar y seguir. No mata la lectura ni reconecta.
        return;
      }
      try {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        }
      } finally {
        bitmap.close();
      }
      lastFrameAt = Date.now(); // llegaron bytes (aunque el contenido sea el mismo)
      gotFirstFrame = true;
      attempt = 0; // un stream que dropea tras estar sano reconecta rápido
      // Novedad de CONTENIDO: solo un frame nuevo marca "streaming" y resetea el reloj
      // de staleness. Un frame reenviado (decode congelado río arriba) NO lo resetea →
      // el watchdog mostrará "Reconectando…" tras `staleContentMs`.
      const sig = sigOf(jpegBytes);
      if (sig !== lastSig) {
        lastSig = sig;
        lastNovelAt = Date.now();
        emit('streaming');
      }
    };

    const scheduleReconnect = () => {
      if (closed) return;
      emit(gotFirstFrame ? 'reconnecting' : 'connecting');
      const delay = Math.min(backoffCapMs, backoffBaseMs * 2 ** attempt);
      attempt += 1;
      reconnectTimer = setTimeout(() => void connect(), delay);
    };

    const connect = async () => {
      if (closed) return;
      emit(gotFirstFrame ? 'reconnecting' : 'connecting');
      controller = new AbortController();
      const signal = controller.signal;
      let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
      try {
        const res = await fetch(url, { signal });
        if (!res.ok || !res.body) throw new Error(`edge respondió ${res.status}`);
        reader = res.body.getReader();
        const parser = new MjpegParser();
        lastFrameAt = Date.now();
        // El abort del watchdog (o del cleanup) corta el read en curso. En fetch real
        // abortar el signal ya rechaza el read; el race lo blinda igual si el body no
        // está cableado al signal.
        const aborted = new Promise<never>((_, reject) => {
          const onAbort = () => reject(new DOMException('aborted', 'AbortError'));
          if (signal.aborted) onAbort();
          else signal.addEventListener('abort', onAbort, { once: true });
        });
        for (;;) {
          const { done, value } = await Promise.race([reader.read(), aborted]);
          if (done) throw new Error('stream cerrado por el server');
          if (value && value.length > 0) {
            const frames = parser.push(value);
            // Pintar solo el más reciente del lote: liveness sobre completitud (los
            // frames intermedios de un mismo push ya quedaron viejos).
            if (frames.length > 0) await drawLatest(frames[frames.length - 1]);
          }
        }
      } catch {
        // Liberar el reader del stream abandonado (best-effort).
        if (reader) {
          try {
            await reader.cancel();
          } catch {
            /* el reader ya pudo quedar liberado por el abort */
          }
        }
        // Abort deliberado del cleanup → no reconectar. Cualquier otra cosa (done,
        // error de red, abort del watchdog) → reconectar con backoff.
        if (closed) return;
        scheduleReconnect();
      }
    };

    // Watchdog: dos fallas distintas.
    //  1) Sin BYTES (stream/edge muerto) → abortar el fetch (el catch reconecta).
    //  2) Sin CONTENIDO nuevo pero con bytes (decode río arriba congelado, el edge
    //     reenvía el mismo frame) → solo overlay "Reconectando…", NO abortar (reabrir el
    //     MJPEG no ayuda; el fix real es la reconexión in-process de ffmpeg). Al volver
    //     frames nuevos, `drawLatest` emite 'streaming' y el overlay se va.
    const watchdog = setInterval(() => {
      const timeout = gotFirstFrame ? freezeTimeoutMs : initialTimeoutMs;
      if (Date.now() - lastFrameAt > timeout) {
        lastFrameAt = Date.now(); // no re-abortar durante el backoff
        controller?.abort();
      } else if (gotFirstFrame && Date.now() - lastNovelAt > staleContentMs) {
        emit('reconnecting');
      }
    }, watchdogIntervalMs);

    void connect();

    return () => {
      closed = true;
      clearInterval(watchdog);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      controller?.abort();
    };
  }, [
    cameraId,
    enabled,
    freezeTimeoutMs,
    initialTimeoutMs,
    watchdogIntervalMs,
    backoffBaseMs,
    backoffCapMs,
    staleContentMs,
  ]);

  return (
    <div className={`absolute inset-0 ${className ?? ''}`}>
      <canvas
        ref={canvasRef}
        width={1280}
        height={720}
        aria-hidden="true"
        style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
      />
      {status === 'reconnecting' && (
        <div className="absolute inset-0 grid place-items-center">
          <div className="flex items-center gap-2 rounded-full border border-warn/30 bg-black/60 px-4 py-2 backdrop-blur-md">
            <span className="h-2 w-2 animate-pulse rounded-full bg-warn" />
            <span className="text-[11px] font-extrabold uppercase tracking-[0.12em] text-warn">
              Reconectando…
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
