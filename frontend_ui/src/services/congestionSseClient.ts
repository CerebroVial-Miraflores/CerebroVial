// HU-22 / CA-22.2 — Cliente SSE para el feed de congestión de red.
//
// Espeja `openControlActiveStateStream` (HU-05, sseClient.ts) pero apunta al
// stream network-wide de congestión. Es un archivo/función NUEVOS: NO se toca
// sseClient.ts ni se lo vuelve genérico. Duplicación deliberada — acoplar HU-22
// al cliente de control (o generalizarlo) arriesgaría una regresión en el filtro
// de pings de CA-05.4. Las piezas pequeñas (BACKOFF_MS, FatalSSEError, el helper
// de delay) se duplican local para que HU-22 sea autocontenida.
//
// EventSource nativo NO acepta headers custom, así que usamos
// @microsoft/fetch-event-source (sobre fetch) para pasar el Bearer del authBridge
// (HU-01), simétrico al httpClient axios.
//
// Anti-loop: ante un 401 invocamos authBridge.onUnauthorized() y lanzamos
// FatalSSEError para cortar la reconexión automática (sin esto, loop de 401).
//
// Backoff exponencial 1s → 2s → 4s → 8s → 16s (cap), con reset al primer evento
// de negocio de cada nueva conexión.
import {
  fetchEventSource,
  EventStreamContentType,
} from '@microsoft/fetch-event-source';
import { authBridge } from '../auth/authBridge';

const baseURL = import.meta.env.VITE_CORE_API_URL ?? 'http://localhost:8001';

const BACKOFF_MS: readonly number[] = [1000, 2000, 4000, 8000, 16000] as const;

/**
 * Delay (ms) antes del próximo intento de reconexión SSE. Cap al último step
 * (16 s); cualquier ``attempt`` ≥ 4 retorna 16 s. Exportado para test unitario.
 */
export function congestionBackoffMsForAttempt(attempt: number): number {
  if (attempt < 0) return BACKOFF_MS[0];
  return BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];
}

export interface CongestionStreamOptions {
  /**
   * Wake del feed: el backend emite ``congestion-updated`` SIN payload
   * semántico, así que el callback solo señala "re-leé el estado". El caller
   * re-consulta ``getState()`` y recolorea.
   */
  onWake: () => void;
  onOpen?: () => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
}

class FatalSSEError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FatalSSEError';
  }
}

/**
 * Abre el stream SSE de ``/congestion/state/stream`` (network-wide, sin
 * ``nodeId``) y devuelve un AbortController para cerrarlo. El caller debe
 * invocar ``controller.abort()`` en su cleanup (useEffect return) para detener
 * la conexión y desuscribirse del broadcaster del backend.
 */
export function openCongestionStream(
  opts: CongestionStreamOptions,
): AbortController {
  const controller = new AbortController();
  const token = authBridge.getToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  // Contador de reintentos por conexión. Se RESETEA a 0 al recibir un evento de
  // negocio. Vive en el closure: es per-call, cada AbortController tiene su
  // propia secuencia de backoff.
  let retryAttempt = 0;

  // fetchEventSource devuelve una Promise que resuelve al cerrarse el stream.
  // No la awaitamos: el control de vida va por el AbortController.
  void fetchEventSource(`${baseURL}/congestion/state/stream`, {
    method: 'GET',
    headers,
    openWhenHidden: true,
    signal: controller.signal,
    async onopen(response) {
      const contentType = response.headers.get('content-type') ?? '';
      if (response.ok && contentType.includes(EventStreamContentType)) {
        opts.onOpen?.();
        return;
      }
      if (response.status === 401) {
        // Token expirado/inválido: dispara el auto-logout (HU-01) y CORTA la
        // reconexión — sin esto el cliente entraría en un loop 401.
        authBridge.onUnauthorized();
        throw new FatalSSEError('401 Unauthorized');
      }
      throw new FatalSSEError(`Stream rechazado con status ${response.status}`);
    },
    onmessage(ev) {
      // Filtro: solo ``congestion-updated`` cuenta como wake del feed. Bloquea
      // los pings de sse-starlette (``{event:'', data:''}``) y cualquier tipo
      // de evento futuro que el server agregue — blindaje contra cambios
      // silenciosos del contrato.
      if (ev.event !== 'congestion-updated') return;

      // Reset del backoff SOLO al recibir un evento de negocio. El wake no trae
      // payload útil: NO parseamos ``ev.data``, solo señalamos "re-leé".
      retryAttempt = 0;
      opts.onWake();
    },
    onerror(err) {
      // FatalSSEError → re-lanzar detiene los reintentos automáticos.
      if (err instanceof FatalSSEError) {
        throw err;
      }
      opts.onError?.(err);
      // Backoff exponencial: retornar ms le indica a la librería el delay antes
      // del próximo intento.
      const delay = congestionBackoffMsForAttempt(retryAttempt);
      retryAttempt += 1;
      return delay;
    },
    onclose() {
      opts.onClose?.();
    },
  }).catch(() => {
    // FatalSSEError ya disparó su side-effect; silenciamos el rechazo para no
    // contaminar consola con "Unhandled rejection".
  });

  return controller;
}
