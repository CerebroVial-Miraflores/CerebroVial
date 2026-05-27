// HU-05 / CA-05.3 + CA-05.4 — Cliente SSE para el stream de estrategia
// vigente.
//
// EventSource nativo NO acepta headers custom, así que no podemos pasarle el
// Bearer del SessionContext. Usamos @microsoft/fetch-event-source, que está
// implementado sobre fetch y sí permite headers — reusa el JWT del authBridge
// (HU-01) de forma simétrica al httpClient axios.
//
// Regla crítica anti-loop: ante un 401 (token expirado o inválido), invocamos
// authBridge.onUnauthorized() y lanzamos FatalSSEError para que la librería
// NO reintente la conexión. Sin esto, el cliente entraría en un loop de
// reconexión con 401 cada vez, golpeando el backend y disparando logouts en
// cascada.
//
// CA-05.4 (Fase 6): backoff exponencial 1s → 2s → 4s → 8s → 16s (cap), con
// reset al primer evento de cada nueva conexión. La intención: bajo
// indisponibilidad del backend, no martillar reconexiones; al volver el
// servicio, una recuperación rápida reanuda el flujo normal.
import {
  fetchEventSource,
  EventStreamContentType,
} from '@microsoft/fetch-event-source';
import { authBridge } from '../auth/authBridge';

const baseURL = import.meta.env.VITE_CORE_API_URL ?? 'http://localhost:8001';

const BACKOFF_MS: readonly number[] = [1000, 2000, 4000, 8000, 16000] as const;

/**
 * Devuelve el delay (ms) antes del próximo intento de reconexión SSE. Cap a
 * los 16 s del último step; cualquier ``attempt`` ≥ 4 retorna 16 s. Exportado
 * para test unitario directo del backoff.
 */
export function backoffMsForAttempt(attempt: number): number {
  if (attempt < 0) return BACKOFF_MS[0];
  return BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];
}

export interface SSEMessage {
  type: string;
  data: unknown;
}

export interface SSEClientOptions {
  onMessage: (msg: SSEMessage) => void;
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
 * Abre el stream SSE de ``/control/active-state/{nodeId}/stream`` y devuelve
 * un AbortController para cerrarlo. El caller debe invocar
 * ``controller.abort()`` en su cleanup (useEffect return) para detener la
 * conexión y desuscribirse del broadcaster del backend.
 */
export function openControlActiveStateStream(
  nodeId: string,
  opts: SSEClientOptions,
): AbortController {
  const controller = new AbortController();
  const token = authBridge.getToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  // Contador de reintentos por conexión. Se RESETEA a 0 cada vez que se
  // recibe un mensaje (CA-05.4: "reset al primer evento de la nueva
  // conexión"). Vive en el closure del openControlActiveStateStream, así
  // que es per-call: cada AbortController tiene su propia secuencia.
  let retryAttempt = 0;

  // fetchEventSource devuelve una Promise que se resuelve cuando el stream
  // se cierra. No la awaitamos: el control de vida del stream va por el
  // AbortController que devolvemos al caller.
  void fetchEventSource(`${baseURL}/control/active-state/${nodeId}/stream`, {
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
        // Token expirado/inválido: dispara el flujo de auto-logout del
        // SessionContext (HU-01) y CORTA la reconexión — sin esto el cliente
        // entraría en un loop 401.
        authBridge.onUnauthorized();
        throw new FatalSSEError('401 Unauthorized');
      }
      throw new FatalSSEError(
        `Stream rechazado con status ${response.status}`,
      );
    },
    onmessage(ev) {
      // Filtro CA-05.4: solo eventos del dominio cuentan como "señal del
      // motor". Cualquier otra cosa que el parser de SSE despache se ignora.
      //
      // En particular:
      // - sse-starlette envía un PING cada DEFAULT_PING_INTERVAL=15s como
      //   keep-alive de proxies: ``: ping - <fecha>\n\n``. Es una línea de
      //   COMENTARIO SSE; @microsoft/fetch-event-source la procesa
      //   correctamente (no setea ningún campo) pero DISPATCHA onmessage
      //   por la empty-line terminadora con ``{event:'', data:''}``.
      // - Si propagáramos ese ping como evento de negocio, el caller haría
      //   re-fetch + reset del staleTimer cada 15s — el banner "no
      //   confirmada" curaría a los 5s sin que el motor haya emitido nada.
      //   Es la regresión exacta de CA-05.4 que este filtro corrige.
      // - Cualquier futuro tipo de evento que el server agregue (heartbeat,
      //   keepalive, etc.) queda igualmente bloqueado hasta que este cliente
      //   lo soporte explícitamente — blindaje contra cambios silenciosos
      //   del contrato.
      if (ev.event !== 'active-state-changed') return;

      // CA-05.4: reset del backoff SOLO al recibir un evento de negocio.
      retryAttempt = 0;
      let parsed: unknown = ev.data;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        // Si el payload no es JSON válido, propagamos el string crudo.
      }
      opts.onMessage({ type: ev.event, data: parsed });
    },
    onerror(err) {
      // FatalSSEError → re-lanzar detiene los reintentos automáticos.
      if (err instanceof FatalSSEError) {
        throw err;
      }
      opts.onError?.(err);
      // CA-05.4: backoff exponencial. Retornar un número en ms le indica a
      // la librería el delay antes del próximo intento. Sin esto la
      // librería usa su default (~1 s constante).
      const delay = backoffMsForAttempt(retryAttempt);
      retryAttempt += 1;
      return delay;
    },
    onclose() {
      opts.onClose?.();
    },
  }).catch(() => {
    // FatalSSEError ya disparó el side-effect (onUnauthorized). Silenciamos
    // el rechazo para no contaminar consola con "Unhandled rejection".
  });

  return controller;
}
