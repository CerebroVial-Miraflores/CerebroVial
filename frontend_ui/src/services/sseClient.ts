// HU-05 / CA-05.3 — Cliente SSE para el stream de estrategia vigente.
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
// Backoff exponencial + lógica stateful "no confirmada" → Fase 6 (CA-05.4).
// Acá entregamos lo mínimo de Fase 5: conectar, parsear, propagar al caller,
// y romper el loop en 401.
import {
  fetchEventSource,
  EventStreamContentType,
} from '@microsoft/fetch-event-source';
import { authBridge } from '../auth/authBridge';

const baseURL = import.meta.env.VITE_CORE_API_URL ?? 'http://localhost:8001';

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
        // Tokens expirado/inválido: dispara el flujo de auto-logout del
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
      let parsed: unknown = ev.data;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        // Si el payload no es JSON válido, propagamos el string crudo.
      }
      opts.onMessage({ type: ev.event || 'message', data: parsed });
    },
    onerror(err) {
      // FatalSSEError → re-lanzar detiene los reintentos automáticos.
      if (err instanceof FatalSSEError) {
        throw err;
      }
      opts.onError?.(err);
      // No retornamos un número de ms: dejamos el default de la librería
      // (~1s). Backoff exponencial es alcance de Fase 6.
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
