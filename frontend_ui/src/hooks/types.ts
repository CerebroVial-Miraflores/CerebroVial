// FASE 2 rediseño UI — tipos públicos de la capa de hooks de datos.
//
// Superficie uniforme {data, error, loading} (regla de fase: los hooks nunca
// lanzan a render). Las vistas importan de acá; internal/ no se importa fuera
// de src/hooks/.

export interface RestResource<T> {
  /** Última respuesta exitosa. Se CONSERVA stale ante un error de refetch. */
  data: T | null;
  /**
   * true SOLO mientras no hay data para los params vigentes (carga inicial o
   * cambio de params). Los refetch (manuales, polling, wakes SSE) son
   * silenciosos: no encienden loading.
   */
  loading: boolean;
  /** Mensaje en español del último intento fallido; null tras el siguiente éxito. */
  error: string | null;
  /**
   * Status HTTP del último intento fallido (null si no fue HTTP o tras éxito).
   * FASE 3: permite discriminar respuestas-contrato (p. ej. el 404
   * `no_active_state` de /control/active-state) sin matchear strings de error.
   */
  errorStatus: number | null;
  /** Epoch ms del último éxito. NO avanza en error (mide antigüedad real). */
  lastUpdated: number | null;
  /** Refresh silencioso. Nunca rechaza: el resultado se refleja en data/error. */
  refetch: () => Promise<void>;
}

/**
 * Estado de conexión de un stream (SSE core o EventSource del edge), derivado
 * de los callbacks del cliente: inicial 'connecting'; onOpen → 'open';
 * onError → 'retrying' (los clients auto-reintentan con backoff 1→16 s);
 * onClose → 'closed'.
 *
 * CAVEAT 401 (SSE del core): ante un 401, sseClient/congestionSseClient
 * lanzan FatalSSEError y su `.catch` interno silencia el rechazo — NI onError
 * NI onClose del consumidor se invocan, así que este estado queda CONGELADO
 * en su último valor (típicamente 'connecting', o 'open' si el token expiró
 * con el stream ya abierto). Es aceptable por diseño: el mismo 401 disparó
 * `authBridge.onUnauthorized()` → auto-logout global → navegación a /login
 * que desmonta el hook. Si alguna vez se observa un chip 'open' con un stream
 * muerto FUERA de ese flujo de logout, el sospechoso es este caveat — no
 * intentar "detectarlo" en los hooks: la corrección iría en los SSE clients
 * (deuda: su unificación, cuando muera v1).
 */
export type StreamConnectionState = 'connecting' | 'open' | 'retrying' | 'closed';
