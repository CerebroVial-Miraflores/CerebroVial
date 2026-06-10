// FASE 2 rediseño UI — máquina de estados compartida de los hooks REST.
//
// Interno a src/hooks/ (las vistas importan los hooks públicos, no esto). Se
// exporta solo para su test directo. Decisión de fase: UN helper en vez de
// duplicar el esqueleto éxito/error/loading/refetch/abort en 6 hooks — la
// duplicación deliberada de los SSE clients protege lógica sutil (filtro de
// pings CA-05.4); acá sería solo superficie de bugs.
//
// Reglas de diseño (lint react-hooks v7 del repo):
// - Ningún setState sincrónico en el body del effect (set-state-in-effect):
//   runFetch difiere todo setState detrás de un microtask (Promise.resolve).
// - Date.now() solo en callbacks async, nunca en render (purity).
// - El caller estabiliza el fetcher con useCallback([params del endpoint]);
//   cambio de identidad del fetcher = nuevo recurso (reset + abort del vuelo
//   anterior — que el hook de la cámara B no muestre data de la cámara A).
// - Un solo vuelo a la vez (last-call-wins): runFetch aborta el anterior antes
//   de disparar; imposible que una respuesta vieja pise una nueva.
import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import type { RestResource } from '../types';

interface ResourceRecord<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  lastUpdated: number | null;
}

/**
 * Traduce un error de fetch a mensaje para la superficie { data, error,
 * loading } (los hooks nunca lanzan a render). Axios-aware, en español.
 */
export function toErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    if (err.response) {
      return `El servidor respondió ${err.response.status}.`;
    }
    return 'Error de red al conectar con el servidor.';
  }
  if (err instanceof Error && err.message) {
    return err.message;
  }
  return 'Error desconocido.';
}

export interface UseRestResourceOptions {
  /** Polling: re-fetch silencioso cada intervalMs. Sin valor = sin polling. */
  intervalMs?: number;
  /** false = el recurso no se consulta (param requerido aún no disponible). */
  enabled?: boolean;
}

export function useRestResource<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  opts?: UseRestResourceOptions,
): RestResource<T> {
  const enabled = opts?.enabled ?? true;
  const intervalMs = opts?.intervalMs;

  const [record, setRecord] = useState<ResourceRecord<T>>(() => ({
    data: null,
    loading: enabled,
    error: null,
    lastUpdated: null,
  }));

  // AbortController del vuelo vigente. El cleanup del effect principal lo
  // aborta (unmount o cambio de fetcher) → axios cancela la request y el
  // settle se silencia (guard signal.aborted: cero setState post-unmount).
  const controllerRef = useRef<AbortController | null>(null);
  const firstRunRef = useRef(true);

  const runFetch = useCallback(
    async (reset: boolean) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const { signal } = controller;

      // Microtask: garantiza que ningún setState corra sincrónico dentro del
      // body del effect que invoca runFetch (react-hooks/set-state-in-effect).
      await Promise.resolve();
      if (signal.aborted) return;

      if (reset) {
        setRecord({ data: null, loading: true, error: null, lastUpdated: null });
      }

      try {
        const data = await fetcher(signal);
        if (signal.aborted) return;
        setRecord({ data, loading: false, error: null, lastUpdated: Date.now() });
      } catch (err) {
        // Cancelación ≠ error: un abort (unmount, cambio de params, last-call-
        // wins) no debe pintar "Error de red" fantasma.
        if (signal.aborted || axios.isCancel(err)) return;
        // Error real: se conserva data stale (espeja CA-05.4) y NO avanza
        // lastUpdated — isStale mide antigüedad real del dato.
        setRecord((prev) => ({
          ...prev,
          loading: false,
          error: toErrorMessage(err),
        }));
      }
    },
    [fetcher],
  );

  // Carga inicial + reset por cambio de fetcher (nuevo recurso). El primer
  // run no resetea: el estado inicial ya sale del initializer de useState.
  useEffect(() => {
    if (!enabled) return;
    const isFirst = firstRunRef.current;
    firstRunRef.current = false;
    void runFetch(!isFirst);
    return () => {
      controllerRef.current?.abort();
    };
  }, [runFetch, enabled]);

  // Polling opcional, siempre silencioso. Cleanup cancela el interval; el
  // vuelo en curso lo aborta el cleanup del effect principal.
  useEffect(() => {
    if (!enabled || intervalMs === undefined) return;
    const id = window.setInterval(() => {
      void runFetch(false);
    }, intervalMs);
    return () => {
      window.clearInterval(id);
    };
  }, [runFetch, enabled, intervalMs]);

  const refetch = useCallback(async () => {
    if (!enabled) return;
    await runFetch(false);
  }, [runFetch, enabled]);

  return {
    data: record.data,
    loading: record.loading,
    error: record.error,
    lastUpdated: record.lastUpdated,
    refetch,
  };
}
