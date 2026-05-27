// HU-05 / CA-05.4 — Tests del backoff exponencial del cliente SSE.
//
// Dos niveles:
// 1. Unitario directo sobre ``backoffMsForAttempt`` (secuencia y cap).
// 2. Integración: mocking de ``@microsoft/fetch-event-source`` para
//    interceptar los handlers que ``openControlActiveStateStream`` pasa,
//    simular un ciclo de ``onerror`` y verificar la secuencia de delays;
//    luego simular un ``onmessage`` y verificar el RESET del contador.
//
// Mocking del módulo entero antes de los imports — Vitest hoistea ``vi.mock``
// automáticamente.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { EventSourceMessage } from '@microsoft/fetch-event-source';

vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(),
  EventStreamContentType: 'text/event-stream',
}));

import { fetchEventSource } from '@microsoft/fetch-event-source';
import {
  backoffMsForAttempt,
  openControlActiveStateStream,
} from '../sseClient';

const fetchEventSourceMock = vi.mocked(fetchEventSource);

interface CapturedOptions {
  onmessage?: (ev: EventSourceMessage) => void;
  onerror?: (err: unknown) => number | void;
}

function captureFetchEventSourceCallbacks(): CapturedOptions {
  const captured: CapturedOptions = {};
  fetchEventSourceMock.mockImplementation((_url, opts) => {
    // El segundo arg incluye los callbacks que sseClient define en su
    // closure; los capturamos para invocarlos directamente desde el test.
    captured.onmessage = opts.onmessage;
    captured.onerror = opts.onerror;
    // La librería real devuelve una Promise que se resuelve al cerrar el
    // stream. Devolvemos una Promise nunca-resuelta para mimetizar la
    // conexión viva durante el test.
    return new Promise<void>(() => {});
  });
  return captured;
}

describe('backoffMsForAttempt — secuencia y cap (CA-05.4)', () => {
  it('intento 0 → 1 s, 1 → 2 s, 2 → 4 s, 3 → 8 s, 4 → 16 s', () => {
    expect(backoffMsForAttempt(0)).toBe(1_000);
    expect(backoffMsForAttempt(1)).toBe(2_000);
    expect(backoffMsForAttempt(2)).toBe(4_000);
    expect(backoffMsForAttempt(3)).toBe(8_000);
    expect(backoffMsForAttempt(4)).toBe(16_000);
  });

  it('intentos posteriores quedan capeados en 16 s (no crecen indefinidamente)', () => {
    expect(backoffMsForAttempt(5)).toBe(16_000);
    expect(backoffMsForAttempt(10)).toBe(16_000);
    expect(backoffMsForAttempt(100)).toBe(16_000);
  });

  it('valores negativos (defensa) caen al primer paso (1 s)', () => {
    expect(backoffMsForAttempt(-1)).toBe(1_000);
    expect(backoffMsForAttempt(-100)).toBe(1_000);
  });
});

describe('openControlActiveStateStream — backoff aplicado al onerror', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('secuencia onerror sin mensajes intermedios: 1s, 2s, 4s, 8s, 16s, 16s', () => {
    const captured = captureFetchEventSourceCallbacks();

    openControlActiveStateStream('larco_schell', {
      onMessage: vi.fn(),
    });

    expect(captured.onerror).toBeDefined();

    // Cada onerror retorna el delay y luego incrementa el contador.
    expect(captured.onerror!(new Error('a'))).toBe(1_000);
    expect(captured.onerror!(new Error('b'))).toBe(2_000);
    expect(captured.onerror!(new Error('c'))).toBe(4_000);
    expect(captured.onerror!(new Error('d'))).toBe(8_000);
    expect(captured.onerror!(new Error('e'))).toBe(16_000);
    // Cap: el sexto error sigue en 16 s.
    expect(captured.onerror!(new Error('f'))).toBe(16_000);
  });

  it('onmessage RESETEA el contador → el siguiente onerror vuelve a 1 s', () => {
    const captured = captureFetchEventSourceCallbacks();
    const onMessage = vi.fn();
    openControlActiveStateStream('larco_schell', { onMessage });

    // Empujamos el contador a 4 s (3 errores).
    captured.onerror!(new Error('1'));
    captured.onerror!(new Error('2'));
    expect(captured.onerror!(new Error('3'))).toBe(4_000);

    // Llega un evento útil → reset del contador.
    captured.onmessage!({
      event: 'active-state-changed',
      data: '{"node_id":"x"}',
      id: '',
      retry: undefined,
    } as EventSourceMessage);
    expect(onMessage).toHaveBeenCalledOnce();

    // El próximo error vuelve a 1 s.
    expect(captured.onerror!(new Error('4'))).toBe(1_000);
    expect(captured.onerror!(new Error('5'))).toBe(2_000);
  });

  it('cada call a openControlActiveStateStream tiene su PROPIO contador (closure independiente)', () => {
    const a = captureFetchEventSourceCallbacks();
    openControlActiveStateStream('node_a', { onMessage: vi.fn() });
    const optsA = { ...a };

    // Avanzamos contador del stream A a 8 s.
    optsA.onerror!(new Error('a1'));
    optsA.onerror!(new Error('a2'));
    optsA.onerror!(new Error('a3'));
    expect(optsA.onerror!(new Error('a4'))).toBe(8_000);

    // Abrimos un segundo stream — debe empezar de cero.
    const b = captureFetchEventSourceCallbacks();
    openControlActiveStateStream('node_b', { onMessage: vi.fn() });
    expect(b.onerror!(new Error('b1'))).toBe(1_000);
  });
});
