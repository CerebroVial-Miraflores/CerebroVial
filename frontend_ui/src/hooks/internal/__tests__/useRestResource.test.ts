/**
 * Tests del helper interno useRestResource (FASE 2 rediseño UI).
 *
 * renderHook + fetchers controlados (deferred). Cubre la máquina de estados
 * completa: éxito/error/refetch, silent refresh, reset por cambio de fetcher,
 * last-call-wins, polling con fake timers, abort en unmount sin setState
 * posterior (sin act warnings) y cancelación axios silenciosa.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import axios, { AxiosError, AxiosHeaders, CanceledError } from 'axios';
import { toErrorMessage, useRestResource } from '../useRestResource';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Flush de microtasks (runFetch difiere su arranque un microtask). */
async function flush() {
  await act(async () => {});
}

afterEach(() => {
  vi.useRealTimers();
});

describe('useRestResource — éxito/error/refetch', () => {
  it('éxito inicial: loading true → false, data poblada, lastUpdated numérico', async () => {
    const d = deferred<string>();
    const fetcher = vi.fn(() => d.promise);

    const { result } = renderHook(() => useRestResource(fetcher));

    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();

    await flush();
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => {
      d.resolve('ok');
      await d.promise;
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.data).toBe('ok');
    expect(result.current.error).toBeNull();
    expect(result.current.lastUpdated).toBeTypeOf('number');
  });

  it('error inicial: error en español, data null, loading false', async () => {
    const fetcher = vi.fn(() => Promise.reject(new Error('falló el motor')));

    const { result } = renderHook(() => useRestResource(fetcher));
    await flush();

    expect(result.current.loading).toBe(false);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('falló el motor');
    expect(result.current.lastUpdated).toBeNull();
  });

  it('refetch tras error limpia error y puebla data', async () => {
    const fetcher = vi
      .fn<(signal: AbortSignal) => Promise<string>>()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce('recuperado');

    const { result } = renderHook(() => useRestResource(fetcher));
    await flush();
    expect(result.current.error).toBe('boom');

    await act(async () => {
      await result.current.refetch();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.data).toBe('recuperado');
  });

  it('refetch con error conserva la data stale y NO avanza lastUpdated', async () => {
    const fetcher = vi
      .fn<(signal: AbortSignal) => Promise<string>>()
      .mockResolvedValueOnce('v1')
      .mockRejectedValueOnce(new Error('caída'));

    const { result } = renderHook(() => useRestResource(fetcher));
    await flush();
    expect(result.current.data).toBe('v1');
    const updatedAt = result.current.lastUpdated;

    await act(async () => {
      await result.current.refetch();
    });

    expect(result.current.data).toBe('v1'); // stale conservada (CA-05.4)
    expect(result.current.error).toBe('caída');
    expect(result.current.lastUpdated).toBe(updatedAt);
  });

  it('refetch es silencioso: loading queda false durante el vuelo', async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const fetcher = vi
      .fn<(signal: AbortSignal) => Promise<string>>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const { result } = renderHook(() => useRestResource(fetcher));
    await act(async () => {
      first.resolve('v1');
      await first.promise;
    });

    let refetchDone: Promise<void>;
    act(() => {
      refetchDone = result.current.refetch();
    });
    await flush();

    // En vuelo: data previa visible, sin spinner.
    expect(result.current.loading).toBe(false);
    expect(result.current.data).toBe('v1');

    await act(async () => {
      second.resolve('v2');
      await refetchDone;
    });
    expect(result.current.data).toBe('v2');
  });
});

describe('useRestResource — identidad del fetcher y concurrencia', () => {
  it('cambio de fetcher = nuevo recurso: reset + abort del vuelo anterior', async () => {
    const signals: AbortSignal[] = [];
    const pendingA = deferred<string>();
    const fetcherA = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return pendingA.promise;
    });
    const fetcherB = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return Promise.resolve('data-B');
    });

    const { result, rerender } = renderHook(
      ({ f }: { f: (signal: AbortSignal) => Promise<string> }) => useRestResource(f),
      { initialProps: { f: fetcherA } },
    );
    await flush();
    expect(signals).toHaveLength(1);

    rerender({ f: fetcherB });
    await flush();

    // El vuelo de A quedó abortado; B es el recurso vigente.
    expect(signals[0].aborted).toBe(true);
    expect(result.current.data).toBe('data-B');

    // Resolver A tarde no pisa a B (last-call-wins).
    await act(async () => {
      pendingA.resolve('data-A-tardía');
      await pendingA.promise;
    });
    expect(result.current.data).toBe('data-B');
  });

  it('refetch sobre un vuelo en curso lo aborta: gana el último (last-call-wins)', async () => {
    const signals: AbortSignal[] = [];
    const initial = deferred<string>();
    const slow = deferred<string>();
    const fast = deferred<string>();
    const fetcher = vi
      .fn<(signal: AbortSignal) => Promise<string>>()
      .mockImplementationOnce((signal) => {
        signals.push(signal);
        return initial.promise;
      })
      .mockImplementationOnce((signal) => {
        signals.push(signal);
        return slow.promise;
      })
      .mockImplementationOnce((signal) => {
        signals.push(signal);
        return fast.promise;
      });

    const { result } = renderHook(() => useRestResource(fetcher));
    await act(async () => {
      initial.resolve('v0');
      await initial.promise;
    });

    let p1!: Promise<void>;
    act(() => {
      p1 = result.current.refetch();
    });
    await flush(); // `slow` queda en vuelo
    expect(signals).toHaveLength(2);

    let p2!: Promise<void>;
    act(() => {
      p2 = result.current.refetch();
    });
    await flush(); // `fast` en vuelo; el vuelo de `slow` quedó abortado

    expect(signals[1].aborted).toBe(true);

    await act(async () => {
      fast.resolve('ganador');
      slow.resolve('perdedor-lento');
      await Promise.all([p1, p2]);
    });
    expect(result.current.data).toBe('ganador');
  });
});

describe('useRestResource — polling y cleanup', () => {
  it('polling con intervalMs dispara re-fetches y el unmount no filtra timers', async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn(() => Promise.resolve('tick'));

    const { unmount } = renderHook(() =>
      useRestResource(fetcher, { intervalMs: 1_000 }),
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0); // carga inicial (microtask)
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3_000);
    });
    expect(fetcher).toHaveBeenCalledTimes(4);

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('unmount con promesa pendiente: signal abortado, resolver después no produce setState ni act warnings', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const signals: AbortSignal[] = [];
    const pending = deferred<string>();
    const fetcher = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return pending.promise;
    });

    const { unmount } = renderHook(() => useRestResource(fetcher));
    await flush();

    unmount();
    expect(signals[0].aborted).toBe(true);

    pending.resolve('tarde');
    await pending.promise;
    await Promise.resolve();

    const actWarnings = errorSpy.mock.calls.filter((args) =>
      args.some((a) => typeof a === 'string' && a.includes('not wrapped in act')),
    );
    expect(actWarnings).toHaveLength(0);
    errorSpy.mockRestore();
  });

  it('cancelación axios (CanceledError) se silencia: no es error', async () => {
    const fetcher = vi.fn(() => Promise.reject(new CanceledError('canceled')));

    const { result } = renderHook(() => useRestResource(fetcher));
    await flush();

    expect(result.current.error).toBeNull();
    expect(result.current.data).toBeNull();
  });

  it('enabled: false → el fetcher jamás se llama y no hay loading', async () => {
    const fetcher = vi.fn(() => Promise.resolve('nunca'));

    const { result } = renderHook(() =>
      useRestResource(fetcher, { enabled: false }),
    );
    await flush();

    expect(fetcher).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });
});

describe('toErrorMessage', () => {
  it('axios con response → status; sin response → red; Error → message; resto → desconocido', () => {
    const config = { headers: new AxiosHeaders() };
    const withResponse = new AxiosError('falló', 'ERR_BAD_REQUEST', config, {}, {
      data: {},
      status: 503,
      statusText: 'Service Unavailable',
      headers: {},
      config,
    } as never);
    expect(toErrorMessage(withResponse)).toBe('El servidor respondió 503.');

    const network = new AxiosError('Network Error', 'ERR_NETWORK', config);
    expect(toErrorMessage(network)).toBe('Error de red al conectar con el servidor.');

    expect(toErrorMessage(new Error('mensaje propio'))).toBe('mensaje propio');
    expect(toErrorMessage('algo raro')).toBe('Error desconocido.');
    expect(axios.isCancel(new CanceledError('c'))).toBe(true); // sanity del guard
  });
});
