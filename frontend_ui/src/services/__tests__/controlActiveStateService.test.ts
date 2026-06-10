// HU-05 / CA-01.6 smoke — lado frontend del lazo "401 sobre ruta consumida
// por la UI → auto-logout".
//
// El backend smoke (core_management_api/tests/bdd/hu_05/test_ca_01_6_smoke.py)
// verifica que un token con ``exp`` ya pasado contra
// ``GET /control/active-state/{node_id}`` devuelve 401. Acá verificamos la
// otra mitad: cuando el httpClient real recibe ese 401 con Authorization
// presente, dispara ``authBridge.onUnauthorized`` — el side-effect que el
// SessionProvider de HU-01 conecta a performLogout({reason:'session-expired'})
// → navigate('/login') + flash "Tu sesión expiró".
//
// Diferencia respecto a httpClient.test.ts (CT-01.9):
// ese test ejercita el interceptor sobre una URL genérica (``/protected``).
// Acá lo hacemos sobre la ruta CONCRETA que la UI consume en su flujo
// natural (``/control/active-state/...``), cerrando el lazo end-to-end
// para CA-01.6.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { AxiosAdapter, InternalAxiosRequestConfig } from 'axios';
import { AxiosError } from 'axios';

import { httpClient } from '../httpClient';
import { controlActiveStateService } from '../controlActiveStateService';
import { authBridge } from '../../auth/authBridge';

type AdapterMock = ReturnType<typeof vi.fn> & AxiosAdapter;

function unauthorizedAdapter(): AdapterMock {
  const fn = vi.fn(async (config: InternalAxiosRequestConfig) => {
    const response = {
      data: { detail: 'Token expirado' },
      status: 401,
      statusText: 'Unauthorized',
      headers: {},
      config,
      request: {},
    };
    throw new AxiosError(
      'Request failed with status code 401',
      'ERR_BAD_REQUEST',
      config,
      {},
      response,
    );
  });
  return fn as unknown as AdapterMock;
}

describe('controlActiveStateService — CA-01.6 lap (401 → onUnauthorized)', () => {
  beforeEach(() => {
    authBridge.reset();
    vi.restoreAllMocks();
  });

  it('401 sobre GET /control/active-state con Authorization presente dispara onUnauthorized', async () => {
    httpClient.defaults.adapter = unauthorizedAdapter();
    const onUnauthorized = vi.fn();
    authBridge.register({
      getToken: () => 'expired.token.value',
      onUnauthorized,
    });

    await expect(
      controlActiveStateService.getActiveState('larco_schell'),
    ).rejects.toBeInstanceOf(AxiosError);

    // El lazo CA-01.6: el interceptor del httpClient ve 401 sobre una
    // request que cargaba Authorization, y delega al SessionProvider vía
    // authBridge.onUnauthorized. SessionProvider hace performLogout →
    // navigate('/login') con flash.
    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });

  it('el error original se re-rechaza para que ActiveStrategyView lo trate como fetch failure', async () => {
    httpClient.defaults.adapter = unauthorizedAdapter();
    authBridge.register({
      getToken: () => 'expired.token.value',
      onUnauthorized: vi.fn(),
    });

    await expect(
      controlActiveStateService.getActiveState('larco_schell'),
    ).rejects.toMatchObject({
      response: { status: 401 },
    });
  });
});

// FASE 2 rediseño UI — passthrough del signal de cancelación (firma extendida
// con `opts?: { signal? }` trailing, backward-compatible). Mismo patrón de
// adapter real de este archivo: se captura el config que llega al adapter.
describe('controlActiveStateService — passthrough de signal (FASE 2)', () => {
  beforeEach(() => {
    authBridge.reset();
    vi.restoreAllMocks();
  });

  it('pasa el signal hasta el config de la request', async () => {
    const captured: InternalAxiosRequestConfig[] = [];
    httpClient.defaults.adapter = vi.fn(async (config: InternalAxiosRequestConfig) => {
      captured.push(config);
      return {
        data: {
          node_id: 'larco_schell',
          strategy_mode: 'webster',
          cycle_seconds: 90,
          phase_timings: [],
          decided_at: '2026-06-10T14:00:00',
          activated_at: '2026-06-10T14:00:01',
          activated_by: null,
        },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
        request: {},
      };
    }) as unknown as AdapterMock;
    authBridge.register({ getToken: () => null, onUnauthorized: vi.fn() });

    const controller = new AbortController();
    const result = await controlActiveStateService.getActiveState('larco_schell', {
      signal: controller.signal,
    });

    expect(result.node_id).toBe('larco_schell');
    expect(captured).toHaveLength(1);
    expect(captured[0].signal).toBe(controller.signal);
  });
});
