// CT-01.9: httpClient injects Authorization Bearer when a token is present.
// CT-01.11 (parcial): 401 on authenticated request triggers onUnauthorized;
// 401 on unauthenticated request (login) does NOT.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { AxiosAdapter, InternalAxiosRequestConfig } from 'axios';
import { AxiosError, AxiosHeaders } from 'axios';
import { httpClient } from '../httpClient';
import { authBridge } from '../../auth/authBridge';

type AdapterMock = ReturnType<typeof vi.fn> & AxiosAdapter;

function okAdapter(): AdapterMock {
  const fn = vi.fn(async (config: InternalAxiosRequestConfig) => ({
    data: {},
    status: 200,
    statusText: 'OK',
    headers: {},
    config,
    request: {},
  }));
  return fn as unknown as AdapterMock;
}

function unauthorizedAdapter(): AdapterMock {
  const fn = vi.fn(async (config: InternalAxiosRequestConfig) => {
    const response = {
      data: { detail: 'Token inválido' },
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

describe('httpClient', () => {
  beforeEach(() => {
    authBridge.reset();
    vi.restoreAllMocks();
  });

  it('attaches Authorization Bearer header when authBridge has a token', async () => {
    const adapter = okAdapter();
    httpClient.defaults.adapter = adapter;
    authBridge.register({
      getToken: () => 'abc.def.ghi',
      onUnauthorized: () => {},
    });

    await httpClient.get('/anything');

    const sentConfig = adapter.mock.calls[0][0];
    const headers = sentConfig.headers as AxiosHeaders;
    expect(headers.get('Authorization')).toBe('Bearer abc.def.ghi');
  });

  it('does not attach Authorization header when authBridge has no token', async () => {
    const adapter = okAdapter();
    httpClient.defaults.adapter = adapter;
    authBridge.register({
      getToken: () => null,
      onUnauthorized: () => {},
    });

    await httpClient.get('/anything');

    const sentConfig = adapter.mock.calls[0][0];
    const headers = sentConfig.headers as AxiosHeaders;
    expect(headers.get('Authorization')).toBeFalsy();
  });

  it('calls authBridge.onUnauthorized on 401 for a request that carried Authorization', async () => {
    httpClient.defaults.adapter = unauthorizedAdapter();
    const onUnauthorized = vi.fn();
    authBridge.register({
      getToken: () => 'abc.def.ghi',
      onUnauthorized,
    });

    await expect(httpClient.get('/protected')).rejects.toBeInstanceOf(AxiosError);
    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });

  it('does NOT call authBridge.onUnauthorized on 401 for a request without Authorization (login)', async () => {
    httpClient.defaults.adapter = unauthorizedAdapter();
    const onUnauthorized = vi.fn();
    authBridge.register({
      getToken: () => null,
      onUnauthorized,
    });

    await expect(httpClient.post('/auth/login')).rejects.toBeInstanceOf(AxiosError);
    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('re-rejects the original error so callers can handle it', async () => {
    httpClient.defaults.adapter = unauthorizedAdapter();
    authBridge.register({ getToken: () => 'x', onUnauthorized: () => {} });

    await expect(httpClient.get('/protected')).rejects.toMatchObject({
      response: { status: 401 },
    });
  });
});
