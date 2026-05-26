import axios, { AxiosHeaders } from 'axios';
import type { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { authBridge } from '../auth/authBridge';

/**
 * Shared axios instance for every call to core_management_api.
 *
 * - Request interceptor: injects `Authorization: Bearer <token>` whenever the
 *   authBridge has a token. Backend endpoints that do not require auth ignore
 *   the header, so it is safe to send unconditionally.
 * - Response interceptor: when a 401 comes back for a request that carried
 *   Authorization, fires authBridge.onUnauthorized (auto-logout + redirect).
 *   The 401 from /auth/login itself does NOT trigger this because the login
 *   request never carries Authorization.
 */
export const httpClient = axios.create({
  baseURL: import.meta.env.VITE_CORE_API_URL ?? 'http://localhost:8001',
});

httpClient.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = authBridge.getToken();
  if (token) {
    if (!config.headers) {
      config.headers = new AxiosHeaders();
    }
    (config.headers as AxiosHeaders).set('Authorization', `Bearer ${token}`);
  }
  return config;
});

httpClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const status = error.response?.status;
    const headers = error.config?.headers as AxiosHeaders | undefined;
    const hadAuthorization = Boolean(headers?.get?.('Authorization'));
    if (status === 401 && hadAuthorization) {
      authBridge.onUnauthorized();
    }
    return Promise.reject(error);
  },
);
