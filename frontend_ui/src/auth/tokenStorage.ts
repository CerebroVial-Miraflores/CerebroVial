export const TOKEN_STORAGE_KEY = 'cerebrovial.auth.token';

/**
 * Persists the JWT access token in localStorage.
 *
 * SECURITY: localStorage is readable by any script running on this origin, so
 * a successful XSS would exfiltrate the token. We accept that trade-off in
 * this iteration in exchange for the simpler SPA flow. The migration path if
 * we need stronger protections is: serve the token as an httpOnly cookie from
 * a thin backend proxy and add CSRF protection; only `getToken()` in
 * `authBridge` needs to change in this codebase.
 */
export const tokenStorage = {
  read(): string | null {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  },
  write(token: string): void {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  },
  clear(): void {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  },
};
