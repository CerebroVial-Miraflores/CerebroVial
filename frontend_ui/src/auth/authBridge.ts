/**
 * Singleton bridge between the React session state and the axios httpClient.
 *
 * The axios interceptors live outside the React tree, so they cannot use
 * hooks to read the current token or trigger a logout. SessionProvider
 * registers two closures here on mount; httpClient delegates to them.
 *
 * Keeping the bridge in a separate module means the interceptor stays
 * trivially testable (no localStorage, no React) and gives us a single seam
 * to swap if the storage strategy changes (e.g. httpOnly cookie).
 */
export interface AuthBridgeCallbacks {
  getToken: () => string | null;
  onUnauthorized: () => void;
}

const defaults: AuthBridgeCallbacks = {
  getToken: () => null,
  onUnauthorized: () => {},
};

let current: AuthBridgeCallbacks = defaults;

export const authBridge = {
  register(callbacks: AuthBridgeCallbacks): void {
    current = callbacks;
  },
  reset(): void {
    current = defaults;
  },
  getToken(): string | null {
    return current.getToken();
  },
  onUnauthorized(): void {
    current.onUnauthorized();
  },
};
