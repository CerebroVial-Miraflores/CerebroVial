import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { useNavigate } from 'react-router-dom';

import { authBridge } from './authBridge';
import { authService } from './authService';
import { decodeJwtPayload } from './decodeJwtPayload';
import { tokenStorage } from './tokenStorage';
import type { LoginCredentials, Role, SessionState } from './types';

interface SessionContextValue extends SessionState {
  login: (creds: LoginCredentials) => Promise<void>;
  logout: () => void;
}

const initialState: SessionState = {
  token: null,
  role: null,
  userId: null,
  isAuthenticated: false,
};

const SessionContext = createContext<SessionContextValue | undefined>(undefined);

function sessionFromToken(token: string): SessionState | null {
  const payload = decodeJwtPayload(token);
  if (!payload) return null;
  if (typeof payload.exp !== 'number') return null;
  if (payload.exp * 1000 <= Date.now()) return null;
  return {
    token,
    role: (payload.role as Role | undefined) ?? null,
    userId: typeof payload.sub === 'string' ? payload.sub : null,
    isAuthenticated: true,
  };
}

interface LogoutOptions {
  reason?: 'session-expired';
}

function hydrateFromStorage(): SessionState {
  const stored = tokenStorage.read();
  if (!stored) return initialState;
  const hydrated = sessionFromToken(stored);
  if (!hydrated) {
    tokenStorage.clear();
    return initialState;
  }
  return hydrated;
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SessionState>(hydrateFromStorage);
  const stateRef = useRef(state);
  const navigate = useNavigate();

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const login = useCallback(async (creds: LoginCredentials): Promise<void> => {
    const tokenResponse = await authService.login(creds);
    const next = sessionFromToken(tokenResponse.access_token);
    if (!next) {
      tokenStorage.clear();
      setState(initialState);
      throw new Error('Backend returned a token the client could not decode');
    }
    tokenStorage.write(tokenResponse.access_token);
    setState(next);
  }, []);

  const performLogout = useCallback(
    (opts: LogoutOptions = {}) => {
      tokenStorage.clear();
      setState(initialState);
      if (opts.reason) {
        navigate('/login', { state: { reason: opts.reason } });
      } else {
        navigate('/login');
      }
    },
    [navigate],
  );

  const performLogoutRef = useRef(performLogout);
  useEffect(() => {
    performLogoutRef.current = performLogout;
  }, [performLogout]);

  useEffect(() => {
    authBridge.register({
      getToken: () => stateRef.current.token,
      onUnauthorized: () => performLogoutRef.current({ reason: 'session-expired' }),
    });
    return () => authBridge.reset();
  }, []);

  const value = useMemo<SessionContextValue>(
    () => ({
      ...state,
      login,
      logout: () => performLogout(),
    }),
    [state, login, performLogout],
  );

  return (
    <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components -- hook colocado al provider
export function useSession(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error('useSession must be used inside a SessionProvider');
  }
  return ctx;
}
