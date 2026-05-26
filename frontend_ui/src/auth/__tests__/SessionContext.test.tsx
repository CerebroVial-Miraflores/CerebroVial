// CT-01.6: login OK persists token, decodes role, updates state.
// CT-01.10: logout clears storage, resets state, redirects to /login.
// CT-01.11 (parcial): authBridge.onUnauthorized triggers logout with session-expired reason.
// CT-01.13: token expired/malformed at boot is discarded; app starts unauthenticated.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { SessionProvider, useSession } from '../SessionContext';
import { authBridge } from '../authBridge';
import { tokenStorage, TOKEN_STORAGE_KEY } from '../tokenStorage';
import { authService } from '../authService';

vi.mock('../authService', () => ({
  authService: { login: vi.fn() },
  InvalidCredentialsError: class extends Error {},
}));

const navigateMock = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>(
    'react-router-dom',
  );
  return { ...actual, useNavigate: () => navigateMock };
});

const loginMock = authService.login as unknown as ReturnType<typeof vi.fn>;

function base64url(obj: unknown): string {
  return btoa(JSON.stringify(obj))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

function makeToken(payload: Record<string, unknown>): string {
  const header = base64url({ alg: 'HS256', typ: 'JWT' });
  return `${header}.${base64url(payload)}.sig`;
}

function freshToken(role: 'operator' | 'manager' | 'admin' = 'operator'): string {
  return makeToken({
    sub: 'user-uuid-1',
    role,
    exp: Math.floor(Date.now() / 1000) + 3600,
  });
}

function expiredToken(): string {
  return makeToken({
    sub: 'user-uuid-1',
    role: 'operator',
    exp: Math.floor(Date.now() / 1000) - 3600,
  });
}

function TestConsumer() {
  const session = useSession();
  return (
    <div>
      <div data-testid="isAuthenticated">{String(session.isAuthenticated)}</div>
      <div data-testid="token">{session.token ?? ''}</div>
      <div data-testid="role">{session.role ?? ''}</div>
      <div data-testid="userId">{session.userId ?? ''}</div>
      <button
        onClick={() =>
          session.login({ email: 'alice@cv.pe', password: 's3cret' })
        }
      >
        login
      </button>
      <button onClick={() => session.logout()}>logout</button>
    </div>
  );
}

function renderWithProvider() {
  return render(
    <MemoryRouter>
      <SessionProvider>
        <TestConsumer />
      </SessionProvider>
    </MemoryRouter>,
  );
}

describe('SessionProvider', () => {
  beforeEach(() => {
    localStorage.clear();
    loginMock.mockReset();
    navigateMock.mockReset();
    authBridge.reset();
  });

  it('starts unauthenticated when no token is stored', () => {
    renderWithProvider();
    expect(screen.getByTestId('isAuthenticated').textContent).toBe('false');
    expect(screen.getByTestId('token').textContent).toBe('');
    expect(screen.getByTestId('role').textContent).toBe('');
  });

  it('hydrates from localStorage on mount when token is valid and not expired', () => {
    const token = freshToken('manager');
    tokenStorage.write(token);

    renderWithProvider();

    expect(screen.getByTestId('isAuthenticated').textContent).toBe('true');
    expect(screen.getByTestId('token').textContent).toBe(token);
    expect(screen.getByTestId('role').textContent).toBe('manager');
    expect(screen.getByTestId('userId').textContent).toBe('user-uuid-1');
  });

  it('clears storage and stays unauthenticated when stored token is expired', () => {
    tokenStorage.write(expiredToken());

    renderWithProvider();

    expect(screen.getByTestId('isAuthenticated').textContent).toBe('false');
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
  });

  it('clears storage and stays unauthenticated when stored token is malformed', () => {
    tokenStorage.write('not.a.valid.jwt');

    renderWithProvider();

    expect(screen.getByTestId('isAuthenticated').textContent).toBe('false');
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
  });

  it('login persists token, decodes role and updates state', async () => {
    const token = freshToken('admin');
    loginMock.mockResolvedValue({
      access_token: token,
      token_type: 'bearer',
      expires_in: 28800,
    });

    renderWithProvider();
    await act(async () => {
      fireEvent.click(screen.getByText('login'));
    });

    expect(loginMock).toHaveBeenCalledWith({
      email: 'alice@cv.pe',
      password: 's3cret',
    });
    expect(screen.getByTestId('isAuthenticated').textContent).toBe('true');
    expect(screen.getByTestId('role').textContent).toBe('admin');
    expect(screen.getByTestId('userId').textContent).toBe('user-uuid-1');
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBe(token);
  });

  it('logout clears storage, resets state and navigates to /login', async () => {
    tokenStorage.write(freshToken('operator'));
    renderWithProvider();
    expect(screen.getByTestId('isAuthenticated').textContent).toBe('true');

    fireEvent.click(screen.getByText('logout'));

    expect(screen.getByTestId('isAuthenticated').textContent).toBe('false');
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
    expect(navigateMock).toHaveBeenCalledWith('/login');
  });

  it('registers authBridge.getToken returning the current token', () => {
    const token = freshToken();
    tokenStorage.write(token);

    renderWithProvider();

    expect(authBridge.getToken()).toBe(token);
  });

  it('authBridge.onUnauthorized triggers logout and navigates with session-expired reason', () => {
    tokenStorage.write(freshToken());
    renderWithProvider();
    expect(screen.getByTestId('isAuthenticated').textContent).toBe('true');

    act(() => {
      authBridge.onUnauthorized();
    });

    expect(screen.getByTestId('isAuthenticated').textContent).toBe('false');
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
    expect(navigateMock).toHaveBeenCalledWith('/login', {
      state: { reason: 'session-expired' },
    });
  });
});
