// CT-01.8: rutas protegidas sin sesión redirigen a /login preservando from.
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';

import { ProtectedRoute } from '../ProtectedRoute';
import { useSession } from '../SessionContext';

vi.mock('../SessionContext', () => ({
  useSession: vi.fn(),
}));

const useSessionMock = useSession as unknown as ReturnType<typeof vi.fn>;

function LoginStub() {
  const location = useLocation();
  const state = location.state as { from?: { pathname: string } } | null;
  return (
    <div>
      <div data-testid="screen">login</div>
      <div data-testid="from">{state?.from?.pathname ?? ''}</div>
    </div>
  );
}

function ProtectedChild() {
  return <div data-testid="screen">protected</div>;
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/login" element={<LoginStub />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/" element={<ProtectedChild />} />
          <Route path="/dashboard" element={<ProtectedChild />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe('ProtectedRoute', () => {
  it('redirects to /login when there is no session', () => {
    useSessionMock.mockReturnValue({ isAuthenticated: false });
    renderAt('/');
    expect(screen.getByTestId('screen').textContent).toBe('login');
  });

  it('renders the outlet when there is a session', () => {
    useSessionMock.mockReturnValue({ isAuthenticated: true });
    renderAt('/');
    expect(screen.getByTestId('screen').textContent).toBe('protected');
  });

  it('preserves the original path in navigation state for post-login redirect', () => {
    useSessionMock.mockReturnValue({ isAuthenticated: false });
    renderAt('/dashboard');
    expect(screen.getByTestId('from').textContent).toBe('/dashboard');
  });
});
