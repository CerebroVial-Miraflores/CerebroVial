// CT-01.6: login OK redirige al destino.
// CT-01.7: 401 muestra "Credenciales incorrectas" y no navega.
// CT-01.11 (flash): mensaje "Tu sesión expiró" si la nav state lo trae.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { LoginView } from '../LoginView';
import { useSession } from '../SessionContext';
import { InvalidCredentialsError } from '../authService';

vi.mock('../SessionContext', () => ({
  useSession: vi.fn(),
}));

const navigateMock = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>(
    'react-router-dom',
  );
  return { ...actual, useNavigate: () => navigateMock };
});

const useSessionMock = useSession as unknown as ReturnType<typeof vi.fn>;
const loginFn = vi.fn();

function renderLogin(initialState?: unknown, initialPath = '/login') {
  return render(
    <MemoryRouter initialEntries={[{ pathname: initialPath, state: initialState }]}>
      <LoginView />
    </MemoryRouter>,
  );
}

describe('LoginView', () => {
  beforeEach(() => {
    loginFn.mockReset();
    navigateMock.mockReset();
    useSessionMock.mockReturnValue({ login: loginFn });
  });

  it('renders email and password inputs and a submit button', () => {
    renderLogin();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/contraseña/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /ingresar/i })).toBeInTheDocument();
  });

  it('disables submit when email or password is empty', () => {
    renderLogin();
    const button = screen.getByRole('button', { name: /ingresar/i });
    expect(button).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'alice@cv.pe' },
    });
    expect(button).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/contraseña/i), {
      target: { value: 's3cret' },
    });
    expect(button).not.toBeDisabled();
  });

  it('submits credentials through useSession.login and navigates to / on success', async () => {
    loginFn.mockResolvedValue(undefined);
    renderLogin();

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'alice@cv.pe' },
    });
    fireEvent.change(screen.getByLabelText(/contraseña/i), {
      target: { value: 's3cret' },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /ingresar/i }));
    });

    expect(loginFn).toHaveBeenCalledWith({
      email: 'alice@cv.pe',
      password: 's3cret',
    });
    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith('/', { replace: true });
    });
  });

  it('redirects to location.state.from after a successful login', async () => {
    loginFn.mockResolvedValue(undefined);
    renderLogin({ from: { pathname: '/dashboard' } });

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'a@b.c' },
    });
    fireEvent.change(screen.getByLabelText(/contraseña/i), {
      target: { value: 'x' },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /ingresar/i }));
    });

    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith('/dashboard', { replace: true });
    });
  });

  it('shows "Credenciales incorrectas" when login throws InvalidCredentialsError and does not navigate', async () => {
    loginFn.mockRejectedValue(new InvalidCredentialsError());
    renderLogin();

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'wrong@cv.pe' },
    });
    fireEvent.change(screen.getByLabelText(/contraseña/i), {
      target: { value: 'wrong' },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /ingresar/i }));
    });

    expect(await screen.findByText('Credenciales incorrectas')).toBeInTheDocument();
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it('shows the session-expired flash when navigation state reason is "session-expired"', () => {
    renderLogin({ reason: 'session-expired' });
    expect(screen.getByText(/tu sesión expiró/i)).toBeInTheDocument();
  });

  it('clears the credentials error after the user edits the email field', async () => {
    loginFn.mockRejectedValue(new InvalidCredentialsError());
    renderLogin();

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'a@b.c' } });
    fireEvent.change(screen.getByLabelText(/contraseña/i), { target: { value: 'x' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /ingresar/i }));
    });
    expect(await screen.findByText('Credenciales incorrectas')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'a@b.co' } });
    expect(screen.queryByText('Credenciales incorrectas')).not.toBeInTheDocument();
  });
});
