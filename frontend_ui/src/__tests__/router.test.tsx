import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RouterProvider, createMemoryRouter, type RouteObject } from 'react-router-dom';
import type { ReactNode } from 'react';

import { appRoutes } from '../router';
import { useSession } from '../auth/SessionContext';
import type { Role } from '../auth/types';

// Sucesor de App.test.tsx a nivel router (HU-01: tab default por rol y redirect
// de tab no permitido, ahora como rutas). Se mockea useSession + SessionProvider
// passthrough (appRoutes monta el provider real) y las vistas (acá se valida
// ruteo por rol, no contenido).
vi.mock('../auth/SessionContext', () => ({
  useSession: vi.fn(),
  SessionProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('../auth/LoginView', () => ({
  LoginView: () => <div data-testid="view-login" />,
}));
// FASE 3: el index es CommandView (Centro de Comando); /alertas y /congestion
// son redirects (D3.1a). FASE 4: /camara/:id monta CameraDetailV2.
vi.mock('../components/views/command/CommandView', () => ({
  CommandView: () => <div data-testid="view-command" />,
}));
vi.mock('../components/views/camera/CameraDetailV2', () => ({
  CameraDetailV2: () => <div data-testid="view-camara" />,
}));
vi.mock('../components/views/AnalyticsView', () => ({
  AnalyticsView: () => <div data-testid="view-analytics" />,
}));
vi.mock('../components/views/AdminView', () => ({
  AdminView: () => <div data-testid="view-admin" />,
}));
vi.mock('../components/views/control/ControlView', () => ({
  ControlView: () => <div data-testid="view-control" />,
}));
vi.mock('../tomtom/TomTomView', () => ({
  TomTomView: () => <div data-testid="view-tomtom" />,
}));
vi.mock('../components/modals/ThesisModal', () => ({
  ThesisModal: () => <div data-testid="thesis-modal" />,
}));

const useSessionMock = vi.mocked(useSession);

function sessionFor(role: Role | null): ReturnType<typeof useSession> {
  return {
    token: role ? 'token-de-prueba' : null,
    role,
    userId: role ? 'u-1' : null,
    isAuthenticated: role !== null,
    login: vi.fn(),
    logout: vi.fn(),
  };
}

function renderAt(role: Role | null, path: string) {
  useSessionMock.mockReturnValue(sessionFor(role));
  const router = createMemoryRouter(appRoutes, { initialEntries: [path] });
  render(<RouterProvider router={router} />);
  return router;
}

describe('rutas por rol (HU-01 a nivel URL)', () => {
  it('operator en / ve el Centro de Comando', () => {
    const router = renderAt('operator', '/');
    expect(router.state.location.pathname).toBe('/');
    expect(screen.getByTestId('view-command')).toBeInTheDocument();
  });

  it('operator en /analitica rebota a / (su default)', () => {
    const router = renderAt('operator', '/analitica');
    expect(router.state.location.pathname).toBe('/');
    expect(screen.getByTestId('view-command')).toBeInTheDocument();
    expect(screen.queryByTestId('view-analytics')).not.toBeInTheDocument();
  });

  it('operator en /admin rebota a /', () => {
    const router = renderAt('operator', '/admin');
    expect(router.state.location.pathname).toBe('/');
    expect(screen.queryByTestId('view-admin')).not.toBeInTheDocument();
  });

  it('operator accede a /trafico (track TomTom)', () => {
    const router = renderAt('operator', '/trafico');
    expect(router.state.location.pathname).toBe('/trafico');
    expect(screen.getByTestId('view-tomtom')).toBeInTheDocument();
  });

  it('operator accede al detalle de cámara /camara/:id (CameraDetailV2)', () => {
    const router = renderAt('operator', '/camara/cam_larco_benavides');
    expect(router.state.location.pathname).toBe('/camara/cam_larco_benavides');
    expect(screen.getByTestId('view-camara')).toBeInTheDocument();
  });

  it('manager en / rebota a /analitica (cubre el aterrizaje post-login)', () => {
    const router = renderAt('manager', '/');
    expect(router.state.location.pathname).toBe('/analitica');
    expect(screen.getByTestId('view-analytics')).toBeInTheDocument();
    expect(screen.queryByTestId('view-command')).not.toBeInTheDocument();
  });

  it('admin en / rebota a /admin (su default)', () => {
    const router = renderAt('admin', '/');
    expect(router.state.location.pathname).toBe('/admin');
    expect(screen.getByTestId('view-admin')).toBeInTheDocument();
  });

  it('admin accede a /control (HU-05 / DHU-020)', () => {
    const router = renderAt('admin', '/control');
    expect(router.state.location.pathname).toBe('/control');
    expect(screen.getByTestId('view-control')).toBeInTheDocument();
  });
});

// FASE 3 — D3.1(a): redirects legacy con <Navigate replace>.
describe('redirects legacy (D3.1a)', () => {
  it('/alertas → /?panel=alertas (operator)', () => {
    const router = renderAt('operator', '/alertas');
    expect(router.state.location.pathname).toBe('/');
    expect(router.state.location.search).toBe('?panel=alertas');
    expect(screen.getByTestId('view-command')).toBeInTheDocument();
  });

  it('/congestion → / (operator)', () => {
    const router = renderAt('operator', '/congestion');
    expect(router.state.location.pathname).toBe('/');
    expect(router.state.location.search).toBe('');
    expect(screen.getByTestId('view-command')).toBeInTheDocument();
  });

  it('manager en /alertas termina en /analitica (dos saltos replace, sin loop)', () => {
    const router = renderAt('manager', '/alertas');
    expect(router.state.location.pathname).toBe('/analitica');
    expect(screen.getByTestId('view-analytics')).toBeInTheDocument();
  });

  it('manager en /congestion termina en /analitica', () => {
    const router = renderAt('manager', '/congestion');
    expect(router.state.location.pathname).toBe('/analitica');
  });
});

describe('catch-all y autenticación', () => {
  it('ruta desconocida aterriza en el default del rol (operator)', () => {
    const router = renderAt('operator', '/no-existe');
    expect(router.state.location.pathname).toBe('/');
    expect(screen.getByTestId('view-command')).toBeInTheDocument();
  });

  it('ruta desconocida aterriza en el default del rol (manager)', () => {
    const router = renderAt('manager', '/no-existe');
    expect(router.state.location.pathname).toBe('/analitica');
  });

  it('no autenticado en ruta protegida cae en /login', () => {
    const router = renderAt(null, '/control');
    expect(router.state.location.pathname).toBe('/login');
    expect(screen.getByTestId('view-login')).toBeInTheDocument();
  });
});

describe('ruta /ui-lab (galería del design system)', () => {
  // Bajo vitest import.meta.env.DEV === true → la ruta existe. En el build de
  // producción el ternario se elimina y /ui-lab cae al catch-all (la ausencia
  // en el bundle se verifica con grep sobre dist/ en el cierre de fase).
  it('existe en DEV', () => {
    const collectPaths = (routes: RouteObject[]): string[] =>
      routes.flatMap((route) => [
        ...(route.path ? [route.path] : []),
        ...(route.children ? collectPaths(route.children) : []),
      ]);
    expect(collectPaths(appRoutes)).toContain('ui-lab');
  });
});
