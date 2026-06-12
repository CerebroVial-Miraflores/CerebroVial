import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';

import { AppShell } from '../AppShell';
import { useSession } from '../../../auth/SessionContext';
import type { Role } from '../../../auth/types';

vi.mock('../../../auth/SessionContext', () => ({
  useSession: vi.fn(),
}));

vi.mock('../../modals/ThesisModal', () => ({
  ThesisModal: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="thesis-modal">
      <button type="button" onClick={onClose}>
        cerrar
      </button>
    </div>
  ),
}));

const useSessionMock = vi.mocked(useSession);

// Shell aislado: router mínimo con AppShell como layout y un hijo dummy, para
// no arrastrar vistas reales (leaflet/SSE). El ruteo por rol se prueba aparte
// en router.test.tsx.
function renderShell(role: Role, initialPath = '/') {
  const logout = vi.fn();
  useSessionMock.mockReturnValue({
    token: 'token-de-prueba',
    role,
    userId: 'u-1',
    isAuthenticated: true,
    login: vi.fn(),
    logout,
  });
  const router = createMemoryRouter(
    [
      {
        path: '/',
        element: <AppShell />,
        children: [
          { index: true, element: <div data-testid="outlet-contenido" /> },
          { path: '*', element: <div data-testid="outlet-contenido" /> },
        ],
      },
    ],
    { initialEntries: [initialPath] },
  );
  render(<RouterProvider router={router} />);
  return { router, logout };
}

function rail() {
  return screen.getByRole('navigation', { name: /navegación principal/i });
}

function bottomNav() {
  return screen.getByRole('navigation', { name: /navegación inferior/i });
}

describe('AppShell — render por rol (hereda los asserts de Sidebar.test)', () => {
  // FASE 3: operator pasa de 5 tabs a 3 — Comando ("/", ex Monitoreo, ahora el
  // Centro de Comando), Motor Adaptativo y Tráfico en vivo. Alertas y Mapa de
  // congestión murieron como tabs (el comando los fusiona; redirects D3.1a).
  it('operator ve sus 3 tabs y no ve Analítica ni Administración', () => {
    renderShell('operator');
    const nav = rail();
    for (const label of ['Comando', 'Motor Adaptativo', 'Tráfico en vivo']) {
      expect(within(nav).getByRole('button', { name: label })).toBeInTheDocument();
    }
    expect(within(nav).queryByRole('button', { name: 'Alertas' })).not.toBeInTheDocument();
    expect(within(nav).queryByRole('button', { name: 'Mapa de congestión' })).not.toBeInTheDocument();
    expect(within(nav).queryByRole('button', { name: 'Analítica e IA' })).not.toBeInTheDocument();
    expect(within(nav).queryByRole('button', { name: 'Administración' })).not.toBeInTheDocument();
  });

  it('manager solo ve Analítica e IA', () => {
    renderShell('manager');
    const nav = rail();
    expect(within(nav).getByRole('button', { name: 'Analítica e IA' })).toBeInTheDocument();
    expect(within(nav).queryByRole('button', { name: 'Comando' })).not.toBeInTheDocument();
    expect(within(nav).queryByRole('button', { name: 'Administración' })).not.toBeInTheDocument();
  });

  it('admin ve Administración y Motor Adaptativo, no Comando', () => {
    renderShell('admin');
    const nav = rail();
    expect(within(nav).getByRole('button', { name: 'Administración' })).toBeInTheDocument();
    expect(within(nav).getByRole('button', { name: 'Motor Adaptativo' })).toBeInTheDocument();
    expect(within(nav).queryByRole('button', { name: 'Comando' })).not.toBeInTheDocument();
  });

  it('la bottom-nav lleva los mismos tabs del rol', () => {
    renderShell('admin');
    const nav = bottomNav();
    expect(within(nav).getByRole('button', { name: /administración/i })).toBeInTheDocument();
    expect(within(nav).getByRole('button', { name: /motor adaptativo/i })).toBeInTheDocument();
    expect(within(nav).queryByRole('button', { name: /comando/i })).not.toBeInTheDocument();
  });

  it('muestra la pill de sesión con el rol en español', () => {
    renderShell('manager');
    expect(screen.getByText('Gerente')).toBeInTheDocument();
  });

  it('muestra el reloj con formato es-PE', () => {
    renderShell('operator');
    expect(screen.getByText(/\d{2}:\d{2}:\d{2}/)).toBeInTheDocument();
  });
});

describe('AppShell — contrato responsive', () => {
  // jsdom no computa CSS, así que la visibilidad real a 390px no es testeable acá;
  // este test fija el CONTRATO de clases responsive (rail solo ≥md, bottom-nav <md).
  // La verificación visual a 390px es parte del cierre manual de la fase.
  it('el rail existe solo en ≥md (hidden + md:flex)', () => {
    renderShell('operator');
    expect(rail()).toHaveClass('hidden', 'md:flex');
  });

  it('la bottom-nav existe solo en <md (md:hidden)', () => {
    renderShell('operator');
    expect(bottomNav()).toHaveClass('md:hidden');
  });
});

describe('AppShell — navegación e interacciones', () => {
  it('click en un tab del rail navega a su ruta (navigate, no setState)', () => {
    const { router } = renderShell('operator');
    fireEvent.click(within(rail()).getByRole('button', { name: 'Tráfico en vivo' }));
    expect(router.state.location.pathname).toBe('/trafico');
  });

  it('click en un tab de la bottom-nav navega a su ruta', () => {
    const { router } = renderShell('admin');
    fireEvent.click(within(bottomNav()).getByRole('button', { name: /motor adaptativo/i }));
    expect(router.state.location.pathname).toBe('/control');
  });

  it('marca el tab activo desde la URL (aria-current)', () => {
    renderShell('operator', '/trafico');
    expect(within(rail()).getByRole('button', { name: 'Tráfico en vivo' })).toHaveAttribute(
      'aria-current',
      'page',
    );
    expect(within(rail()).getByRole('button', { name: 'Comando' })).not.toHaveAttribute(
      'aria-current',
    );
  });

  // FASE 3: el puente /camara/:id pertenece al flujo del comando — el rail
  // mantiene Comando activo (alias camara→dashboard en tabForPath).
  it('en /camara/:id el tab activo es Comando (puente temporal F4)', () => {
    renderShell('operator', '/camara/cam_larco_benavides');
    expect(within(rail()).getByRole('button', { name: 'Comando' })).toHaveAttribute(
      'aria-current',
      'page',
    );
  });

  it('el acceso a la Ficha de Tesis migró al shell y abre el modal (zona protegida)', () => {
    renderShell('operator');
    const triggers = screen.getAllByRole('button', { name: /ficha de tesis/i });
    // Dos accesos: rail (≥md) y topbar (<md). Nunca se pierde.
    expect(triggers.length).toBeGreaterThanOrEqual(2);
    fireEvent.click(triggers[0]);
    expect(screen.getByTestId('thesis-modal')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'cerrar' }));
    expect(screen.queryByTestId('thesis-modal')).not.toBeInTheDocument();
  });

  it('el botón de logout cierra la sesión', () => {
    const { logout } = renderShell('operator');
    fireEvent.click(screen.getByRole('button', { name: /cerrar sesión/i }));
    expect(logout).toHaveBeenCalledTimes(1);
  });
});
