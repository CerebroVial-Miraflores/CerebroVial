// HU-01: App establece el tab por defecto según el rol del usuario y
// redirige cualquier intento de fijar un tab ajeno al defaultTabForRole(role)
// (CA-01.1/.2/.3 — defensa de presentación, RNF-INT-07).
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

vi.mock('../components/layout/Sidebar', () => ({
  Sidebar: ({
    activeTab,
    setActiveTab,
  }: {
    activeTab: string;
    setActiveTab: (tab: string) => void;
    setShowThesis: (show: boolean) => void;
  }) => (
    <div data-testid="sidebar">
      <span data-testid="active-tab">{activeTab}</span>
      <button data-testid="goto-dashboard" onClick={() => setActiveTab('dashboard')}>d</button>
      <button data-testid="goto-analytics" onClick={() => setActiveTab('analytics')}>a</button>
      <button data-testid="goto-alerts" onClick={() => setActiveTab('alerts')}>al</button>
      <button data-testid="goto-admin" onClick={() => setActiveTab('admin')}>ad</button>
      <button data-testid="goto-control" onClick={() => setActiveTab('control')}>c</button>
    </div>
  ),
}));

vi.mock('../components/layout/Header', () => ({
  Header: () => <div data-testid="header" />,
}));

vi.mock('../components/views/DashboardView', () => ({
  DashboardView: () => <div data-testid="view-dashboard" />,
}));
vi.mock('../components/views/CameraDetailView', () => ({
  CameraDetailView: () => <div data-testid="view-camera-detail" />,
}));
vi.mock('../components/views/AnalyticsView', () => ({
  AnalyticsView: () => <div data-testid="view-analytics" />,
}));
vi.mock('../components/views/AlertsView', () => ({
  AlertsView: () => <div data-testid="view-alerts" />,
}));
vi.mock('../components/views/AdminView', () => ({
  AdminView: () => <div data-testid="view-admin" />,
}));
vi.mock('../components/views/control/ControlView', () => ({
  ControlView: () => <div data-testid="view-control" />,
}));
vi.mock('../components/modals/ThesisModal', () => ({
  ThesisModal: () => <div data-testid="thesis-modal" />,
}));

vi.mock('../auth/SessionContext', () => ({
  useSession: vi.fn(),
}));

import App from '../App';
import { useSession } from '../auth/SessionContext';

const useSessionMock = useSession as unknown as ReturnType<typeof vi.fn>;

describe('App — vista por defecto por rol y guard de setActiveTab (HU-01)', () => {
  beforeEach(() => useSessionMock.mockReset());

  describe('rol operator', () => {
    beforeEach(() => useSessionMock.mockReturnValue({ role: 'operator', logout: vi.fn() }));

    it('inicia en la vista por defecto del Operador (dashboard)', () => {
      render(<App />);
      expect(screen.getByTestId('active-tab').textContent).toBe('dashboard');
      expect(screen.getByTestId('view-dashboard')).toBeInTheDocument();
      expect(screen.queryByTestId('view-analytics')).toBeNull();
      expect(screen.queryByTestId('view-admin')).toBeNull();
    });

    it('forzar la vista "analytics" redirige al default del Operador', () => {
      render(<App />);
      fireEvent.click(screen.getByTestId('goto-analytics'));
      expect(screen.getByTestId('active-tab').textContent).toBe('dashboard');
      expect(screen.queryByTestId('view-analytics')).toBeNull();
      expect(screen.getByTestId('view-dashboard')).toBeInTheDocument();
    });

    it('forzar la vista "admin" redirige al default del Operador', () => {
      render(<App />);
      fireEvent.click(screen.getByTestId('goto-admin'));
      expect(screen.getByTestId('active-tab').textContent).toBe('dashboard');
      expect(screen.queryByTestId('view-admin')).toBeNull();
    });
  });

  describe('rol manager', () => {
    beforeEach(() => useSessionMock.mockReturnValue({ role: 'manager', logout: vi.fn() }));

    it('inicia en la vista por defecto del Gerente (analytics)', () => {
      render(<App />);
      expect(screen.getByTestId('active-tab').textContent).toBe('analytics');
      expect(screen.getByTestId('view-analytics')).toBeInTheDocument();
      expect(screen.queryByTestId('view-dashboard')).toBeNull();
      expect(screen.queryByTestId('view-admin')).toBeNull();
    });

    it('forzar la vista "dashboard" redirige al default del Gerente', () => {
      render(<App />);
      fireEvent.click(screen.getByTestId('goto-dashboard'));
      expect(screen.getByTestId('active-tab').textContent).toBe('analytics');
      expect(screen.queryByTestId('view-dashboard')).toBeNull();
      expect(screen.getByTestId('view-analytics')).toBeInTheDocument();
    });
  });

  describe('rol admin', () => {
    beforeEach(() => useSessionMock.mockReturnValue({ role: 'admin', logout: vi.fn() }));

    it('inicia en la vista por defecto del Administrador (admin)', () => {
      render(<App />);
      expect(screen.getByTestId('active-tab').textContent).toBe('admin');
      expect(screen.getByTestId('view-admin')).toBeInTheDocument();
      expect(screen.queryByTestId('view-dashboard')).toBeNull();
      expect(screen.queryByTestId('view-control')).toBeNull();
    });

    // HU-05 / DHU-020: admin recupera el tab "control" para acceder al
    // playground interactivo (ControlPlayground). Cambiar a "control" no
    // redirige más al default; muestra el ControlView.
    it('puede cambiar a la vista "control" (HU-05 / DHU-020 — playground del Administrador)', () => {
      render(<App />);
      fireEvent.click(screen.getByTestId('goto-control'));
      expect(screen.getByTestId('active-tab').textContent).toBe('control');
      expect(screen.getByTestId('view-control')).toBeInTheDocument();
      expect(screen.queryByTestId('view-admin')).toBeNull();
    });
  });
});
