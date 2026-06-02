// HU-01: Sidebar oculta enlaces a vistas no accesibles según el rol
// (CA-01.1/.2/.3 — RNF-INT-07). El footer muestra el label del rol en español.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import { Sidebar } from '../Sidebar';
import { useSession } from '../../../auth/SessionContext';
import type { Role } from '../../../auth/types';

vi.mock('../../../auth/SessionContext', () => ({
  useSession: vi.fn(),
}));

const useSessionMock = useSession as unknown as ReturnType<typeof vi.fn>;

function renderSidebar(role: Role | null) {
  useSessionMock.mockReturnValue({ role, logout: vi.fn() });
  return render(
    <Sidebar
      activeTab={role === 'admin' ? 'admin' : role === 'manager' ? 'analytics' : 'dashboard'}
      setActiveTab={vi.fn()}
      setShowThesis={vi.fn()}
    />,
  );
}

describe('Sidebar — segregación de tabs por rol (CA-01.1/.2/.3, RNF-INT-07)', () => {
  beforeEach(() => useSessionMock.mockReset());

  describe('rol operator', () => {
    // HU-22 / CA-22.1: el Operador también ve "Mapa de congestión".
    it('muestra Monitoreo, Motor Adaptativo, Alertas y Mapa de congestión; oculta Analítica y Administración', () => {
      renderSidebar('operator');
      expect(screen.getByRole('button', { name: /monitoreo/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /motor adaptativo/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /alertas/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /mapa de congesti[oó]n/i })).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /anal[ií]tica/i })).toBeNull();
      expect(screen.queryByRole('button', { name: /administraci[oó]n/i })).toBeNull();
    });
    it('el footer muestra el label "Operador"', () => {
      renderSidebar('operator');
      expect(screen.getByText('Operador')).toBeInTheDocument();
    });
  });

  describe('rol manager', () => {
    it('muestra solo Analítica y oculta Monitoreo, Motor Adaptativo, Alertas y Administración', () => {
      renderSidebar('manager');
      expect(screen.getByRole('button', { name: /anal[ií]tica/i })).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /monitoreo/i })).toBeNull();
      expect(screen.queryByRole('button', { name: /motor adaptativo/i })).toBeNull();
      expect(screen.queryByRole('button', { name: /alertas/i })).toBeNull();
      expect(screen.queryByRole('button', { name: /administraci[oó]n/i })).toBeNull();
      // HU-22: el Gerente NO ve el mapa de congestión (operator-only).
      expect(screen.queryByRole('button', { name: /mapa de congesti[oó]n/i })).toBeNull();
    });
    it('el footer muestra el label "Gerente"', () => {
      renderSidebar('manager');
      expect(screen.getByText('Gerente')).toBeInTheDocument();
    });
  });

  describe('rol admin', () => {
    // HU-05 / DHU-020: admin gana "Motor Adaptativo" para acceder al
    // playground interactivo (ControlPlayground). Sigue oculto Monitoreo,
    // Alertas y Analítica.
    it('muestra Administración y Motor Adaptativo; oculta Monitoreo, Alertas y Analítica', () => {
      renderSidebar('admin');
      expect(screen.getByRole('button', { name: /administraci[oó]n/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /motor adaptativo/i })).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /monitoreo/i })).toBeNull();
      expect(screen.queryByRole('button', { name: /alertas/i })).toBeNull();
      expect(screen.queryByRole('button', { name: /anal[ií]tica/i })).toBeNull();
      // HU-22: el Administrador NO ve el mapa de congestión (operator-only).
      expect(screen.queryByRole('button', { name: /mapa de congesti[oó]n/i })).toBeNull();
    });
    it('el footer muestra el label "Administrador"', () => {
      renderSidebar('admin');
      expect(screen.getByText('Administrador')).toBeInTheDocument();
    });
  });
});
