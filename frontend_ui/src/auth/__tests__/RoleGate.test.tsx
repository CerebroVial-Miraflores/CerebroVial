// HU-01: RoleGate envuelve árboles que solo deben renderizar si el rol
// del usuario está en `allowed`. Defensa de presentación (RNF-INT-07 +
// validación dual lado cliente / RNF-SEC-06).
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import { RoleGate } from '../RoleGate';
import { useSession } from '../SessionContext';

vi.mock('../SessionContext', () => ({
  useSession: vi.fn(),
}));

const useSessionMock = useSession as unknown as ReturnType<typeof vi.fn>;

describe('RoleGate', () => {
  beforeEach(() => useSessionMock.mockReset());

  it('renderiza children cuando el rol está en allowed', () => {
    useSessionMock.mockReturnValue({ role: 'admin' });
    render(
      <RoleGate allowed={['admin']}>
        <div data-testid="content">secreto</div>
      </RoleGate>,
    );
    expect(screen.getByTestId('content').textContent).toBe('secreto');
  });

  it('no renderiza children cuando el rol no está en allowed', () => {
    useSessionMock.mockReturnValue({ role: 'operator' });
    render(
      <RoleGate allowed={['admin']}>
        <div data-testid="content">secreto</div>
      </RoleGate>,
    );
    expect(screen.queryByTestId('content')).toBeNull();
  });

  it('renderiza fallback cuando el rol no está en allowed y se provee', () => {
    useSessionMock.mockReturnValue({ role: 'manager' });
    render(
      <RoleGate allowed={['admin']} fallback={<div data-testid="fb">denegado</div>}>
        <div data-testid="content">secreto</div>
      </RoleGate>,
    );
    expect(screen.queryByTestId('content')).toBeNull();
    expect(screen.getByTestId('fb').textContent).toBe('denegado');
  });

  it('no renderiza children cuando el rol es null', () => {
    useSessionMock.mockReturnValue({ role: null });
    render(
      <RoleGate allowed={['admin']}>
        <div data-testid="content">secreto</div>
      </RoleGate>,
    );
    expect(screen.queryByTestId('content')).toBeNull();
  });
});
