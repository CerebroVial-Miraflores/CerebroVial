// HU-01: mapas y helpers de rol (CA-01.1/.2/.3).
import { describe, it, expect } from 'vitest';

import {
  TABS_BY_ROLE,
  DEFAULT_TAB_BY_ROLE,
  ROLE_LABEL_ES,
  roleAllowsTab,
  defaultTabForRole,
  roleLabel,
} from '../roles';

describe('TABS_BY_ROLE', () => {
  it('operator tiene dashboard, control y alerts', () => {
    expect([...TABS_BY_ROLE.operator]).toEqual(['dashboard', 'control', 'alerts']);
  });
  it('manager tiene solo analytics', () => {
    expect([...TABS_BY_ROLE.manager]).toEqual(['analytics']);
  });
  it('admin tiene solo admin', () => {
    expect([...TABS_BY_ROLE.admin]).toEqual(['admin']);
  });
});

describe('DEFAULT_TAB_BY_ROLE', () => {
  it('operator default = dashboard', () => {
    expect(DEFAULT_TAB_BY_ROLE.operator).toBe('dashboard');
  });
  it('manager default = analytics', () => {
    expect(DEFAULT_TAB_BY_ROLE.manager).toBe('analytics');
  });
  it('admin default = admin', () => {
    expect(DEFAULT_TAB_BY_ROLE.admin).toBe('admin');
  });
});

describe('ROLE_LABEL_ES', () => {
  it('mapea cada rol a su label en español', () => {
    expect(ROLE_LABEL_ES.operator).toBe('Operador');
    expect(ROLE_LABEL_ES.manager).toBe('Gerente');
    expect(ROLE_LABEL_ES.admin).toBe('Administrador');
  });
});

describe('roleAllowsTab', () => {
  it('retorna true cuando la pestaña pertenece al rol', () => {
    expect(roleAllowsTab('operator', 'dashboard')).toBe(true);
    expect(roleAllowsTab('operator', 'control')).toBe(true);
    expect(roleAllowsTab('operator', 'alerts')).toBe(true);
    expect(roleAllowsTab('manager', 'analytics')).toBe(true);
    expect(roleAllowsTab('admin', 'admin')).toBe(true);
  });
  it('retorna false cuando la pestaña no pertenece al rol', () => {
    expect(roleAllowsTab('operator', 'analytics')).toBe(false);
    expect(roleAllowsTab('operator', 'admin')).toBe(false);
    expect(roleAllowsTab('manager', 'dashboard')).toBe(false);
    expect(roleAllowsTab('manager', 'control')).toBe(false);
    expect(roleAllowsTab('admin', 'control')).toBe(false);
    expect(roleAllowsTab('admin', 'analytics')).toBe(false);
  });
  it('retorna false cuando el rol es null', () => {
    expect(roleAllowsTab(null, 'dashboard')).toBe(false);
  });
});

describe('defaultTabForRole', () => {
  it('retorna el tab por defecto de cada rol', () => {
    expect(defaultTabForRole('operator')).toBe('dashboard');
    expect(defaultTabForRole('manager')).toBe('analytics');
    expect(defaultTabForRole('admin')).toBe('admin');
  });
  it('retorna null cuando el rol es null', () => {
    expect(defaultTabForRole(null)).toBeNull();
  });
});

describe('roleLabel', () => {
  it('retorna el label en español de cada rol', () => {
    expect(roleLabel('operator')).toBe('Operador');
    expect(roleLabel('manager')).toBe('Gerente');
    expect(roleLabel('admin')).toBe('Administrador');
  });
  it('retorna cadena vacía cuando el rol es null', () => {
    expect(roleLabel(null)).toBe('');
  });
});
