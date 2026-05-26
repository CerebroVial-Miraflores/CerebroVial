// HU-01 / CA-01.1-.3 / RNF-INT-07: mapas y helpers de rol.
import type { Role } from './types';

export type Tab = 'dashboard' | 'analytics' | 'alerts' | 'admin' | 'control';

export const TABS_BY_ROLE: Record<Role, readonly Tab[]> = {
  operator: ['dashboard', 'control', 'alerts'],
  manager: ['analytics'],
  admin: ['admin'],
};

export const DEFAULT_TAB_BY_ROLE: Record<Role, Tab> = {
  operator: 'dashboard',
  manager: 'analytics',
  admin: 'admin',
};

export const ROLE_LABEL_ES: Record<Role, string> = {
  operator: 'Operador',
  manager: 'Gerente',
  admin: 'Administrador',
};

export function roleAllowsTab(role: Role | null, tab: Tab): boolean {
  if (!role) return false;
  return TABS_BY_ROLE[role].includes(tab);
}

export function defaultTabForRole(role: Role | null): Tab | null {
  if (!role) return null;
  return DEFAULT_TAB_BY_ROLE[role];
}

export function roleLabel(role: Role | null): string {
  if (!role) return '';
  return ROLE_LABEL_ES[role];
}
