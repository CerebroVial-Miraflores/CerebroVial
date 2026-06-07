// HU-01 / CA-01.1-.3 / RNF-INT-07: mapas y helpers de rol.
import type { Role } from './types';

export type Tab = 'dashboard' | 'analytics' | 'alerts' | 'admin' | 'control' | 'congestion' | 'tomtom';

export const TABS_BY_ROLE: Record<Role, readonly Tab[]> = {
  // HU-22 / CA-22.1: el Operador gana el tab 'congestion' (mapa de congestión
  // en tiempo real). Operator-only: HU-01/CA-01.2 excluye al Gerente de vistas
  // operativas y el require_role(ADMIN) del endpoint es acceso de API, no UI.
  // Track feature/tomtom (EXPERIMENTAL, Fase A): el tab 'tomtom' (tráfico en vivo
  // de TomTom) es operator-only, misma familia que 'congestion' y 'dashboard'.
  operator: ['dashboard', 'control', 'alerts', 'congestion', 'tomtom'],
  manager: ['analytics'],
  // HU-05 / DHU-020: admin recupera el tab 'control' para acceder al
  // playground interactivo (ControlPlayground). El render condicional por
  // rol dentro de ControlView decide qué vista mostrar.
  admin: ['admin', 'control'],
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
