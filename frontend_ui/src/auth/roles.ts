// HU-01 / CA-01.1-.3 / RNF-INT-07: mapas y helpers de rol.
import type { Role } from './types';

// FASE 3 rediseño UI: mueren los tabs 'alerts' y 'congestion' — el Centro de
// Comando ("/", tab dashboard) fusiona dashboard + congestión + alertas.
// /alertas y /congestion redirigen (D3.1a) en el router.
export type Tab = 'dashboard' | 'analytics' | 'admin' | 'control' | 'tomtom';

export const TABS_BY_ROLE: Record<Role, readonly Tab[]> = {
  // FASE 3: operador con 3 tabs — Comando ('dashboard', "/"), Motor
  // ('control') y Tráfico ('tomtom', track EXPERIMENTAL Fase A). La exclusión
  // del Gerente de vistas operativas (HU-01/CA-01.2) sigue vigente.
  operator: ['dashboard', 'control', 'tomtom'],
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
