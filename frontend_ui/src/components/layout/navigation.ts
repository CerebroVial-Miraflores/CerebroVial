// FASE 0 rediseño UI: fuente única del mapeo tab ↔ ruta y de los descriptores
// de navegación del shell (labels + iconos). La autorización por rol sigue
// viviendo en auth/roles.ts (TABS_BY_ROLE se reusa tal cual, sin modificarlo).
import type { ComponentType } from 'react';
import {
  LayoutGrid,
  BarChart3,
  Settings,
  SlidersHorizontal,
  Navigation,
} from 'lucide-react';

import { TABS_BY_ROLE, type Tab } from '../../auth/roles';
import type { Role } from '../../auth/types';

// FASE 3: /alertas y /congestion ya no son tabs (redirects D3.1a en el router).
export const PATH_BY_TAB: Record<Tab, string> = {
  dashboard: '/',
  control: '/control',
  tomtom: '/trafico',
  analytics: '/analitica',
  admin: '/admin',
};

export function pathForTab(tab: Tab): string {
  return PATH_BY_TAB[tab];
}

// Resuelve el tab desde el primer segmento del pathname ('/' → dashboard).
// Devuelve null para rutas que no mapean a ningún tab (catch-all del router).
export function tabForPath(pathname: string): Tab | null {
  const segment = pathname.split('/').filter(Boolean)[0];
  if (!segment) return 'dashboard';
  // FASE 3: el puente /camara/:id pertenece al flujo del comando — sin este
  // alias el rail quedaría sin tab activo. El puente muere en Fase 4.
  if (segment === 'camara') return 'dashboard';
  const entry = (Object.entries(PATH_BY_TAB) as [Tab, string][]).find(
    ([, path]) => path === `/${segment}`,
  );
  return entry?.[0] ?? null;
}

// Derivado de TABS_BY_ROLE para alimentar RoleGate sin duplicar la matriz rol×tab.
export function rolesForTab(tab: Tab): readonly Role[] {
  return (Object.keys(TABS_BY_ROLE) as Role[]).filter((role) =>
    TABS_BY_ROLE[role].includes(tab),
  );
}

export interface NavDescriptor {
  tab: Tab;
  path: string;
  label: string;
  icon: ComponentType<{ size?: number; className?: string }>;
}

// FASE 3: 'dashboard' pasa a ser el Centro de Comando (label "Comando", icono
// grid 2×2 como el rail del prototipo). Las filas de Alertas y Mapa de
// congestión mueren con sus tabs — el conteo de alertas mock NO migra al rail
// (decisión D3: acoplar el feed demo al shell sería coupling sin dato real).
export const NAV_DESCRIPTORS: readonly NavDescriptor[] = [
  { tab: 'dashboard', path: PATH_BY_TAB.dashboard, label: 'Comando', icon: LayoutGrid },
  { tab: 'analytics', path: PATH_BY_TAB.analytics, label: 'Analítica e IA', icon: BarChart3 },
  { tab: 'admin', path: PATH_BY_TAB.admin, label: 'Administración', icon: Settings },
  { tab: 'control', path: PATH_BY_TAB.control, label: 'Motor Adaptativo', icon: SlidersHorizontal },
  { tab: 'tomtom', path: PATH_BY_TAB.tomtom, label: 'Tráfico en vivo', icon: Navigation },
];
