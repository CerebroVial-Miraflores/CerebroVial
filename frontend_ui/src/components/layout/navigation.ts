// FASE 0 rediseño UI: fuente única del mapeo tab ↔ ruta y de los descriptores
// de navegación del shell (labels + iconos). La autorización por rol sigue
// viviendo en auth/roles.ts (TABS_BY_ROLE se reusa tal cual, sin modificarlo).
import type { ComponentType } from 'react';
import {
  Map as MapIcon,
  BarChart3,
  AlertTriangle,
  Settings,
  SlidersHorizontal,
  Network,
  Navigation,
} from 'lucide-react';

import { TABS_BY_ROLE, type Tab } from '../../auth/roles';
import type { Role } from '../../auth/types';

export const PATH_BY_TAB: Record<Tab, string> = {
  dashboard: '/',
  control: '/control',
  alerts: '/alertas',
  congestion: '/congestion',
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

// Migrados de Sidebar.tsx (mismos labels e iconos, mismo orden). El badge '3'
// hardcodeado de Alertas no migra: era un literal sin dato real detrás.
export const NAV_DESCRIPTORS: readonly NavDescriptor[] = [
  { tab: 'dashboard', path: PATH_BY_TAB.dashboard, label: 'Monitoreo', icon: MapIcon },
  { tab: 'analytics', path: PATH_BY_TAB.analytics, label: 'Analítica e IA', icon: BarChart3 },
  { tab: 'alerts', path: PATH_BY_TAB.alerts, label: 'Alertas', icon: AlertTriangle },
  { tab: 'admin', path: PATH_BY_TAB.admin, label: 'Administración', icon: Settings },
  { tab: 'control', path: PATH_BY_TAB.control, label: 'Motor Adaptativo', icon: SlidersHorizontal },
  { tab: 'congestion', path: PATH_BY_TAB.congestion, label: 'Mapa de congestión', icon: Network },
  { tab: 'tomtom', path: PATH_BY_TAB.tomtom, label: 'Tráfico en vivo', icon: Navigation },
];
