// FASE 0 rediseño UI — guard de ruta por rol (HU-01 / RNF-INT-07).
// Equivalente a nivel URL del viejo setActiveTab de App.tsx: una ruta cuyo tab
// no está permitido para el rol redirige al default del rol, nunca a un estado
// intermedio. Cubre también el aterrizaje post-login (LoginView navega a '/' y,
// si el rol no ve dashboard, esta guard lo manda a su ruta default). Adentro,
// RoleGate se mantiene tal cual como defensa presentacional (validación dual
// lado cliente); el enforcement real vive en require_role (backend).
import type { ReactNode } from 'react';
import { Navigate } from 'react-router-dom';

import { useSession } from './SessionContext';
import { RoleGate } from './RoleGate';
import { defaultTabForRole, roleAllowsTab, type Tab } from './roles';
import { pathForTab, rolesForTab } from '../components/layout/navigation';

interface RoleRouteProps {
  tab: Tab;
  children: ReactNode;
}

export function RoleRoute({ tab, children }: RoleRouteProps) {
  const { role } = useSession();
  if (!roleAllowsTab(role, tab)) {
    const fallback = defaultTabForRole(role);
    // role null no llega acá en runtime (ProtectedRoute filtra antes); el
    // fallback a /login cubre el caso degenerado sin romper el render.
    return <Navigate to={fallback ? pathForTab(fallback) : '/login'} replace />;
  }
  return <RoleGate allowed={rolesForTab(tab)}>{children}</RoleGate>;
}

// Catch-all '*' del router: cualquier ruta desconocida aterriza en el tab
// default del rol en sesión.
export function DefaultTabRedirect() {
  const { role } = useSession();
  const fallback = defaultTabForRole(role);
  return <Navigate to={fallback ? pathForTab(fallback) : '/login'} replace />;
}
