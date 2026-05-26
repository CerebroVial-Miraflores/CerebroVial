// HU-01 / RNF-INT-07 + RNF-SEC-06 (validación dual lado cliente):
// envuelve un árbol de UI y lo renderiza únicamente cuando el rol del
// usuario en sesión está en `allowed`. Si no, renderiza `fallback` (null
// por defecto). Defensa de presentación; el enforcement real vive en
// require_role (backend).
import type { ReactNode } from 'react';

import { useSession } from './SessionContext';
import type { Role } from './types';

interface RoleGateProps {
  allowed: readonly Role[];
  children: ReactNode;
  fallback?: ReactNode;
}

export function RoleGate({ allowed, children, fallback = null }: RoleGateProps) {
  const { role } = useSession();
  if (role && allowed.includes(role)) {
    return <>{children}</>;
  }
  return <>{fallback}</>;
}
