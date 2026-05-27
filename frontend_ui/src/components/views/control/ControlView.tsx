// HU-05 / DHU-020 — Switch por rol del tab "control".
//
// El tab unificado se separa en dos vistas según el rol activo:
// - Operador (RNF-INT-07): vista pasiva ActiveStrategyView (read-only, sin
//   inputs, sin botones que disparen /recommend ni __internal/activate).
// - Administrador: playground interactivo ControlPlayground, preservado
//   intacto desde antes de HU-05 (DHU-020: "el playground actual se preserva
//   como herramienta de Administrador").
//
// El gate primario (qué roles ven el tab) vive en TABS_BY_ROLE de roles.ts y
// en el RoleGate del App.tsx. Acá solo decidimos QUÉ renderizar para cada
// rol que ya pasó el gate.
import { useSession } from '../../../auth/SessionContext';
import { ActiveStrategyView } from './ActiveStrategyView';
import { ControlPlayground } from './ControlPlayground';

export const ControlView = () => {
  const { role } = useSession();
  if (role === 'admin') {
    return <ControlPlayground />;
  }
  // Operador (y cualquier otro rol al que se conceda 'control' en el futuro)
  // ve la vista pasiva. Defensa en profundidad: si el rol no debería estar
  // acá, ya falló el RoleGate del App.tsx; igual mostramos la versión
  // read-only por defecto.
  return <ActiveStrategyView />;
};
