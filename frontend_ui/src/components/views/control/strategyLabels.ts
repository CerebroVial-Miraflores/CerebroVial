// HU-05 / CA-05.1 — DHU-006: mapping del modo interno del motor a la etiqueta
// legible que ve el Operador.
//
// PARAMETRIZABLE. Los textos finales están pendientes de validación con el PO
// de Subgerencia de Movilidad — NO hardcodear como definitivos en producto.
//
// Justificación de los textos elegidos:
// - "Optimización por demanda" para webster: el algoritmo de Webster reparte el
//   verde proporcionalmente a la razón flujo/saturación (la demanda). Se evita
//   "tiempos fijos" porque CONTROL.md contrapone Webster AL control de tiempos
//   fijos — el texto invertiría el significado.
// - "Prioridad por congestión" para max_pressure: el algoritmo prioriza el
//   acceso con mayor presión (cola más larga), que es lo que el Operador
//   percibe.
//
// Patrón espejo de ROLE_LABEL_ES en frontend_ui/src/auth/roles.ts (HU-01).
import type { ControlMode } from '../../../services/controlService';

export interface StrategyLabel {
  label: string;
  subtitle: string;
}

export const STRATEGY_LABEL_ES: Record<ControlMode, StrategyLabel> = {
  webster: {
    label: 'Optimización por demanda',
    subtitle:
      'Reparte el verde según la demanda de cada acceso. Activa en tráfico normal.',
  },
  max_pressure: {
    label: 'Prioridad por congestión',
    subtitle:
      'Da paso primero al acceso con más cola. Activa en tráfico saturado.',
  },
};

// Fallback de dominio cuando el backend devuelve un modo que no está en el
// mapping (por ejemplo, una nueva estrategia añadida en el motor antes de que
// el frontend conozca su etiqueta). NUNCA mostramos el código técnico crudo
// — eso violaría DHU-006.
export const STRATEGY_FALLBACK: StrategyLabel = {
  label: 'Estrategia no reconocida',
  subtitle:
    'El motor reportó una estrategia que este panel todavía no sabe describir.',
};

export function labelForStrategy(
  mode: string | null | undefined,
): StrategyLabel {
  if (mode && mode in STRATEGY_LABEL_ES) {
    return STRATEGY_LABEL_ES[mode as ControlMode];
  }
  return STRATEGY_FALLBACK;
}
