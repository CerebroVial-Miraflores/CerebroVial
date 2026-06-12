// FASE 3 rediseño UI — TODO lo simulado del Centro de Comando vive acá.
//
// Política de paridad: estas estructuras alimentan superficies marcadas con
// DemoBadge (panel de alertas, KPI demora, recomendación IA e historial del
// drawer, series mock del KPI modal). Nada de este archivo toca capas reales.
// Valores calcados del prototipo design/cerebrovial-ui-concept.html.
import type { AlertSeverity } from '../../ui/AlertAccordion';

export type AlertActionKind = 'see' | 'predview' | 'apply' | 'restart' | 'generic';

export interface AlertAction {
  label: string;
  kind: AlertActionKind;
  primary?: boolean;
}

export interface CommandAlert {
  id: string;
  severity: AlertSeverity;
  title: string;
  place: string;
  timeAgo: string;
  /** Confianza % (solo informativa del mock). */
  conf?: number;
  /** Stats del cuerpo expandible (Cola/Velocidad/Demora del prototipo). */
  mstats?: { label: string; value: string }[];
  /** Texto del cuerpo cuando no hay acciones (alerta informativa). */
  body?: string;
  actions: AlertAction[];
  /** Nodo del mapa al que apunta "Ver intersección" (acople al seed, flaggeado). */
  cameraId?: string;
}

export const INITIAL_ALERTS: CommandAlert[] = [
  {
    id: 'al-crit-larco',
    severity: 'crit',
    title: 'Congestión crítica',
    place: 'Larco × Benavides',
    timeAgo: 'hace 2 min',
    conf: 94,
    mstats: [
      { label: 'Cola', value: '142 m' },
      { label: 'Velocidad', value: '6 km/h' },
      { label: 'Demora', value: '96 s' },
    ],
    actions: [
      { label: 'Ver intersección', kind: 'see', primary: true },
      { label: 'Aplicar plan IA', kind: 'apply' },
      { label: 'Escalar', kind: 'generic' },
    ],
    cameraId: 'cam_larco_benavides',
  },
  {
    id: 'al-pred-pardo',
    severity: 'pred',
    title: 'IA · Bloqueo probable en ~10 min',
    place: 'Av. Pardo',
    timeAgo: 'tendencia de cola',
    conf: 91,
    actions: [
      { label: 'Ver predicción', kind: 'predview', primary: true },
      { label: 'Preparar plan', kind: 'generic' },
    ],
  },
  {
    id: 'al-hw-edge4',
    severity: 'hw',
    title: 'Hardware · Latencia alta',
    place: 'Nodo Edge #4',
    timeAgo: 'hace 12 min',
    actions: [
      { label: 'Reiniciar nodo', kind: 'restart', primary: true },
      { label: 'Diagnóstico', kind: 'generic' },
    ],
  },
  {
    id: 'al-info-adaptativo',
    severity: 'info',
    title: 'Modo adaptativo reactivado',
    place: '3 cruces',
    timeAgo: 'hace 25 min',
    body: 'Sin acciones pendientes. Registrado en bitácora del turno.',
    actions: [],
  },
];

/** Alerta dinámica que entra a los ~9 s (prototipo: scheduled incoming alert). */
export const INCOMING_ALERT: CommandAlert = {
  id: 'al-pred-libertadores',
  severity: 'pred',
  title: 'IA · Cola creciente',
  place: 'Pardo × Libertadores',
  timeAgo: 'ahora',
  conf: 88,
  actions: [
    { label: 'Ver predicción', kind: 'predview', primary: true },
    { label: 'Preparar plan', kind: 'generic' },
  ],
};

export const INCOMING_ALERT_DELAY_MS = 9000;

/** KPI demora media (MOCK — sin fuente real; DemoBadge en la card). */
export const DEM_KPI = {
  value: 38,
  spark: [22, 24, 26, 28, 30, 32, 34, 36, 37, 38] as readonly number[],
} as const;

/** Series mock del KPI modal (vel/flu/dem — DemoBadge en el modal). */
export const MODAL_SERIES_MOCK: Record<
  'vel' | 'flu' | 'dem',
  { series: readonly number[]; forecast: readonly number[] }
> = {
  vel: { series: [31, 30, 29, 27.5, 26, 25, 24, 23.2, 22.8, 22.4], forecast: [22, 21.4, 20.8] },
  flu: {
    series: [760, 820, 900, 980, 1040, 1100, 1160, 1200, 1230, 1245],
    forecast: [1255, 1262, 1270],
  },
  dem: { series: [22, 24, 26, 28, 30, 32, 34, 36, 37, 38], forecast: [40, 42, 44] },
};

/** Recomendación IA del drawer (MOCK señalizado — D7: acciones → toasts demo). */
export const DRAWER_AI_MOCK = {
  recommendation: 'Extender fase N–S +12 s durante 3 ciclos',
  impacts: ['−23% cola', '−18 s demora'],
  confidence: 94,
  why: [
    'La cola creció 38% en los últimos 5 min',
    'Flujo aguas arriba (Óvalo) +14%',
    'Patrón histórico: martes, hora pico',
    'Sin demanda peatonal elevada registrada',
  ],
} as const;

/** Historial reciente del drawer (MOCK señalizado). */
export const DRAWER_HISTORY_MOCK = [
  { time: '15:42', event: 'Plan IA aplicado · operador OP · −15% cola' },
  { time: '14:10', event: 'Cambio a modo adaptativo · automático' },
  { time: '09:30', event: 'Calibración de cámara · mantenimiento' },
] as const;
