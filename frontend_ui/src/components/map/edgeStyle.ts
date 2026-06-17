import type { ExpressionSpecification, LineLayerSpecification } from 'maplibre-gl';

// FASE 4 migración MapLibre — estilo de tramos por jam_level como expresión
// data-driven (función pura). Reemplaza al enfoque className/PathOptions de
// Leaflet: MapLibre no resuelve var() en un paint, así que los colores van como
// literales byte-alineados a tokens.css (mapeo histórico: 0-1 ok-road / 2-3
// warn / 4 bad / 5 sev / sin-dato ink-3). El recolor en vivo lo da source.setData
// (sin remontar la capa), por eso muere el remount-por-key.
//
// Grosor: escala FINA 2→5 (D-Fase4) — variable por nivel para preservar la
// redundancia no-cromática de CA-22.3 (el nivel se lee también por grosor, no
// solo por color: daltonismo), pero con líneas base delgadas y limpias. Es un
// refinamiento deliberado del 3→9 anterior; calibrable a ojo contra las 1660
// aristas reales (subir el techo si las congestionadas no se distinguen).

export type JamLevel = 0 | 1 | 2 | 3 | 4 | 5;

interface LevelStyle {
  color: string;
  weight: number;
  opacity: number;
}

// Color (token), grosor fino y opacidad por nivel. Mapeo cromático idéntico al
// histórico: 0-1 ok-road, 2-3 warn, 4 bad, 5 sev.
const STYLE_BY_LEVEL: Record<JamLevel, LevelStyle> = {
  0: { color: '#0fae79', weight: 2, opacity: 0.85 }, // --color-ok-road
  1: { color: '#0fae79', weight: 2.4, opacity: 0.85 },
  2: { color: '#f59e0b', weight: 2.8, opacity: 0.92 }, // --color-warn
  3: { color: '#f59e0b', weight: 3.4, opacity: 0.92 },
  4: { color: '#ef4444', weight: 4.2, opacity: 1 }, // --color-bad
  5: { color: '#a855f7', weight: 5, opacity: 1 }, // --color-sev
};

// Tramo sin dato (edge huérfano, nivel inválido/null): neutro tenue, nunca rojo.
const NEUTRAL_STYLE: LevelStyle = { color: '#5b6275', weight: 2, opacity: 0.35 }; // --color-ink-3

const LEVELS: JamLevel[] = [0, 1, 2, 3, 4, 5];

/**
 * Paint data-driven para una capa `line` de MapLibre: color/grosor/opacidad por
 * nivel de congestión leído de `property` (default `congestion_level`; el demo
 * de /ui-lab usa `jam_level`). Entrada null/ausente (arista sin estado, pre-merge)
 * → cae al fallback neutro.
 */
export function jamLevelPaint(
  property = 'congestion_level',
): LineLayerSpecification['paint'] {
  const input: ExpressionSpecification = ['get', property];
  const colorExpr = [
    'match',
    input,
    ...LEVELS.flatMap((l) => [l, STYLE_BY_LEVEL[l].color]),
    NEUTRAL_STYLE.color,
  ] as ExpressionSpecification;
  const widthExpr = [
    'match',
    input,
    ...LEVELS.flatMap((l) => [l, STYLE_BY_LEVEL[l].weight]),
    NEUTRAL_STYLE.weight,
  ] as ExpressionSpecification;
  const opacityExpr = [
    'match',
    input,
    ...LEVELS.flatMap((l) => [l, STYLE_BY_LEVEL[l].opacity]),
    NEUTRAL_STYLE.opacity,
  ] as ExpressionSpecification;
  return {
    'line-color': colorExpr,
    'line-width': widthExpr,
    'line-opacity': opacityExpr,
  };
}

// Escala semántica para la leyenda — misma fuente que el mapping de arriba.
export const JAM_LEVEL_LEGEND: readonly { label: string; swatchClass: string }[] = [
  { label: 'Fluido', swatchClass: 'bg-ok-road' },
  { label: 'Lento', swatchClass: 'bg-warn' },
  { label: 'Congestionado', swatchClass: 'bg-bad' },
  { label: 'Crítico', swatchClass: 'bg-sev' },
];

// Leyenda del modo predicción (FASE 3): niveles 0-4 de timeLoss bucketeados a
// los mismos colores del mapping (0-1 ok-road / 2-3 warn / 4 bad). El nivel 5
// no existe en predicción → sev no aparece.
export const PREDICTION_LEGEND: readonly { label: string; swatchClass: string }[] = [
  { label: 'Sin demora · leve', swatchClass: 'bg-ok-road' },
  { label: 'Demora moderada · alta', swatchClass: 'bg-warn' },
  { label: 'Demora severa', swatchClass: 'bg-bad' },
];
