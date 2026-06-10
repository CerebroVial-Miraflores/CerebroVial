import type { PathOptions } from 'leaflet';

// FASE 1 rediseño UI — estilo de tramos por jam_level con tokens (función pura,
// reutilizable en Fase 3 cuando las vistas de mapa migren al rediseño).
//
// Por qué className y no color: Leaflet aplica PathOptions.color con
// path.setAttribute('stroke', …) y los atributos de presentación SVG NO procesan
// var(); el color del token se aplica vía CSS en map.css
// (.edge-jam-N { stroke: var(--color-…) }), cuyas reglas pisan al atributo.
// Requiere el renderer SVG (el default de react-leaflet).
//
// Caveat Fase 3: Leaflet asigna className solo al CREAR el path (setStyle no la
// actualiza) → recolorear = remontar la capa por `key`, exactamente el patrón
// que CongestionMapView ya usa y documenta en sus tests.
//
// Mapping PROVISIONAL 0-1 ok-road / 2-3 warn / 4 bad / 5 sev. Condición de
// cierre: validar umbrales contra la semántica real de jam_level que usan
// utils/congestion.ts y el backend antes de integrar en Fase 3.

export type JamLevel = 0 | 1 | 2 | 3 | 4 | 5;

export interface EdgeStyle {
  className: string;
  weight: number;
  opacity: number;
}

const STYLE_BY_LEVEL: Record<JamLevel, EdgeStyle> = {
  0: { className: 'edge-jam-0', weight: 3, opacity: 0.85 },
  1: { className: 'edge-jam-1', weight: 4, opacity: 0.85 },
  2: { className: 'edge-jam-2', weight: 5, opacity: 0.92 },
  3: { className: 'edge-jam-3', weight: 6, opacity: 0.92 },
  4: { className: 'edge-jam-4', weight: 7.5, opacity: 1 },
  5: { className: 'edge-jam-5', weight: 9, opacity: 1 },
};

// Tramo sin dato (edge huérfano, level inválido): neutro tenue, nunca rojo.
const NEUTRAL_STYLE: EdgeStyle = { className: 'edge-jam-none', weight: 3, opacity: 0.35 };

export function jamLevelStyle(level: number | null | undefined): EdgeStyle {
  if (level == null || !Number.isInteger(level) || level < 0 || level > 5) {
    return NEUTRAL_STYLE;
  }
  return STYLE_BY_LEVEL[level as JamLevel];
}

// Azúcar para <GeoJSON style={...}> de react-leaflet.
export function jamLevelPathOptions(level: number | null | undefined): PathOptions {
  const style = jamLevelStyle(level);
  return { className: style.className, weight: style.weight, opacity: style.opacity, fill: false };
}

// Escala semántica para la leyenda — misma fuente que el mapping de arriba.
export const JAM_LEVEL_LEGEND: readonly { label: string; swatchClass: string }[] = [
  { label: 'Fluido', swatchClass: 'bg-ok-road' },
  { label: 'Lento', swatchClass: 'bg-warn' },
  { label: 'Congestionado', swatchClass: 'bg-bad' },
  { label: 'Crítico', swatchClass: 'bg-sev' },
];
