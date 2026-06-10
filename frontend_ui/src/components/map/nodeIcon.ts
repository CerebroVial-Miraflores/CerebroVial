// FASE 1 rediseño UI — builder puro del HTML del divIcon de nodos semáforo
// (spec: .node del prototipo: ring + core por estado, halo pulsante si crítico).
// Visual NUEVO del rediseño — NO confundir con utils/markerVisual.ts (marcadores
// de cámaras), que no se toca.
//
// Colores vía style inline: los inline styles SÍ procesan var(); los atributos
// de presentación SVG no. El halo anima con la clase animate-halo-ping (token
// --animate-halo-ping) y necesita transform-box: fill-box para escalar desde su
// centro. El <svg> lleva overflow="visible" para que el halo expandido no se
// recorte dentro del iconSize.

export type NodeStatus = 'ok' | 'warn' | 'bad';

const STROKE_BY_STATUS: Record<NodeStatus, string> = {
  ok: 'var(--color-ok)',
  warn: 'var(--color-warn)',
  bad: 'var(--color-bad)',
};

export const NODE_ICON_SIZE = 48;

export function buildNodeIconHtml(status: NodeStatus, opts: { critical?: boolean } = {}): string {
  const stroke = STROKE_BY_STATUS[status];
  const center = NODE_ICON_SIZE / 2;
  const halo = opts.critical
    ? `<circle r="9" cx="${center}" cy="${center}" fill="none" class="animate-halo-ping" style="stroke: ${stroke}; stroke-width: 2; transform-box: fill-box; transform-origin: center;"></circle>`
    : '';
  return (
    `<svg width="${NODE_ICON_SIZE}" height="${NODE_ICON_SIZE}" viewBox="0 0 ${NODE_ICON_SIZE} ${NODE_ICON_SIZE}" overflow="visible" aria-hidden="true">` +
    halo +
    `<circle r="9" cx="${center}" cy="${center}" style="fill: var(--color-canvas-2); stroke: ${stroke}; stroke-width: 2.6;"></circle>` +
    `<circle r="3.4" cx="${center}" cy="${center}" style="fill: var(--color-ink);"></circle>` +
    `</svg>`
  );
}
