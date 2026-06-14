import type { DetectionBox, DetectionFrame } from '../../../types/detections';

// FASE 4 Mitad A — overlay SVG de cajas de detección sobre el <video> del
// HlsPlayer (D-019). Reemplaza las bboxes que el edge quemaba en el MJPEG.
//
// ALINEACIÓN: el viewBox usa las dims del frame de inferencia (aspecto de la
// fuente) y preserveAspectRatio="xMidYMid meet" letterboxea idéntico al
// object-fit:contain del video → las coords normalizadas mapean lineal, sin medir
// el DOM. El SVG comparte contenedor y tamaño con el video (inset-0, h/w-full).
//
// Las cajas se reciben YA filtradas por frescura/isActive desde useDetections +
// el panel; este componente solo dibuja. `pointer-events-none`: no intercepta
// clicks del video.

// Color de stroke por tipo, desde tokens.css (mandato CLAUDE.md). Map con strings
// literales para que el scanner de Tailwind genere las utilidades stroke-*.
const STROKE_BY_TYPE: Record<string, string> = {
  car: 'stroke-ok',
  bus: 'stroke-warn',
  truck: 'stroke-bad',
  motorcycle: 'stroke-info',
};
const STROKE_DEFAULT = 'stroke-brand';

interface DetectionOverlayProps {
  boxes: DetectionBox[];
  frame: DetectionFrame | null;
}

export function DetectionOverlay({ boxes, frame }: DetectionOverlayProps) {
  // Sin frame (cámara sin datos) o sin cajas (stale/vacío) → no se dibuja nada.
  if (!frame || boxes.length === 0) return null;

  const { width, height } = frame;

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      {boxes.map((b) => {
        const [x1, y1, x2, y2] = b.bbox;
        const x = x1 * width;
        const y = y1 * height;
        const w = (x2 - x1) * width;
        const h = (y2 - y1) * height;
        const stroke = STROKE_BY_TYPE[b.type] ?? STROKE_DEFAULT;
        return (
          <g key={b.id}>
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill="none"
              className={stroke}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
            {/* Label mínimo de tipo (sin id/speed). `fill-ink` (token) sobre el
                video; el color de la caja lleva la señal por tipo. */}
            <text x={x} y={y - 4} fontSize={24} fontWeight={700} className="fill-ink">
              {b.type}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
