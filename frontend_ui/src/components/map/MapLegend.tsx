import { JAM_LEVEL_LEGEND } from './edgeStyle';

// FASE 1 rediseño UI — leyenda del mapa (spec: .legend del prototipo). Por
// defecto muestra la escala semántica de tramos (misma fuente que edgeStyle).

interface MapLegendProps {
  title?: string;
  items?: readonly { label: string; swatchClass: string }[];
  className?: string;
}

export function MapLegend({ title = 'Observado', items = JAM_LEVEL_LEGEND, className = '' }: MapLegendProps) {
  return (
    <div
      className={`rounded-ctl border border-line bg-canvas-2/80 px-[13px] py-2.5 text-[11px] backdrop-blur-md ${className}`}
    >
      <b className="mb-[7px] block text-[9.5px] font-bold uppercase tracking-[0.12em] text-ink-2">
        {title}
      </b>
      {items.map((item) => (
        <div key={item.label} className="my-1 flex items-center gap-2 text-ink-2">
          <i aria-hidden="true" className={`inline-block h-1 w-[18px] rounded-xs ${item.swatchClass}`} />
          {item.label}
        </div>
      ))}
    </div>
  );
}
