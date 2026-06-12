// FASE 1 rediseño UI — badge de modo temporal del mapa (spec: .modebadge del
// prototipo): EN VIVO / PREDICCIÓN +N MIN (warn) / HISTÓRICO · ts (info).

export type MapMode = 'live' | 'prediction' | 'historic';

interface MapModeBadgeProps {
  mode: MapMode;
  /** Horizonte en minutos (solo prediction). */
  offsetMin?: number;
  /** Etiqueta temporal (solo historic), p. ej. "hoy 16:00". */
  timestampLabel?: string;
  className?: string;
}

const MODE_CLASSES: Record<MapMode, string> = {
  live: 'border-line-2 text-ink',
  prediction: 'border-warn/50 text-warn',
  historic: 'border-info/50 text-info',
};

export function MapModeBadge({ mode, offsetMin = 15, timestampLabel, className = '' }: MapModeBadgeProps) {
  const label =
    mode === 'live'
      ? 'EN VIVO'
      : mode === 'prediction'
        ? `PREDICCIÓN · +${offsetMin} MIN`
        : `HISTÓRICO${timestampLabel ? ` · ${timestampLabel}` : ''}`;
  return (
    <span
      className={`num inline-flex items-center gap-2 rounded-full border bg-canvas-2/80 px-3 py-1.5 text-[11px] font-bold tracking-[0.05em] backdrop-blur-md ${MODE_CLASSES[mode]} ${className}`}
    >
      {label}
    </span>
  );
}
