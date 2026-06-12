// FASE 1 rediseño UI — barra de confianza del modelo (spec: .conf/.confrow del
// prototipo). percent se clampa a [0, 100].

interface ConfBarProps {
  percent: number;
  className?: string;
}

export function ConfBar({ percent, className = '' }: ConfBarProps) {
  const pct = Math.min(100, Math.max(0, percent));
  return (
    <div className={`flex items-center gap-2 text-[10.5px] text-ink-2 ${className}`}>
      <span>conf.</span>
      <div className="h-1 flex-1 overflow-hidden rounded-xs bg-white/9">
        <div
          role="progressbar"
          aria-valuenow={Math.round(pct)}
          aria-valuemin={0}
          aria-valuemax={100}
          className="h-full rounded-xs bg-linear-to-r from-ok to-ok-road transition-[width] duration-1000 ease-fluid"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="num">{Math.round(pct)}%</span>
    </div>
  );
}
