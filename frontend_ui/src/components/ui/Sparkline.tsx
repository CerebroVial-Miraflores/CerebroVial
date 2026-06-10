// FASE 1 rediseño UI — sparkline SVG puro (spec: sparkline() del prototipo:
// área + línea + punto final, normalización min/max). El color va por
// currentColor: el padre define text-* (var() no es válido como atributo de
// presentación SVG; currentColor sí).

interface SparklineProps {
  data: readonly number[];
  className?: string;
}

// eslint-disable-next-line react-refresh/only-export-components -- helper puro colocado al componente (testeable sin render)
export function buildSparklinePoints(data: readonly number[], width = 120, height = 36): string {
  if (data.length < 2) return '';
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  return data
    .map((value, i) => {
      const x = ((i / (data.length - 1)) * width).toFixed(1);
      const y = (height - 4 - ((value - min) / range) * (height - 10)).toFixed(1);
      return `${x},${y}`;
    })
    .join(' ');
}

export function Sparkline({ data, className = '' }: SparklineProps) {
  if (data.length < 2) return null;
  const width = 120;
  const height = 36;
  const points = buildSparklinePoints(data, width, height);
  const [lastX, lastY] = points.split(' ').at(-1)!.split(',');
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      className={className}
    >
      <polyline points={`${points} ${width},${height} 0,${height}`} fill="currentColor" fillOpacity={0.09} stroke="none" />
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
      <circle cx={lastX} cy={lastY} r={2.6} fill="currentColor" />
    </svg>
  );
}
