// FASE 3 (B2) — chart grande del KPI modal (spec: bigChart() del prototipo):
// 4 gridlines, área + línea de la serie, forecast PUNTEADO a continuación,
// divisor vertical "ahora" en el último punto observado y labels del eje x.
// Patrón Sparkline escalado: SVG puro 720×248, color por currentColor (el
// padre define text-*; var() no es válido como atributo de presentación SVG).

interface BigChartGeometry {
  gridY: number[];
  areaPoints: string;
  linePoints: string;
  /** null si no hay forecast (p. ej. predicción 503 → se omite con nota). */
  forecastPoints: string | null;
  nowX: number;
  nowY: number;
  labelXs: number[];
}

// eslint-disable-next-line react-refresh/only-export-components -- helper puro colocado al componente (testeable sin render)
export function buildBigChartGeometry(
  series: readonly number[],
  forecast: readonly number[] = [],
  width = 720,
  height = 248,
  pad = 26,
  labelCount = 5,
): BigChartGeometry | null {
  if (series.length < 2) return null;
  const all = [...series, ...forecast];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const range = max - min || 1;
  const n = series.length + forecast.length - 1;
  const x = (i: number) => pad + (i / n) * (width - 2 * pad);
  const y = (v: number) => height - pad - ((v - min) / range) * (height - 2 * pad);
  const point = (i: number, v: number) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`;

  const gridY = Array.from({ length: 4 }, (_, i) => pad + (i / 3) * (height - 2 * pad));
  const linePoints = series.map((v, i) => point(i, v)).join(' ');
  const nowX = x(series.length - 1);
  const nowY = y(series[series.length - 1]);
  const areaPoints = `${linePoints} ${nowX.toFixed(1)},${(height - pad).toFixed(1)} ${x(0).toFixed(1)},${(height - pad).toFixed(1)}`;
  const forecastPoints =
    forecast.length > 0
      ? [point(series.length - 1, series[series.length - 1]), ...forecast.map((v, i) => point(series.length + i, v))].join(' ')
      : null;
  const labelXs = Array.from({ length: labelCount }, (_, i) => pad + (i / (labelCount - 1)) * (width - 2 * pad));

  return { gridY, areaPoints, linePoints, forecastPoints, nowX, nowY, labelXs };
}

interface BigChartProps {
  series: readonly number[];
  /** Continuación predicha (línea punteada). Vacío → no se dibuja. */
  forecast?: readonly number[];
  /** Labels del eje x, repartidos uniformemente (prototipo: −3h…+45m). */
  xLabels?: readonly string[];
  className?: string;
}

export function BigChart({ series, forecast = [], xLabels = [], className = '' }: BigChartProps) {
  const width = 720;
  const height = 248;
  const geometry = buildBigChartGeometry(series, forecast, width, height, 26, Math.max(2, xLabels.length));
  if (geometry === null) return null;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      className={className}
    >
      {geometry.gridY.map((gy) => (
        <line
          key={gy}
          x1={26}
          y1={gy}
          x2={width - 26}
          y2={gy}
          stroke="currentColor"
          strokeOpacity={0.12}
          strokeWidth={1}
        />
      ))}
      <polyline points={geometry.areaPoints} fill="currentColor" fillOpacity={0.12} stroke="none" />
      <polyline
        points={geometry.linePoints}
        fill="none"
        stroke="currentColor"
        strokeWidth={2.5}
        strokeLinecap="round"
      />
      {geometry.forecastPoints !== null && (
        <polyline
          data-testid="bigchart-forecast"
          points={geometry.forecastPoints}
          fill="none"
          stroke="currentColor"
          strokeWidth={2.5}
          strokeDasharray="6 6"
          strokeOpacity={0.75}
        />
      )}
      <line
        x1={geometry.nowX}
        y1={20}
        x2={geometry.nowX}
        y2={height - 26}
        stroke="currentColor"
        strokeOpacity={0.25}
        strokeWidth={1}
        strokeDasharray="3 4"
      />
      <circle cx={geometry.nowX} cy={geometry.nowY} r={4} fill="currentColor" />
      <text
        x={geometry.nowX}
        y={14}
        textAnchor="middle"
        fill="currentColor"
        fillOpacity={0.65}
        fontSize={10}
        fontWeight={700}
      >
        ahora
      </text>
      {xLabels.map((label, i) => (
        <text
          key={label}
          x={geometry.labelXs[i]}
          y={height - 7}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.45}
          fontSize={10}
        >
          {label}
        </text>
      ))}
    </svg>
  );
}
