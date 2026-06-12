// FASE 1 rediseño UI — contador de alertas (spec: .bdg del rail y .pcount de los
// panels del prototipo). NO se llama Badge para no colisionar con ui/Badge.tsx
// (vistas v1). Con count 0 no renderiza, salvo showZero.

interface CountBadgeProps {
  count: number;
  variant?: 'rail' | 'panel';
  showZero?: boolean;
  className?: string;
}

export function CountBadge({ count, variant = 'panel', showZero = false, className = '' }: CountBadgeProps) {
  if (count === 0 && !showZero) return null;

  if (variant === 'rail') {
    return (
      <span
        className={`num grid h-[15px] min-w-[15px] place-items-center rounded-lg border-2 border-canvas-2 bg-bad px-[3px] text-[9px] font-extrabold leading-none text-white ${className}`}
      >
        {count}
      </span>
    );
  }

  return (
    <span
      className={`num rounded-full border border-bad/35 bg-bad/16 px-2 py-[2px] text-[10px] font-extrabold text-bad ${className}`}
    >
      {count}
    </span>
  );
}
