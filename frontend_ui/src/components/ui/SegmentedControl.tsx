import type { ReactNode } from 'react';

// FASE 1 rediseño UI — control segmentado (spec: .seg del prototipo; lo usan el
// toggle Operación/Gerencia y el selector Histórico/Ahora/Predicción).

interface SegmentedOption<T extends string> {
  value: T;
  label: ReactNode;
}

interface SegmentedControlProps<T extends string> {
  options: readonly SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel?: string;
  className?: string;
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  className = '',
}: SegmentedControlProps<T>) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className={`flex w-fit gap-[2px] rounded-full border border-line bg-panel p-[3px] ${className}`}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={`rounded-full px-[13px] py-[5px] text-[12.5px] font-semibold transition-all duration-200 ease-fluid ${
              active
                ? 'bg-linear-to-br from-brand to-accent text-white shadow-md shadow-brand/40'
                : 'text-ink-2 hover:text-ink'
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
