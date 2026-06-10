import type { ButtonHTMLAttributes } from 'react';

// FASE 1 rediseño UI — botón base del design system (spec: .btn del prototipo).
// done = estado "aplicado" (verde, no interactivo): además de pointer-events-none
// va disabled, porque pointer-events solo no bloquea la activación por teclado.

type ButtonVariant = 'default' | 'pri' | 'done';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  default: 'border-line bg-panel text-ink hover:-translate-y-px hover:border-line-2',
  pri: 'border-transparent bg-linear-to-br from-brand to-accent text-white shadow-lg shadow-brand/35 hover:-translate-y-px hover:brightness-110',
  done: 'pointer-events-none border-ok/45 bg-ok/14 text-ok',
};

export function Button({ variant = 'default', className = '', disabled, type, ...rest }: ButtonProps) {
  return (
    <button
      type={type ?? 'button'}
      disabled={variant === 'done' ? true : disabled}
      className={`rounded-btn border px-3 py-[7px] text-[11.5px] font-semibold transition-all duration-200 ${VARIANT_CLASSES[variant]} ${className}`}
      {...rest}
    />
  );
}
