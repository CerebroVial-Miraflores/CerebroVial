// FASE 1 rediseño UI — señalizador obligatorio de datos simulados (regla
// "Rediseño UI" del CLAUDE.md: todo mock visible en la app lo lleva). Paleta
// info, no brand: es un aviso de procedencia del dato, no decoración de marca.

interface DemoBadgeProps {
  className?: string;
}

export function DemoBadge({ className = '' }: DemoBadgeProps) {
  return (
    <span
      className={`whitespace-nowrap rounded-full border border-info/40 bg-info/10 px-[9px] py-1 text-[9.5px] font-bold uppercase tracking-[0.12em] text-info ${className}`}
    >
      Demo · datos simulados
    </span>
  );
}
