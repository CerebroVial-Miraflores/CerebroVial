import { Chip, HChip } from '../ui/Chip';

// FASE 1 rediseño UI — fila de toggles de capas del mapa (spec: .chips del
// prototipo). tone 'warn' usa HChip (horizontes/históricos).

interface LayerChipsProps {
  layers: readonly { id: string; label: string; on: boolean; tone?: 'default' | 'warn' }[];
  onToggle: (id: string) => void;
  className?: string;
}

export function LayerChips({ layers, onToggle, className = '' }: LayerChipsProps) {
  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      {layers.map((layer) => {
        const ChipComponent = layer.tone === 'warn' ? HChip : Chip;
        return (
          <ChipComponent key={layer.id} on={layer.on} onToggle={() => onToggle(layer.id)}>
            {layer.label}
          </ChipComponent>
        );
      })}
    </div>
  );
}
