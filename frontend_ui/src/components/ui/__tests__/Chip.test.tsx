import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { Chip, HChip } from '../Chip';

describe('Chip', () => {
  it('refleja on en aria-pressed y muestra el dot solo encendido', () => {
    const { rerender } = render(<Chip on onToggle={() => {}}>Tráfico</Chip>);
    const chip = screen.getByRole('button', { name: 'Tráfico' });
    expect(chip).toHaveAttribute('aria-pressed', 'true');
    expect(chip.querySelector('span')).not.toBeNull();

    rerender(<Chip on={false} onToggle={() => {}}>Tráfico</Chip>);
    expect(chip).toHaveAttribute('aria-pressed', 'false');
    expect(chip.querySelector('span')).toBeNull();
  });

  it('click invierte el estado vía onToggle', () => {
    const onToggle = vi.fn();
    render(<Chip on={false} onToggle={onToggle}>Cámaras</Chip>);
    fireEvent.click(screen.getByRole('button', { name: 'Cámaras' }));
    expect(onToggle).toHaveBeenCalledWith(true);
  });

  it('HChip usa la paleta warn cuando está encendido', () => {
    const onToggle = vi.fn();
    render(<HChip on onToggle={onToggle}>+15 min</HChip>);
    const chip = screen.getByRole('button', { name: '+15 min' });
    expect(chip).toHaveClass('border-warn/55', 'text-warn');
    fireEvent.click(chip);
    expect(onToggle).toHaveBeenCalledWith(false);
  });
});
