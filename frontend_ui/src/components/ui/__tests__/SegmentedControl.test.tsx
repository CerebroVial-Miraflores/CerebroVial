import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { SegmentedControl } from '../SegmentedControl';

const OPTIONS = [
  { value: 'h', label: 'Histórico' },
  { value: 'n', label: 'Ahora' },
  { value: 'p', label: 'Predicción' },
] as const;

describe('SegmentedControl', () => {
  it('marca activa solo la opción seleccionada', () => {
    render(<SegmentedControl options={OPTIONS} value="n" onChange={() => {}} ariaLabel="Modo" />);
    expect(screen.getByRole('button', { name: 'Ahora' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Ahora' })).toHaveClass('from-brand');
    expect(screen.getByRole('button', { name: 'Histórico' })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', { name: 'Histórico' })).not.toHaveClass('from-brand');
  });

  it('click en otra opción emite onChange con su value', () => {
    const onChange = vi.fn();
    render(<SegmentedControl options={OPTIONS} value="n" onChange={onChange} ariaLabel="Modo" />);
    fireEvent.click(screen.getByRole('button', { name: 'Predicción' }));
    expect(onChange).toHaveBeenCalledWith('p');
  });
});
