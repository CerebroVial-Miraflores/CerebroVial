import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { StatusChip } from '../StatusChip';

describe('StatusChip', () => {
  it.each([
    ['ok', 'border-ok/40'],
    ['warn', 'border-warn/40'],
    ['bad', 'border-bad/40'],
  ] as const)('aplica la paleta de %s', (status, expectedClass) => {
    render(<StatusChip status={status}>ETIQUETA</StatusChip>);
    expect(screen.getByText('ETIQUETA')).toHaveClass(expectedClass);
  });

  it('recovery usa paleta warn y label default "EN RECUPERACIÓN"', () => {
    render(<StatusChip status="recovery" />);
    const chip = screen.getByText('EN RECUPERACIÓN');
    expect(chip).toHaveClass('border-warn/40', 'text-warn');
  });

  it('children pisa el label default', () => {
    render(<StatusChip status="bad">CRÍTICO · ADAPTATIVO EN PAUSA</StatusChip>);
    expect(screen.getByText('CRÍTICO · ADAPTATIVO EN PAUSA')).toBeInTheDocument();
  });
});
