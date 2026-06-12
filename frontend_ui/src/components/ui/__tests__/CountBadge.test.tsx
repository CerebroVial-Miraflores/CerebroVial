import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { CountBadge } from '../CountBadge';

describe('CountBadge', () => {
  it('count 0 no renderiza salvo showZero', () => {
    const { container, rerender } = render(<CountBadge count={0} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<CountBadge count={0} showZero />);
    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('variante rail: sólida con borde oscuro', () => {
    render(<CountBadge count={3} variant="rail" />);
    expect(screen.getByText('3')).toHaveClass('bg-bad', 'border-canvas-2');
  });

  it('variante panel: translúcida con texto bad', () => {
    render(<CountBadge count={4} variant="panel" />);
    expect(screen.getByText('4')).toHaveClass('bg-bad/16', 'text-bad');
  });
});
