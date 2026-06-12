import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { ConfBar } from '../ConfBar';

describe('ConfBar', () => {
  it('pinta el ancho según percent y lo rotula', () => {
    render(<ConfBar percent={87} />);
    expect(screen.getByRole('progressbar')).toHaveStyle({ width: '87%' });
    expect(screen.getByText('87%')).toBeInTheDocument();
  });

  it('clampa fuera de rango a [0, 100]', () => {
    const { rerender } = render(<ConfBar percent={120} />);
    expect(screen.getByRole('progressbar')).toHaveStyle({ width: '100%' });
    rerender(<ConfBar percent={-5} />);
    expect(screen.getByRole('progressbar')).toHaveStyle({ width: '0%' });
    expect(screen.getByText('0%')).toBeInTheDocument();
  });
});
