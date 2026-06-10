import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { DemoBadge } from '../DemoBadge';

describe('DemoBadge', () => {
  it('muestra el texto fijo con paleta info', () => {
    render(<DemoBadge />);
    expect(screen.getByText('Demo · datos simulados')).toHaveClass('text-info', 'border-info/40');
  });
});
