import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { Pill } from '../Pill';

describe('Pill', () => {
  it('muestra el contenido con paleta ok y dot pulsante', () => {
    render(<Pill>IA ACTIVA</Pill>);
    const pill = screen.getByText('IA ACTIVA');
    expect(pill).toHaveClass('border-ok/35', 'text-ok');
    expect(pill.querySelector('.animate-pulse')).not.toBeNull();
  });
});
