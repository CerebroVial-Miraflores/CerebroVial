import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { Button } from '../Button';

describe('Button', () => {
  it('default dispara onClick y usa borde/panel de tokens', () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Simular</Button>);
    const button = screen.getByRole('button', { name: 'Simular' });
    expect(button).toHaveClass('rounded-btn', 'border-line', 'bg-panel');
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('pri lleva el gradiente brand→accent', () => {
    render(<Button variant="pri">Aplicar</Button>);
    expect(screen.getByRole('button', { name: 'Aplicar' })).toHaveClass('from-brand', 'to-accent');
  });

  it('done queda deshabilitado y no dispara onClick', () => {
    const onClick = vi.fn();
    render(
      <Button variant="done" onClick={onClick}>
        ✓ Aplicado
      </Button>,
    );
    const button = screen.getByRole('button', { name: '✓ Aplicado' });
    expect(button).toBeDisabled();
    expect(button).toHaveClass('pointer-events-none');
    fireEvent.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });
});
