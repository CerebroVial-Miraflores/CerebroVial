import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { AlertAccordion } from '../AlertAccordion';

describe('AlertAccordion', () => {
  it.each([
    ['crit', 'border-l-bad'],
    ['pred', 'border-l-sev'],
    ['hw', 'border-l-warn'],
    ['info', 'border-l-info'],
  ] as const)('severidad %s pinta el borde izquierdo de su escala', (severity, expected) => {
    const { container } = render(
      <AlertAccordion severity={severity} title="t" open={false} onToggle={() => {}} />,
    );
    expect(container.firstChild).toHaveClass(expected);
  });

  it('header expone aria-expanded y dispara onToggle', () => {
    const onToggle = vi.fn();
    const { rerender } = render(
      <AlertAccordion severity="crit" title="Congestión crítica" open={false} onToggle={onToggle} />,
    );
    const header = screen.getByRole('button', { expanded: false });
    fireEvent.click(header);
    expect(onToggle).toHaveBeenCalledTimes(1);

    rerender(
      <AlertAccordion severity="crit" title="Congestión crítica" open onToggle={onToggle} />,
    );
    expect(screen.getByRole('button', { expanded: true })).toBeInTheDocument();
  });

  it('cuerpo colapsado usa grid-rows-[0fr] y abierto grid-rows-[1fr]', () => {
    const { container, rerender } = render(
      <AlertAccordion severity="hw" title="t" open={false} onToggle={() => {}}>
        <p>detalle</p>
      </AlertAccordion>,
    );
    const body = container.querySelector('.grid');
    expect(body).toHaveClass('grid-rows-[0fr]');

    rerender(
      <AlertAccordion severity="hw" title="t" open onToggle={() => {}}>
        <p>detalle</p>
      </AlertAccordion>,
    );
    expect(container.querySelector('.grid')).toHaveClass('grid-rows-[1fr]');
    expect(screen.getByText('detalle')).toBeInTheDocument();
  });

  it('isNew enciende la animación de entrada; sin isNew no', () => {
    const { container, rerender } = render(
      <AlertAccordion severity="pred" title="t" open={false} onToggle={() => {}} isNew />,
    );
    expect(container.firstChild).toHaveClass('animate-new-alert');

    rerender(<AlertAccordion severity="pred" title="t" open={false} onToggle={() => {}} />);
    expect(container.firstChild).not.toHaveClass('animate-new-alert');
  });

  it('renderiza meta e icono cuando se proveen', () => {
    render(
      <AlertAccordion
        severity="crit"
        icon={<svg data-testid="ic" />}
        title="Congestión crítica"
        meta={
          <>
            <span>Larco × Benavides</span>
            <span>hace 2 min</span>
          </>
        }
        open={false}
        onToggle={() => {}}
      />,
    );
    expect(screen.getByTestId('ic')).toBeInTheDocument();
    expect(screen.getByText('Larco × Benavides')).toBeInTheDocument();
    expect(screen.getByText('hace 2 min')).toBeInTheDocument();
  });
});
