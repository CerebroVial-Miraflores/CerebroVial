import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { Drawer } from '../Drawer';

function renderDrawer(open: boolean, onClose = vi.fn()) {
  const utils = render(
    <Drawer open={open} onClose={onClose} title="Av. Larco × Av. Benavides" status="bad" statusLabel="CRÍTICO">
      <p>detalle</p>
    </Drawer>,
  );
  return { ...utils, onClose };
}

describe('Drawer', () => {
  it('cerrado: montado pero oculto (translate fuera + aria-hidden, sin lock)', () => {
    renderDrawer(false);
    const panel = document.querySelector('aside')!;
    expect(panel).toHaveClass('translate-x-[106%]');
    expect(panel).toHaveAttribute('aria-hidden', 'true');
    expect(document.body).not.toHaveClass('overflow-hidden');
  });

  it('abierto: visible, con StatusChip en el header y body lockeado', () => {
    renderDrawer(true);
    const panel = document.querySelector('aside')!;
    expect(panel).toHaveClass('translate-x-0');
    expect(panel).toHaveAttribute('aria-hidden', 'false');
    expect(screen.getByText('CRÍTICO')).toBeInTheDocument();
    expect(screen.getByText('Av. Larco × Av. Benavides')).toBeInTheDocument();
    expect(document.body).toHaveClass('overflow-hidden');
  });

  // jsdom no evalúa media queries: este es el CONTRATO de clases responsive
  // (sheet full <md; panel min(480px,94vw) ≥md). La visibilidad real se valida
  // en el manual a 390px.
  it('contrato responsive: sheet <md, panel lateral ≥md', () => {
    renderDrawer(true);
    const panel = document.querySelector('aside')!;
    expect(panel).toHaveClass('inset-0', 'md:left-auto', 'md:w-[min(480px,94vw)]');
  });

  it('Esc y click en scrim cierran', () => {
    const { onClose } = renderDrawer(true);
    fireEvent.click(screen.getByTestId('drawer-scrim'));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it('cerrado no escucha Esc', () => {
    const { onClose } = renderDrawer(false);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
  });
});
