import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { Modal } from '../Modal';

describe('Modal', () => {
  it('cerrado no renderiza nada', () => {
    render(
      <Modal open={false} onClose={() => {}}>
        <p>contenido</p>
      </Modal>,
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('abierto: dialog accesible, foco al contenedor y body lockeado', () => {
    render(
      <Modal open onClose={() => {}} title="Título">
        <p>contenido</p>
      </Modal>,
    );
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(document.activeElement).toBe(dialog);
    expect(document.body).toHaveClass('overflow-hidden');
  });

  it('Esc y click en scrim cierran; click en el contenido no', () => {
    const onClose = vi.fn();
    render(
      <Modal open onClose={onClose} title="Título">
        <p>contenido</p>
      </Modal>,
    );
    fireEvent.click(screen.getByText('contenido'));
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId('modal-scrim'));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it('al cerrar libera el lock y restaura el foco al trigger', () => {
    function Harness() {
      return (
        <>
          <button type="button">trigger</button>
          <Modal open onClose={() => {}} title="T">
            <p>x</p>
          </Modal>
        </>
      );
    }
    const { rerender } = render(<Harness />);
    // Simula que el trigger tenía el foco antes de reabrir
    rerender(
      <>
        <button type="button">trigger</button>
        <Modal open={false} onClose={() => {}} title="T">
          <p>x</p>
        </Modal>
      </>,
    );
    expect(document.body).not.toHaveClass('overflow-hidden');
  });

  it('restaura el foco al elemento previamente enfocado', () => {
    function Harness({ open }: { open: boolean }) {
      return (
        <>
          <button type="button">trigger</button>
          <Modal open={open} onClose={() => {}} title="T">
            <p>x</p>
          </Modal>
        </>
      );
    }
    const { rerender } = render(<Harness open={false} />);
    const trigger = screen.getByRole('button', { name: 'trigger' });
    trigger.focus();
    rerender(<Harness open />);
    expect(document.activeElement).toBe(screen.getByRole('dialog'));
    rerender(<Harness open={false} />);
    expect(document.activeElement).toBe(trigger);
  });
});
