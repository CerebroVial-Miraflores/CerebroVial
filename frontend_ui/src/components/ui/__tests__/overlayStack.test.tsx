/**
 * overlayStack (FASE 3 B1) — Esc cierra SOLO el overlay de arriba.
 * Unit del singleton + integración Drawer+Modal (la deuda que Fase 1 anotó).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render } from '@testing-library/react';

import { overlayCount, pushOverlay, resetOverlayStackForTests } from '../overlayStack';
import { Drawer } from '../Drawer';
import { Modal } from '../Modal';

afterEach(() => {
  resetOverlayStackForTests();
});

describe('overlayStack — unit', () => {
  it('el último pusheado es top; release devuelve el top al anterior', () => {
    const a = pushOverlay();
    const b = pushOverlay();
    expect(a.isTop()).toBe(false);
    expect(b.isTop()).toBe(true);

    b.release();
    expect(a.isTop()).toBe(true);
    expect(overlayCount()).toBe(1);
  });

  it('release fuera de orden no corrompe el stack y es idempotente', () => {
    const a = pushOverlay();
    const b = pushOverlay();
    const c = pushOverlay();

    a.release(); // el de abajo primero
    a.release(); // idempotente
    expect(c.isTop()).toBe(true);
    expect(b.isTop()).toBe(false);
    expect(overlayCount()).toBe(2);

    c.release();
    expect(b.isTop()).toBe(true);
  });

  it('resetOverlayStackForTests limpia todo', () => {
    pushOverlay();
    pushOverlay();
    resetOverlayStackForTests();
    expect(overlayCount()).toBe(0);
  });
});

describe('overlayStack — integración Drawer + Modal', () => {
  it('con Modal sobre Drawer, Esc cierra solo el Modal; el siguiente Esc cierra el Drawer', () => {
    const onCloseDrawer = vi.fn();
    const onCloseModal = vi.fn();
    const { rerender } = render(
      <>
        <Drawer open onClose={onCloseDrawer} title="Intersección">
          contenido drawer
        </Drawer>
        <Modal open onClose={onCloseModal} title="KPI">
          contenido modal
        </Modal>
      </>,
    );

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onCloseModal).toHaveBeenCalledTimes(1);
    expect(onCloseDrawer).not.toHaveBeenCalled();

    // El consumidor cierra el modal → el drawer vuelve a ser top.
    rerender(
      <>
        <Drawer open onClose={onCloseDrawer} title="Intersección">
          contenido drawer
        </Drawer>
        <Modal open={false} onClose={onCloseModal} title="KPI">
          contenido modal
        </Modal>
      </>,
    );

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onCloseDrawer).toHaveBeenCalledTimes(1);
    expect(onCloseModal).toHaveBeenCalledTimes(1);
  });

  it('cambiar la identidad de onClose con el overlay abierto NO re-pushea (el orden se conserva)', () => {
    const onCloseDrawer = vi.fn();
    const onCloseModal = vi.fn();
    const { rerender } = render(
      <>
        <Drawer open onClose={onCloseDrawer} title="d">
          x
        </Drawer>
        <Modal open onClose={onCloseModal} title="m">
          y
        </Modal>
      </>,
    );

    // Nueva identidad del onClose del Drawer (de abajo): si el effect
    // dependiera de onClose, haría release+push y subiría el Drawer arriba.
    const onCloseDrawer2 = vi.fn();
    rerender(
      <>
        <Drawer open onClose={onCloseDrawer2} title="d">
          x
        </Drawer>
        <Modal open onClose={onCloseModal} title="m">
          y
        </Modal>
      </>,
    );

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onCloseModal).toHaveBeenCalledTimes(1);
    expect(onCloseDrawer2).not.toHaveBeenCalled();
  });

  it('un overlay solo sigue cerrando con Esc (sin regresión de Fase 1)', () => {
    const onClose = vi.fn();
    render(
      <Drawer open onClose={onClose} title="solo">
        x
      </Drawer>,
    );
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
