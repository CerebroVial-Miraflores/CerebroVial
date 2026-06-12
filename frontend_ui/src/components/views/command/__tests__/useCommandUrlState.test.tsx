/**
 * Tests de la máquina de estados URL del comando (FASE 3).
 *
 * createMemoryRouter + Probe: los setters se disparan por click y la URL se
 * asserta vía router.state.location.search (y navigate(-1) para distinguir
 * push de replace).
 */
import { describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';

import { useCommandUrlState } from '../useCommandUrlState';

function Probe() {
  const s = useCommandUrlState();
  return (
    <div>
      <span data-testid="mode">{s.mode}</span>
      <span data-testid="dia">{s.dia}</span>
      <span data-testid="t">{s.tRaw}</span>
      <span data-testid="nodo">{s.nodo ?? '∅'}</span>
      <span data-testid="panel">{s.panel ?? '∅'}</span>
      <button onClick={() => s.setMode('historico')}>modo-historico</button>
      <button onClick={() => s.setMode('ahora')}>modo-ahora</button>
      <button onClick={() => s.setDia('2026-06-05')}>set-dia</button>
      <button onClick={() => s.setT(9)}>set-t</button>
      <button onClick={() => s.openNode('cam_larco_benavides')}>open-nodo</button>
      <button onClick={() => s.closeNode()}>close-nodo</button>
      <button onClick={() => s.clearPanel()}>clear-panel</button>
    </div>
  );
}

function mount(initialEntry: string) {
  const router = createMemoryRouter([{ path: '/', element: <Probe /> }], {
    initialEntries: [initialEntry],
  });
  render(<RouterProvider router={router} />);
  return router;
}

describe('useCommandUrlState — lectura', () => {
  it('defaults sin params: ahora, hoy, t 0, sin nodo ni panel', () => {
    mount('/');
    expect(screen.getByTestId('mode')).toHaveTextContent('ahora');
    expect(screen.getByTestId('dia').textContent).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(screen.getByTestId('t')).toHaveTextContent('0');
    expect(screen.getByTestId('nodo')).toHaveTextContent('∅');
    expect(screen.getByTestId('panel')).toHaveTextContent('∅');
  });

  it('params válidos se leen tal cual', () => {
    mount('/?modo=historico&dia=2026-06-05&t=7&nodo=cam_x&panel=alertas');
    expect(screen.getByTestId('mode')).toHaveTextContent('historico');
    expect(screen.getByTestId('dia')).toHaveTextContent('2026-06-05');
    expect(screen.getByTestId('t')).toHaveTextContent('7');
    expect(screen.getByTestId('nodo')).toHaveTextContent('cam_x');
    expect(screen.getByTestId('panel')).toHaveTextContent('alertas');
  });

  it('basura se LEE corregida sin reescribir la URL: modo→ahora, t<0→0, dia inválido→hoy', () => {
    const router = mount('/?modo=basura&t=-3&dia=ayer');
    expect(screen.getByTestId('mode')).toHaveTextContent('ahora');
    expect(screen.getByTestId('t')).toHaveTextContent('0');
    expect(screen.getByTestId('dia').textContent).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    // Anti-loop: la URL sucia queda estable, no hay write-back.
    expect(router.state.location.search).toBe('?modo=basura&t=-3&dia=ayer');
  });
});

describe('useCommandUrlState — escritura', () => {
  it('setMode escribe modo; volver a ahora limpia modo/dia/t (URL canónica)', () => {
    const router = mount('/?modo=historico&dia=2026-06-05&t=7');
    fireEvent.click(screen.getByText('modo-ahora'));
    expect(router.state.location.search).toBe('');

    fireEvent.click(screen.getByText('modo-historico'));
    expect(router.state.location.search).toBe('?modo=historico');
  });

  it('setDia escribe el día y borra t', () => {
    const router = mount('/?modo=historico&dia=2026-06-01&t=5');
    fireEvent.click(screen.getByText('set-dia'));
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get('dia')).toBe('2026-06-05');
    expect(params.get('t')).toBeNull();
  });

  it('setT usa replace: back NO deshace pasos del slider (vuelve antes del push previo)', async () => {
    const router = mount('/?modo=historico&dia=2026-06-05');
    // push (openNode) → replace (setT): back debe saltar a ANTES del push,
    // sin pasar por un entry intermedio con t.
    fireEvent.click(screen.getByText('open-nodo'));
    fireEvent.click(screen.getByText('set-t'));
    let params = new URLSearchParams(router.state.location.search);
    expect(params.get('t')).toBe('9');
    expect(params.get('nodo')).toBe('cam_larco_benavides');

    await act(async () => {
      await router.navigate(-1);
    });
    params = new URLSearchParams(router.state.location.search);
    expect(params.get('nodo')).toBeNull();
    expect(params.get('t')).toBeNull();
  });

  it('openNode/closeNode conservan los params concurrentes', () => {
    const router = mount('/?modo=historico&dia=2026-06-05&t=3');
    fireEvent.click(screen.getByText('open-nodo'));
    let params = new URLSearchParams(router.state.location.search);
    expect(params.get('nodo')).toBe('cam_larco_benavides');
    expect(params.get('modo')).toBe('historico');
    expect(params.get('t')).toBe('3');

    fireEvent.click(screen.getByText('close-nodo'));
    params = new URLSearchParams(router.state.location.search);
    expect(params.get('nodo')).toBeNull();
    expect(params.get('modo')).toBe('historico');
  });

  it('clearPanel borra el param (one-shot)', () => {
    const router = mount('/?panel=alertas&modo=prediccion');
    fireEvent.click(screen.getByText('clear-panel'));
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get('panel')).toBeNull();
    expect(params.get('modo')).toBe('prediccion');
  });
});
