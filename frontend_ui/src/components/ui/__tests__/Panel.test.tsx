import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { Panel } from '../Panel';

describe('Panel', () => {
  it('renderiza título, contador, mini y children', () => {
    render(
      <Panel title="Alertas priorizadas por IA" count={3} mini="orden: impacto × confianza">
        <p>contenido</p>
      </Panel>,
    );
    expect(screen.getByText('Alertas priorizadas por IA')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('orden: impacto × confianza')).toBeInTheDocument();
    expect(screen.getByText('contenido')).toBeInTheDocument();
  });

  it('sin count no renderiza contador', () => {
    render(
      <Panel title="Panel">
        <p>x</p>
      </Panel>,
    );
    expect(screen.queryByText('0')).not.toBeInTheDocument();
  });
});
