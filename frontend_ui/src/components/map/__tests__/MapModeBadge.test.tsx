import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { MapModeBadge } from '../MapModeBadge';

describe('MapModeBadge', () => {
  it('live: EN VIVO neutro', () => {
    render(<MapModeBadge mode="live" />);
    expect(screen.getByText('EN VIVO')).toHaveClass('border-line-2');
  });

  it('prediction: horizonte en minutos con paleta warn', () => {
    render(<MapModeBadge mode="prediction" offsetMin={30} />);
    expect(screen.getByText('PREDICCIÓN · +30 MIN')).toHaveClass('text-warn');
  });

  it('historic: etiqueta temporal con paleta info', () => {
    render(<MapModeBadge mode="historic" timestampLabel="hoy 16:00" />);
    expect(screen.getByText('HISTÓRICO · hoy 16:00')).toHaveClass('text-info');
  });
});
