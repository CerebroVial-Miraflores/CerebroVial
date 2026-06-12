import { describe, expect, it } from 'vitest';

import { acquireBodyLock } from '../bodyLock';

describe('acquireBodyLock', () => {
  it('refcount: el lock se libera recién con el último release', () => {
    const release1 = acquireBodyLock();
    const release2 = acquireBodyLock();
    expect(document.body).toHaveClass('overflow-hidden');

    release1();
    expect(document.body).toHaveClass('overflow-hidden');

    release2();
    expect(document.body).not.toHaveClass('overflow-hidden');
  });

  it('release doble es idempotente (no descuenta dos veces)', () => {
    const release1 = acquireBodyLock();
    const release2 = acquireBodyLock();
    release1();
    release1();
    expect(document.body).toHaveClass('overflow-hidden');
    release2();
    expect(document.body).not.toHaveClass('overflow-hidden');
  });
});
