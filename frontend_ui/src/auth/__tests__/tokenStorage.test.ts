import { describe, it, expect, beforeEach } from 'vitest';
import { tokenStorage, TOKEN_STORAGE_KEY } from '../tokenStorage';

describe('tokenStorage', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('uses the canonical storage key', () => {
    expect(TOKEN_STORAGE_KEY).toBe('cerebrovial.auth.token');
  });

  it('write stores the token under the canonical key', () => {
    tokenStorage.write('abc.def.ghi');
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBe('abc.def.ghi');
  });

  it('read returns the token previously written', () => {
    tokenStorage.write('abc.def.ghi');
    expect(tokenStorage.read()).toBe('abc.def.ghi');
  });

  it('read returns null when nothing is stored', () => {
    expect(tokenStorage.read()).toBeNull();
  });

  it('clear removes the token', () => {
    tokenStorage.write('abc.def.ghi');
    tokenStorage.clear();
    expect(tokenStorage.read()).toBeNull();
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
  });
});
