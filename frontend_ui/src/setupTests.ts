import '@testing-library/jest-dom';

// Stubs globales para APIs de browser que jsdom no implementa (FASE 0 rediseño UI:
// regla "stubs globales en setupTests.ts" del CLAUDE.md). Se definen con
// configurable + writable para que los tests que ya declaran sus propias versiones
// (useVisionStream, useVisionAggregates, etc.) puedan seguir pisándolos sin
// romper — esos overrides locales quedan como están.

class StubIntersectionObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}

class StubEventSource {
  onerror: unknown = null;
  onmessage: unknown = null;
  onopen: unknown = null;
  addEventListener() {}
  removeEventListener() {}
  close() {}
}

Object.defineProperty(globalThis, 'IntersectionObserver', {
  configurable: true,
  writable: true,
  value: StubIntersectionObserver,
});

Object.defineProperty(globalThis, 'EventSource', {
  configurable: true,
  writable: true,
  value: StubEventSource,
});

// jsdom no implementa window.matchMedia; lo usa prefers-reduced-motion (useCountUp).
// matches: false = sin reducción de movimiento; los tests que necesiten el camino
// reducido lo pisan localmente.
Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
    dispatchEvent: () => false,
  }),
});
