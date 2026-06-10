import '@testing-library/jest-dom';

// Stubs globales para APIs de browser que jsdom no implementa (FASE 0 rediseño UI:
// regla "stubs globales en setupTests.ts" del CLAUDE.md). Se definen con
// configurable + writable para que los tests que ya declaran sus propias versiones
// (CameraStrip, CameraGrid, CameraDetailView, DashboardView.sse) puedan seguir
// pisándolos sin romper — esos overrides locales quedan como están.

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
