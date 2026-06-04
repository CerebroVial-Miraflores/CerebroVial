/**
 * CongestionMapView (Fase 4, Gate 3b) — controles del modo predicción + base coherente.
 *
 * Archivo SEPARADO del smoke vivo (`*.test.tsx`) y del histórico (`*.historic.test.tsx`)
 * a propósito: predicción no debe afectar a esos modos. Calca su patrón de mock de
 * react-leaflet, pero captura DOS capas `<GeoJSON>` (la base observada y la de
 * predicción superpuesta). Como React no pasa `key` como prop, se distinguen por la
 * presencia de `onEachFeature`: solo la base lo lleva (CA-22.5); la predicción no
 * (es `interactive: false`).
 *
 * Afirma (3a + 3b): la capa de predicción pinta el horizonte elegido con `predictionStyle`
 * (0 no pinta); las dos capas COEXISTEN (un wake SSE remonta la base sin remontar la
 * predicción); el horizonte reindexa LOCAL sin re-pedir; el corte 'pinned' trae la base
 * coherente del día-fuente (getSeries(source_date)), con debounce 2 s, salto sincronizado
 * base↔predicción, SSE cortado, una sola base montada; y los errores del endpoint muestran
 * su mensaje sin romper la base. Más la leyenda de predicción separada de la observada.
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor, act, fireEvent } from '@testing-library/react';
import { CongestionMapView } from '../CongestionMapView';
import { congestionService } from '../../../services/congestionService';
import { openCongestionStream } from '../../../services/congestionSseClient';
import type {
  GeometryFeatureCollection,
  CongestionStateResponse,
  CongestionSeriesResponse,
  CongestionPredictionResponse,
} from '../../../types/congestion';

// Dos slots: la base (con onEachFeature) y la predicción (sin él), cada una con su
// contador de montajes para detectar remontes independientes por `key`.
const captured = vi.hoisted(() => ({
  base: null as null | Record<string, unknown>,
  baseMounts: 0,
  prediction: null as null | Record<string, unknown>,
  predictionMounts: 0,
}));

const sse = vi.hoisted(() => ({
  onWake: null as null | (() => void),
  abort: vi.fn(),
}));

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
  GeoJSON: (props: Record<string, unknown>) => {
    const isBase = typeof props.onEachFeature === 'function';
    if (isBase) captured.base = props;
    else captured.prediction = props;
    React.useEffect(() => {
      if (isBase) captured.baseMounts += 1;
      else captured.predictionMounts += 1;
    }, [isBase]);
    return <div data-testid={isBase ? 'geojson-base' : 'geojson-prediction'} />;
  },
}));

vi.mock('../../../services/congestionService', () => ({
  congestionService: {
    getGeometry: vi.fn(),
    getState: vi.fn(),
    getSeries: vi.fn(),
    getPrediction: vi.fn(),
  },
}));

vi.mock('../../../services/congestionSseClient', () => ({
  openCongestionStream: vi.fn((opts: { onWake: () => void }) => {
    sse.onWake = opts.onWake;
    return { abort: sse.abort };
  }),
}));

const getGeometryMock = congestionService.getGeometry as unknown as ReturnType<typeof vi.fn>;
const getStateMock = congestionService.getState as unknown as ReturnType<typeof vi.fn>;
const getSeriesMock = congestionService.getSeries as unknown as ReturnType<typeof vi.fn>;
const getPredictionMock = congestionService.getPrediction as unknown as ReturnType<typeof vi.fn>;
const openStreamMock = openCongestionStream as unknown as ReturnType<typeof vi.fn>;

const COUNT = 1660;
const HORIZON = 30;
const SOURCE_DATE = '2026-06-08';
// Corte sembrado al entrar a pinned = base_timestep. No múltiplo de 6 → la base
// reindexada difiere del estado vivo (edge-i: vivo = i%6; serie en t=601 = (i+601)%6).
const BASE_T = 601;

function buildGeometry(): GeometryFeatureCollection {
  return {
    type: 'FeatureCollection',
    count: COUNT,
    features: Array.from({ length: COUNT }, (_, i) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'LineString' as const,
        coordinates: [
          [-77.0335, -12.118],
          [-77.0334, -12.1179],
        ] as [number, number][],
      },
      properties: {
        edge_id: `edge-${i}`,
        source_node: `src-${i}`,
        target_node: `dst-${i}`,
        distance_m: 100,
        lanes: 1,
      },
    })),
  };
}

// Estado vivo (observado "now"): edge-i → i % 6.
function buildState(): CongestionStateResponse {
  return {
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      congestion_level: i % 6,
      snapshot_timestamp: '2025-01-06T23:59:00',
    })),
  };
}

/** Predicción a 30 pasos: edge-i → levels[k] = (i+k) % 5 (escala 0-4). En el horizonte
 *  16 (índice 15), edge-0 → (0+15)%5 = 0; edge-2 → (2+15)%5 = 2 (niveles conocidos). */
function buildPrediction(): CongestionPredictionResponse {
  return {
    base_timestep: BASE_T,
    horizon: HORIZON,
    source: 'seed051 (day_idx=9)',
    source_date: SOURCE_DATE,
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      levels: Array.from({ length: HORIZON }, (_, k) => (i + k) % 5),
    })),
  };
}

/** Predicción alternativa (todos los pasos = 3): para distinguir un re-fetch del corte. */
function buildPredictionAlt(): CongestionPredictionResponse {
  return {
    ...buildPrediction(),
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      levels: Array.from({ length: HORIZON }, () => 3),
    })),
  };
}

/** Serie observada del día-fuente: t0 = 00:00, paso 60 s → índice = minuto del día.
 *  edge-i en el minuto m → (i + m) % 6 (escala observada 0-5). */
function buildSeries(): CongestionSeriesResponse {
  const LEN = 1440;
  return {
    day: SOURCE_DATE,
    t0: `${SOURCE_DATE}T00:00:00`,
    step_s: 60,
    coverage_end: `${SOURCE_DATE}T23:59:00`,
    count: COUNT,
    edges: Array.from({ length: COUNT }, (_, i) => ({
      edge_id: `edge-${i}`,
      levels: Array.from({ length: LEN }, (_, m) => (i + m) % 6),
    })),
  };
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

type CapturedData = { data: { features: { properties: { congestion_level: number } }[] } };
type StyleFn = (f: unknown) => {
  color: string;
  weight: number;
  opacity?: number;
  interactive?: boolean;
};

async function renderAndLoad() {
  render(<CongestionMapView />);
  await waitFor(() => expect(screen.getByTestId('geojson-base')).toBeInTheDocument());
}

function clickPrediction() {
  fireEvent.click(screen.getByRole('button', { name: /predicción/i }));
}
function clickMomento() {
  fireEvent.click(screen.getByRole('button', { name: /momento elegido/i }));
}
function clickAhora() {
  fireEvent.click(screen.getByRole('button', { name: /ahora/i }));
}
function setHorizon(value: number) {
  fireEvent.change(screen.getByLabelText(/horizonte de predicción/i), {
    target: { value: String(value) },
  });
}

describe('CongestionMapView — modo predicción (Fase 4 Gate 3b)', () => {
  beforeEach(() => {
    getGeometryMock.mockReset();
    getStateMock.mockReset();
    getSeriesMock.mockReset();
    getPredictionMock.mockReset();
    openStreamMock.mockClear();
    sse.onWake = null;
    sse.abort.mockClear();
    captured.base = null;
    captured.baseMounts = 0;
    captured.prediction = null;
    captured.predictionMounts = 0;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('horizonte 0 no pinta; al mover a 16 pinta el índice 15 con predictionStyle', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    // Al entrar, horizonte = 0 → la capa de predicción NO se monta (solo observado).
    expect(screen.queryByTestId('geojson-prediction')).not.toBeInTheDocument();

    // Mover el horizonte a 16 → pinta levels[15].
    await act(async () => {
      setHorizon(16);
    });
    await waitFor(() =>
      expect(screen.getByTestId('geojson-prediction')).toBeInTheDocument(),
    );
    const pred = captured.prediction as CapturedData;
    expect(pred.data.features[0].properties.congestion_level).toBe(0); // (0+15)%5
    expect(pred.data.features[2].properties.congestion_level).toBe(2); // (2+15)%5

    const style = (captured.prediction as { style: StyleFn }).style;
    const s2 = style({ properties: { congestion_level: 2 } });
    expect(s2.color).toBe('#818CF8'); // nivel 2 de la escala de predicción
    expect(s2.opacity).toBe(0.55);
    expect(s2.interactive).toBe(false);
    expect(style({ properties: { congestion_level: null } }).opacity).toBe(0);
  });

  it('mover el horizonte reindexa LOCAL: remonta la predicción sin re-pedir al backend', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    await waitFor(() => expect(getPredictionMock).toHaveBeenCalledTimes(1));

    await act(async () => {
      setHorizon(1);
    });
    const predBefore = captured.predictionMounts;
    const baseBefore = captured.baseMounts;
    await act(async () => {
      setHorizon(2);
    });
    await act(async () => {
      setHorizon(3);
    });

    // Reindexado local: ni una llamada más a getPrediction.
    expect(getPredictionMock).toHaveBeenCalledTimes(1);
    // La capa de predicción remonta por su key (horizonte); la base NO se toca.
    expect(captured.predictionMounts).toBeGreaterThan(predBefore);
    expect(captured.baseMounts).toBe(baseBefore);
    const pred = captured.prediction as CapturedData;
    expect(pred.data.features[0].properties.congestion_level).toBe(2); // horizonte 3 → levels[2] = (0+2)%5
  });

  it('las dos capas coexisten: un wake SSE remonta la BASE sin remontar la predicción', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    await act(async () => {
      setHorizon(16);
    });
    await waitFor(() =>
      expect(screen.getByTestId('geojson-prediction')).toBeInTheDocument(),
    );

    const baseBefore = captured.baseMounts;
    const predBefore = captured.predictionMounts;

    await act(async () => {
      sse.onWake?.();
    });
    await waitFor(() => expect(captured.baseMounts).toBeGreaterThan(baseBefore));
    expect(captured.predictionMounts).toBe(predBefore);
  });

  it('pinned: trae la base coherente del día-fuente (getSeries(source_date)) reindexada en el corte', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockResolvedValue(buildPrediction());
    getSeriesMock.mockResolvedValue(buildSeries());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    await waitFor(() => expect(getPredictionMock).toHaveBeenCalled());

    // Entrar a pinned: corte sembrado = base_timestep (601).
    await act(async () => {
      clickMomento();
    });
    // getSeries del día-fuente, una sola vez.
    await waitFor(() => expect(getSeriesMock).toHaveBeenCalledWith(SOURCE_DATE));
    expect(getSeriesMock).toHaveBeenCalledTimes(1);

    // La base se recolorea por el observado del día-fuente en t=601: edge-i → (i+601)%6.
    await waitFor(() => {
      const base = captured.base as CapturedData;
      expect(base.data.features[0].properties.congestion_level).toBe(BASE_T % 6); // (0+601)%6 = 1
    });
    const base = captured.base as CapturedData;
    expect(base.data.features[3].properties.congestion_level).toBe((3 + BASE_T) % 6);
    // El corte pidió la predicción con t fijado.
    expect(getPredictionMock).toHaveBeenLastCalledWith(BASE_T);
  });

  it('pinned: base y predicción saltan JUNTAS — la base no se adelanta al fetch', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());
    // 1ª llamada (driver now) resuelve; 2ª (driver pinned) queda pendiente y la controlamos.
    const d = deferred<CongestionPredictionResponse>();
    getPredictionMock.mockResolvedValueOnce(buildPrediction());
    getPredictionMock.mockReturnValueOnce(d.promise);

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    await act(async () => {
      setHorizon(1); // montar la capa de predicción (levels[0])
    });
    await waitFor(() =>
      expect(screen.getByTestId('geojson-prediction')).toBeInTheDocument(),
    );

    // Entrar a pinned: getSeries resuelve, getPrediction(601) queda PENDIENTE.
    await act(async () => {
      clickMomento();
    });
    await waitFor(() => expect(getSeriesMock).toHaveBeenCalled());

    // Con el fetch pendiente: la base NO se movió (sigue el observado vivo, edge-0 = 0) y
    // la predicción sigue la vieja (now: levels[0] de edge-0 = 0).
    const baseMid = captured.base as CapturedData;
    expect(baseMid.data.features[0].properties.congestion_level).toBe(0);
    const predMid = captured.prediction as CapturedData;
    expect(predMid.data.features[0].properties.congestion_level).toBe(0);
    expect(screen.getByText(/cargando predicción/i)).toBeInTheDocument();

    // Resolver: base (serie en 601 → 1) y predicción (alt → 3) conmutan en el mismo render.
    await act(async () => {
      d.resolve(buildPredictionAlt());
    });
    await waitFor(() => {
      const base = captured.base as CapturedData;
      expect(base.data.features[0].properties.congestion_level).toBe(BASE_T % 6); // 1
    });
    const pred = captured.prediction as CapturedData;
    expect(pred.data.features[0].properties.congestion_level).toBe(3);
  });

  it('pinned: mover el corte no re-pide de inmediato; tras 2 s pide getPrediction(t) una sola vez', async () => {
    vi.useFakeTimers();
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());
    getPredictionMock.mockResolvedValue(buildPrediction());

    const flush = async (ms = 0) => {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(ms);
      });
    };

    render(<CongestionMapView />);
    await flush(); // carga inicial geometry+state
    await act(async () => {
      clickPrediction();
    });
    await flush(); // driver now
    await act(async () => {
      clickMomento();
    });
    await flush(); // getSeries + driver pinned (t = base_timestep)
    getPredictionMock.mockClear();

    const cut = screen.getByLabelText(/momento del día/i);
    await act(async () => {
      fireEvent.change(cut, { target: { value: '700' } });
      fireEvent.change(cut, { target: { value: '800' } });
      fireEvent.change(cut, { target: { value: '900' } });
    });
    // Antes de los 2 s: nada.
    expect(getPredictionMock).not.toHaveBeenCalled();
    await flush(1999);
    expect(getPredictionMock).not.toHaveBeenCalled();
    // Cruzar los 2 s de quietud: una sola llamada, con el último t.
    await flush(2);
    expect(getPredictionMock).toHaveBeenCalledTimes(1);
    expect(getPredictionMock).toHaveBeenCalledWith(900);
  });

  it('mientras carga el corte: horizonte deshabilitado, "cargando" y la capa vieja sigue montada', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());
    const d = deferred<CongestionPredictionResponse>();
    getPredictionMock.mockResolvedValueOnce(buildPrediction());
    getPredictionMock.mockReturnValueOnce(d.promise);

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    await act(async () => {
      setHorizon(16);
    });
    await waitFor(() =>
      expect(screen.getByTestId('geojson-prediction')).toBeInTheDocument(),
    );

    await act(async () => {
      clickMomento();
    });
    await waitFor(() => expect(getSeriesMock).toHaveBeenCalled());

    // Fetch del corte pendiente: horizonte deshabilitado + "cargando" + capa vieja montada.
    expect(screen.getByLabelText(/horizonte de predicción/i)).toBeDisabled();
    expect(screen.getByText(/cargando predicción/i)).toBeInTheDocument();
    expect(screen.getByTestId('geojson-prediction')).toBeInTheDocument();

    await act(async () => {
      d.resolve(buildPredictionAlt());
    });
    await waitFor(() =>
      expect(screen.getByLabelText(/horizonte de predicción/i)).not.toBeDisabled(),
    );
  });

  it('una sola capa base montada, tanto en now como en pinned', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    expect(screen.getAllByTestId('geojson-base')).toHaveLength(1);

    await act(async () => {
      clickMomento();
    });
    await waitFor(() => expect(getSeriesMock).toHaveBeenCalled());
    expect(screen.getAllByTestId('geojson-base')).toHaveLength(1);
  });

  it('pinned corta el SSE; volver a now lo reabre y re-pide getPrediction() sin t + getState', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getSeriesMock.mockResolvedValue(buildSeries());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });
    await waitFor(() => expect(getPredictionMock).toHaveBeenCalled());
    const streamsAfterNow = openStreamMock.mock.calls.length;

    // Entrar a pinned: no se abre un stream nuevo (SSE cortado).
    await act(async () => {
      clickMomento();
    });
    await waitFor(() => expect(getSeriesMock).toHaveBeenCalled());
    expect(openStreamMock.mock.calls.length).toBe(streamsAfterNow);

    // En pinned, un wake viejo (stream cancelado) no remonta la base.
    const baseBefore = captured.baseMounts;
    await act(async () => {
      sse.onWake?.();
    });
    expect(captured.baseMounts).toBe(baseBefore);

    // Volver a now: reabre el stream, re-pide getState y getPrediction() SIN t.
    getStateMock.mockClear();
    await act(async () => {
      clickAhora();
    });
    await waitFor(() =>
      expect(openStreamMock.mock.calls.length).toBeGreaterThan(streamsAfterNow),
    );
    await waitFor(() => expect(getStateMock).toHaveBeenCalled());
    expect(getPredictionMock).toHaveBeenLastCalledWith();
  });

  it('leyenda de predicción ("Predicción (demora)") coexiste con la observada', async () => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockResolvedValue(buildPrediction());

    await renderAndLoad();
    // Fuera de predicción: solo la observada.
    expect(screen.getByText('Observado')).toBeInTheDocument();
    expect(screen.queryByText(/predicción \(demora\)/i)).not.toBeInTheDocument();

    await act(async () => {
      clickPrediction();
    });
    // En predicción: conviven ambas leyendas + las 5 etiquetas de demora.
    expect(screen.getByText('Observado')).toBeInTheDocument();
    expect(screen.getByText(/predicción \(demora\)/i)).toBeInTheDocument();
    expect(screen.getByText(/sin demora/i)).toBeInTheDocument();
    expect(screen.getByText(/demora severa/i)).toBeInTheDocument();
  });

  it.each([
    [503, /no está disponible/i],
    [409, /suficiente historia/i],
    [422, /fuera de rango/i],
  ])('error %i del endpoint: muestra su mensaje sin romper la vista ni la base', async (status, re) => {
    getGeometryMock.mockResolvedValue(buildGeometry());
    getStateMock.mockResolvedValue(buildState());
    getPredictionMock.mockRejectedValue({ response: { status } });

    await renderAndLoad();
    await act(async () => {
      clickPrediction();
    });

    await waitFor(() => expect(screen.getByText(re)).toBeInTheDocument());
    expect(screen.getByTestId('geojson-base')).toBeInTheDocument();
    expect(screen.queryByTestId('geojson-prediction')).not.toBeInTheDocument();
  });
});
