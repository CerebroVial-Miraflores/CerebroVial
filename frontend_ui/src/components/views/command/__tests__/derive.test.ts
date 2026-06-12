/**
 * Tests de las derivaciones puras del Centro de Comando (FASE 3).
 * Las fórmulas de KPIs y la key del GeoJSON son CONTRATO de la vista.
 */
import { describe, expect, it } from 'vitest';

import {
  adaptActiveState,
  clampIndex,
  drawerStatusFor,
  edgeLevelForNode,
  featuresForMode,
  formatHourLabel,
  geoJsonKeyFor,
  legendForMode,
  markersFrom,
  networkCongestionIndex,
  nodeStatusFor,
  predictionLabelAt,
  pushSample,
  seriesToNetworkIndex,
} from '../derive';
import type {
  CongestionPredictionResponse,
  CongestionSeriesResponse,
  CongestionStateResponse,
  GeometryFeatureCollection,
} from '../../../../types/congestion';
import type { ActiveStateResponse } from '../../../../services/controlActiveStateService';

function geometry(
  edges: { id: string; source: string; target: string }[],
): GeometryFeatureCollection {
  return {
    type: 'FeatureCollection',
    count: edges.length,
    features: edges.map((e) => ({
      type: 'Feature' as const,
      geometry: { type: 'LineString' as const, coordinates: [[0, 0], [1, 1]] as [number, number][] },
      properties: {
        edge_id: e.id,
        source_node: e.source,
        target_node: e.target,
        distance_m: 100,
        lanes: 2,
      },
    })),
  };
}

function state(levels: Record<string, number>): CongestionStateResponse {
  const edges = Object.entries(levels).map(([edge_id, congestion_level]) => ({
    edge_id,
    congestion_level,
    snapshot_timestamp: '2026-06-08T12:00:00',
  }));
  return { edges, count: edges.length };
}

const GEO = geometry([
  { id: 'e1', source: 'n1', target: 'n2' },
  { id: 'e2', source: 'n2', target: 'n3' },
  { id: 'e3', source: 'n4', target: 'n5' },
]);

describe('networkCongestionIndex', () => {
  it('media de niveles escalada a /100 y redondeada', () => {
    // (2+3)/2 = 2.5 → 2.5/5*100 = 50
    expect(networkCongestionIndex(state({ e1: 2, e2: 3 }))).toBe(50);
    expect(networkCongestionIndex(state({ e1: 5 }))).toBe(100);
    expect(networkCongestionIndex(state({ e1: 0 }))).toBe(0);
  });

  it('estado vacío o null → null (ventana sin datos)', () => {
    expect(networkCongestionIndex(state({}))).toBeNull();
    expect(networkCongestionIndex(null)).toBeNull();
  });
});

describe('nodeStatusFor / markersFrom', () => {
  it('fluid→ok, moderate→warn, critical→bad, desconocido→ok', () => {
    expect(nodeStatusFor('fluid')).toBe('ok');
    expect(nodeStatusFor('moderate')).toBe('warn');
    expect(nodeStatusFor('critical')).toBe('bad');
    expect(nodeStatusFor('lo-que-sea')).toBe('ok');
  });

  it('markersFrom proyecta posición [lat,lng] y critical', () => {
    const markers = markersFrom([
      {
        id: 'cam_larco_benavides',
        name: 'Larco × Benavides',
        speed: 0,
        flow: 0,
        status: 'critical',
        lat: -12.13,
        lng: -77.02,
        stream_url: null,
      },
    ]);
    expect(markers).toEqual([
      {
        id: 'cam_larco_benavides',
        name: 'Larco × Benavides',
        position: [-12.13, -77.02],
        status: 'bad',
        critical: true,
      },
    ]);
    expect(markersFrom(null)).toEqual([]);
  });
});

describe('clampIndex', () => {
  it('clampea a [0, len-1]; len 0 → 0; valores no finitos → 0', () => {
    expect(clampIndex(5, 10)).toBe(5);
    expect(clampIndex(999, 10)).toBe(9);
    expect(clampIndex(-3, 10)).toBe(0);
    expect(clampIndex(NaN, 10)).toBe(0);
    expect(clampIndex(4, 0)).toBe(0);
  });
});

describe('predictionLabelAt', () => {
  const prediction: CongestionPredictionResponse = {
    base_timestep: 720,
    horizon: 30,
    source: 'seed051 (day_idx=9)',
    source_date: '2026-06-05',
    count: 2,
    edges: [
      { edge_id: 'e1', levels: Array(30).fill(3) },
      { edge_id: 'e2', levels: Array(30).fill(4) },
    ],
  };

  it('+15 usa el índice 14 y etiqueta la media redondeada', () => {
    // media 3.5 → round 4 → 'Demora severa'
    expect(predictionLabelAt(prediction, 15)).toBe('Demora severa');
  });

  it('+30 usa el último paso disponible', () => {
    expect(predictionLabelAt(prediction, 30)).toBe('Demora severa');
  });

  it('sin data o sin aristas → null', () => {
    expect(predictionLabelAt(null, 15)).toBeNull();
    expect(predictionLabelAt({ ...prediction, edges: [] }, 15)).toBeNull();
  });
});

describe('featuresForMode', () => {
  const series: CongestionSeriesResponse = {
    day: '2026-06-05',
    t0: '2026-06-05T00:00:00',
    step_s: 60,
    coverage_end: '2026-06-05T23:59:00',
    count: 2,
    edges: [
      { edge_id: 'e1', levels: [0, 1, 2] },
      { edge_id: 'e2', levels: [3, 4, 5] },
    ],
  };

  it('ahora: merge del estado vigente; sin state → null', () => {
    const features = featuresForMode({
      mode: 'ahora',
      geometry: GEO,
      state: state({ e1: 4 }),
      series: null,
      tClamped: 0,
      prediction: null,
      horizon: 15,
    });
    expect(features?.find((f) => f.properties.edge_id === 'e1')?.properties.congestion_level).toBe(4);
    expect(features?.find((f) => f.properties.edge_id === 'e2')?.properties.congestion_level).toBeNull();

    expect(
      featuresForMode({
        mode: 'ahora',
        geometry: GEO,
        state: null,
        series: null,
        tClamped: 0,
        prediction: null,
        horizon: 15,
      }),
    ).toBeNull();
  });

  it('historico: merge por índice; día vacío (t0 null) → null', () => {
    const features = featuresForMode({
      mode: 'historico',
      geometry: GEO,
      state: null,
      series,
      tClamped: 2,
      prediction: null,
      horizon: 15,
    });
    expect(features?.find((f) => f.properties.edge_id === 'e2')?.properties.congestion_level).toBe(5);

    expect(
      featuresForMode({
        mode: 'historico',
        geometry: GEO,
        state: null,
        series: { ...series, t0: null },
        tClamped: 0,
        prediction: null,
        horizon: 15,
      }),
    ).toBeNull();
  });

  it('prediccion: usa el paso del horizonte clampeado al horizon real', () => {
    const prediction: CongestionPredictionResponse = {
      base_timestep: 720,
      horizon: 30,
      source: 's',
      source_date: '2026-06-05',
      count: 1,
      edges: [{ edge_id: 'e1', levels: Array.from({ length: 30 }, (_, i) => (i === 29 ? 4 : 1)) }],
    };
    const features = featuresForMode({
      mode: 'prediccion',
      geometry: GEO,
      state: null,
      series: null,
      tClamped: 0,
      prediction,
      horizon: 30,
    });
    expect(features?.find((f) => f.properties.edge_id === 'e1')?.properties.congestion_level).toBe(4);
  });

  it('sin geometry → null en cualquier modo', () => {
    expect(
      featuresForMode({
        mode: 'ahora',
        geometry: null,
        state: state({ e1: 1 }),
        series: null,
        tClamped: 0,
        prediction: null,
        horizon: 15,
      }),
    ).toBeNull();
  });
});

describe('geoJsonKeyFor (contrato del remount)', () => {
  const parts = { lastUpdated: 1700000000000, dia: '2026-06-05', tClamped: 7, horizon: 30 as const, baseTimestep: 720 };

  it('las tres formas', () => {
    expect(geoJsonKeyFor('ahora', parts)).toBe('live-1700000000000');
    expect(geoJsonKeyFor('ahora', { ...parts, lastUpdated: null })).toBe('live-init');
    expect(geoJsonKeyFor('historico', parts)).toBe('h-2026-06-05-7');
    expect(geoJsonKeyFor('prediccion', parts)).toBe('p-30-720');
    expect(geoJsonKeyFor('prediccion', { ...parts, baseTimestep: null })).toBe('p-30-na');
  });
});

describe('formatHourLabel (contrato de series v1: t0 + i*step_s, naive-UTC)', () => {
  it('parsea t0 naive como UTC sin desfase local', () => {
    expect(formatHourLabel('2026-06-05T08:00:00', 60, 0)).toBe('08:00');
    expect(formatHourLabel('2026-06-05T08:00:00', 60, 90)).toBe('09:30');
    expect(formatHourLabel('2026-06-05T08:00:00Z', 3600, 2)).toBe('10:00');
  });

  it('t0 null → placeholder', () => {
    expect(formatHourLabel(null, 60, 5)).toBe('--:--');
  });
});

describe('seriesToNetworkIndex', () => {
  it('media por timestep escalada /100 + labels', () => {
    const s: CongestionSeriesResponse = {
      day: '2026-06-05',
      t0: '2026-06-05T00:00:00',
      step_s: 60,
      coverage_end: null,
      count: 2,
      edges: [
        { edge_id: 'e1', levels: [0, 5] },
        { edge_id: 'e2', levels: [0, 5] },
      ],
    };
    const result = seriesToNetworkIndex(s);
    expect(result?.points).toEqual([0, 100]);
    expect(result?.tLabels).toEqual(['00:00', '00:01']);
  });

  it('día vacío (t0 null) o sin aristas → null', () => {
    expect(seriesToNetworkIndex(null)).toBeNull();
    expect(
      seriesToNetworkIndex({
        day: 'x',
        t0: null,
        step_s: null,
        coverage_end: null,
        count: 0,
        edges: [],
      }),
    ).toBeNull();
  });
});

describe('edgeLevelForNode', () => {
  it('MAX nivel entre aristas que tocan el nodo con dato en state', () => {
    expect(edgeLevelForNode(GEO, state({ e1: 2, e2: 4 }), 'n2')).toBe(4);
    expect(edgeLevelForNode(GEO, state({ e1: 2 }), 'n1')).toBe(2);
  });

  it('nodo sin aristas con dato → null', () => {
    expect(edgeLevelForNode(GEO, state({ e1: 2 }), 'n4')).toBeNull();
    expect(edgeLevelForNode(GEO, state({}), 'n1')).toBeNull();
    expect(edgeLevelForNode(null, state({ e1: 1 }), 'n1')).toBeNull();
    expect(edgeLevelForNode(GEO, state({ e1: 1 }), '')).toBeNull();
  });
});

describe('adaptActiveState (HU-05 → TrafficLightCycle)', () => {
  it('proyecta el shape exacto de ControlRecommendation', () => {
    const active: ActiveStateResponse = {
      node_id: 'larco_schell',
      strategy_mode: 'webster',
      cycle_seconds: 90,
      phase_timings: [{ phase_id: 'NS', green: 42, yellow: 4, all_red: 2 }],
      decided_at: '2026-06-10T12:00:00',
      activated_at: '2026-06-10T12:00:05',
      activated_by: null,
    };
    expect(adaptActiveState(active)).toEqual({
      intersection_id: 'larco_schell',
      mode: 'webster',
      cycle_seconds: 90,
      phase_timings: active.phase_timings,
      next_phase: null,
      reasoning: '',
      adjustments: [],
    });
  });
});

describe('drawerStatusFor', () => {
  it('mapea status real y prioriza el estado demo aplicado', () => {
    expect(drawerStatusFor('critical', false)).toEqual({
      status: 'bad',
      label: 'CRÍTICO · ADAPTATIVO EN PAUSA',
    });
    expect(drawerStatusFor('moderate', false)).toEqual({ status: 'warn', label: 'EN OBSERVACIÓN' });
    expect(drawerStatusFor('fluid', false)).toEqual({ status: 'ok', label: 'ADAPTATIVO · NORMAL' });
    expect(drawerStatusFor('critical', true)).toEqual({ status: 'recovery', label: 'EN RECUPERACIÓN' });
  });
});

describe('pushSample', () => {
  it('appendea no-null, ignora null y recorta al cap', () => {
    expect(pushSample([1, 2], 3)).toEqual([1, 2, 3]);
    const history = [1, 2];
    expect(pushSample(history, null)).toBe(history);
    expect(pushSample([1, 2, 3], 4, 3)).toEqual([2, 3, 4]);
  });
});

describe('legendForMode', () => {
  it('título y escala por modo', () => {
    expect(legendForMode('ahora', '').title).toBe('Observado');
    expect(legendForMode('historico', '16:00').title).toBe('Histórico · 16:00');
    const pred = legendForMode('prediccion', '');
    expect(pred.title).toBe('Demora prevista');
    expect(pred.items.some((i) => i.swatchClass === 'bg-sev')).toBe(false);
  });
});
