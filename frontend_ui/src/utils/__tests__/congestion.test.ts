/**
 * Tests de los helpers puros del mapa de congestión (HU-22, Fase 1) — sin DOM.
 *
 * Cubre la escala visual 0-5 (CA-22.3, incl. el salto de grosor en el nivel 3 y
 * los casos defensivos), el cruce geometry × state por `edge_id` (incl. la arista
 * huérfana), y la detección de staleness con normalización UTC (CA-22.4).
 */
import { describe, it, expect } from 'vitest';
import {
    congestionStyle,
    predictionStyle,
    mergeCongestion,
    mergeCongestionAtIndex,
    elapsedSeconds,
    isStale,
} from '../congestion';
import type {
    GeometryFeature,
    GeometryFeatureCollection,
    CongestionStateResponse,
    CongestionSeriesResponse,
} from '../../types/congestion';

describe('congestionStyle', () => {
    it.each([
        [0, '#2ECC71', 3],
        [1, '#A3D83A', 3.5],
        [2, '#F4C20D', 4],
        [3, '#F08C1D', 6],
        [4, '#E24B4A', 7.5],
        [5, '#8E1B1B', 9],
    ])('level %i → color %s, weight %f', (level, color, weight) => {
        expect(congestionStyle(level)).toEqual({ color, weight });
    });

    it('aplica el salto de grosor deliberado en el nivel 3 (redundancia no-cromática CA-22.3)', () => {
        // El umbral ≥3 salta de 4 px (nivel 2) a 6 px (nivel 3): >50% más grueso.
        expect(congestionStyle(3).weight).toBe(6);
        expect(congestionStyle(3).weight).toBeGreaterThan(congestionStyle(2).weight);
    });

    it.each([
        ['null', null],
        ['undefined', undefined],
        ['NaN', NaN],
        ['negativo', -1],
        ['fuera de rango alto', 6],
        ['no-entero', 3.5],
        ['Infinity', Number.POSITIVE_INFINITY],
    ])('cae a estilo neutro gris para %s', (_label, level) => {
        expect(congestionStyle(level as number)).toEqual({ color: '#9E9E9E', weight: 3 });
    });
});

describe('predictionStyle', () => {
    it.each([
        [0, '#C7D2FE', 3, 0.5],
        [1, '#A5B4FC', 3.5, 0.525],
        [2, '#818CF8', 4.5, 0.55],
        [3, '#6366F1', 6, 0.575],
        [4, '#4C1D95', 7.5, 0.6],
    ])('level %i → color %s, weight %f, opacity %f', (level, color, weight, opacity) => {
        expect(predictionStyle(level)).toEqual({ color, weight, opacity });
    });

    it('conserva la redundancia no-cromática de grosor (creciente, salto en el tramo alto)', () => {
        expect(predictionStyle(4).weight).toBeGreaterThan(predictionStyle(0).weight);
        // salto del nivel 2 (4.5 px) al nivel 3 (6 px), espejo del observado
        expect(predictionStyle(3).weight).toBeGreaterThan(predictionStyle(2).weight);
    });

    it.each([
        ['null', null],
        ['undefined', undefined],
        ['NaN', NaN],
        ['negativo', -1],
        ['fuera de rango alto (5, válido en el observado)', 5],
        ['no-entero', 4.5],
        ['Infinity', Number.POSITIVE_INFINITY],
    ])('cae al neutro transparente (no pinta) para %s', (_label, level) => {
        expect(predictionStyle(level as number)).toEqual({
            color: 'transparent',
            weight: 0,
            opacity: 0,
        });
    });

    it('distingue "nivel 0 válido" (pinta, opacity > 0) de "sin predicción" (null → no pinta)', () => {
        // El horizonte índice 0 se modela como ausencia (null), NO como nivel 0:
        // el merge de Gate 3 decide qué pasar. La función pura ya separa ambos casos.
        expect(predictionStyle(0).opacity).toBeGreaterThan(0);
        expect(predictionStyle(null).opacity).toBe(0);
    });
});

// --- fixtures de cruce ---

function feature(edge_id: string): GeometryFeature {
    return {
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: [[-77.03, -12.11], [-77.02, -12.10]] },
        properties: {
            edge_id,
            source_node: `src_${edge_id}`,
            target_node: `tgt_${edge_id}`,
            distance_m: 100,
            lanes: 2,
        },
    };
}

function geometryOf(...edgeIds: string[]): GeometryFeatureCollection {
    const features = edgeIds.map(feature);
    return { type: 'FeatureCollection', features, count: features.length };
}

describe('mergeCongestion', () => {
    it('adjunta congestion_level y snapshot_timestamp por edge_id', () => {
        const geometry = geometryOf('a', 'b');
        const state: CongestionStateResponse = {
            edges: [
                { edge_id: 'a', congestion_level: 4, snapshot_timestamp: '2025-01-06T23:59:00' },
                { edge_id: 'b', congestion_level: 0, snapshot_timestamp: '2025-01-06T23:59:00' },
            ],
            count: 2,
        };

        const merged = mergeCongestion(geometry, state);

        expect(merged).toHaveLength(2);
        expect(merged[0].properties.congestion_level).toBe(4);
        expect(merged[0].properties.snapshot_timestamp).toBe('2025-01-06T23:59:00');
        expect(merged[1].properties.congestion_level).toBe(0);
    });

    it('es robusto a una arista sin estado: nivel y timestamp null, no lanza', () => {
        const geometry = geometryOf('a', 'huerfana');
        const state: CongestionStateResponse = {
            edges: [{ edge_id: 'a', congestion_level: 2, snapshot_timestamp: '2025-01-06T23:59:00' }],
            count: 1,
        };

        const merged = mergeCongestion(geometry, state);

        expect(merged[1].properties.edge_id).toBe('huerfana');
        expect(merged[1].properties.congestion_level).toBeNull();
        expect(merged[1].properties.snapshot_timestamp).toBeNull();
        // el nivel null cae al estilo neutro
        expect(congestionStyle(merged[1].properties.congestion_level)).toEqual({
            color: '#9E9E9E',
            weight: 3,
        });
    });

    it('conserva el conteo (una feature de salida por feature de geometría) y la geometría original', () => {
        const geometry = geometryOf('a', 'b', 'c');
        const state: CongestionStateResponse = { edges: [], count: 0 };

        const merged = mergeCongestion(geometry, state);

        expect(merged).toHaveLength(3);
        expect(merged[0].geometry).toEqual(geometry.features[0].geometry);
        expect(merged[0].properties.source_node).toBe('src_a');
        expect(merged.every((f) => f.properties.congestion_level === null)).toBe(true);
    });
});

describe('mergeCongestionAtIndex', () => {
    function seriesOf(
        edges: { edge_id: string; levels: number[] }[],
    ): CongestionSeriesResponse {
        return {
            day: '2025-01-06',
            t0: '2025-01-06T00:00:00',
            step_s: 300,
            coverage_end: '2025-01-06T23:59:00',
            count: edges.length,
            edges,
        };
    }

    it('adjunta el nivel del índice i por edge_id', () => {
        const geometry = geometryOf('a', 'b');
        const series = seriesOf([
            { edge_id: 'a', levels: [0, 4, 2] },
            { edge_id: 'b', levels: [5, 1, 3] },
        ]);

        const merged = mergeCongestionAtIndex(geometry, series, 1);

        expect(merged).toHaveLength(2);
        expect(merged[0].properties.edge_id).toBe('a');
        expect(merged[0].properties.congestion_level).toBe(4);
        expect(merged[1].properties.congestion_level).toBe(1);
        // la serie no aporta timestamp por muestra
        expect(merged[0].properties.snapshot_timestamp).toBeNull();
    });

    it('cae a nivel null (neutro) cuando el índice está fuera del rango de levels', () => {
        const geometry = geometryOf('a');
        const series = seriesOf([{ edge_id: 'a', levels: [0, 1] }]);

        const merged = mergeCongestionAtIndex(geometry, series, 5);

        expect(merged[0].properties.congestion_level).toBeNull();
        expect(congestionStyle(merged[0].properties.congestion_level)).toEqual({
            color: '#9E9E9E',
            weight: 3,
        });
    });

    it('es robusto a una arista sin entrada en la serie: nivel null, no lanza', () => {
        const geometry = geometryOf('a', 'huerfana');
        const series = seriesOf([{ edge_id: 'a', levels: [2, 3] }]);

        const merged = mergeCongestionAtIndex(geometry, series, 0);

        expect(merged[1].properties.edge_id).toBe('huerfana');
        expect(merged[1].properties.congestion_level).toBeNull();
        expect(merged[1].properties.snapshot_timestamp).toBeNull();
    });

    it('conserva el conteo 1:1 y la geometría original', () => {
        const geometry = geometryOf('a', 'b', 'c');
        const series = seriesOf([
            { edge_id: 'a', levels: [1] },
            { edge_id: 'b', levels: [2] },
            { edge_id: 'c', levels: [3] },
        ]);

        const merged = mergeCongestionAtIndex(geometry, series, 0);

        expect(merged).toHaveLength(3);
        expect(merged[0].geometry).toEqual(geometry.features[0].geometry);
        expect(merged[0].properties.source_node).toBe('src_a');
    });
});

describe('elapsedSeconds', () => {
    it('calcula los segundos transcurridos (el "hace X")', () => {
        const now = new Date('2025-01-06T23:59:30Z');
        // 30 s después del snapshot
        expect(elapsedSeconds('2025-01-06T23:59:00Z', now)).toBe(30);
    });

    it('normaliza el timestamp naive como UTC (no lo interpreta como hora local)', () => {
        const now = new Date('2025-01-06T23:59:30Z');
        // un naive sin Z debe dar el MISMO elapsed que con Z; si se interpretara como
        // hora local, el resultado dependería del huso de la máquina (bug en Lima UTC-5).
        const elapsedNaive = elapsedSeconds('2025-01-06T23:59:00', now);
        const elapsedUtc = elapsedSeconds('2025-01-06T23:59:00Z', now);
        expect(elapsedNaive).toBe(elapsedUtc);
        expect(elapsedNaive).toBe(30);
    });

    it('respeta un offset explícito cuando viene en el string', () => {
        const now = new Date('2025-01-07T05:00:00Z');
        // 00:00:00-05:00 == 05:00:00Z → mismo instante que `now`, elapsed 0
        expect(elapsedSeconds('2025-01-07T00:00:00-05:00', now)).toBe(0);
    });
});

describe('isStale', () => {
    const snapshot = '2025-01-06T23:59:00'; // naive UTC

    it('no es stale por debajo del umbral por defecto (90 s)', () => {
        const now = new Date('2025-01-06T23:59:30Z'); // 30 s
        expect(isStale(snapshot, now)).toBe(false);
    });

    it('es stale por encima del umbral por defecto', () => {
        const now = new Date('2025-01-07T00:01:00Z'); // 120 s
        expect(isStale(snapshot, now)).toBe(true);
    });

    it('en el límite exacto del umbral NO es stale (comparación estricta >)', () => {
        const now = new Date('2025-01-07T00:00:30Z'); // 90 s exactos
        expect(isStale(snapshot, now, 90)).toBe(false);
    });

    it('respeta un thresholdSec personalizado', () => {
        const now = new Date('2025-01-06T23:59:45Z'); // 45 s
        expect(isStale(snapshot, now, 30)).toBe(true);
        expect(isStale(snapshot, now, 60)).toBe(false);
    });
});
