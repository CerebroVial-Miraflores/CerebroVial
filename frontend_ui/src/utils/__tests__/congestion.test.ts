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
    mergeCongestion,
    elapsedSeconds,
    isStale,
} from '../congestion';
import type {
    GeometryFeature,
    GeometryFeatureCollection,
    CongestionStateResponse,
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
