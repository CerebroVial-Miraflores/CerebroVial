/**
 * Tests de los helpers puros de datos de congestión (HU-22/HU-23) — sin DOM.
 *
 * Cubre el cruce geometry × state por `edge_id` (incl. la arista huérfana) y la
 * detección de staleness con normalización UTC (CA-22.4). FASE 3: murieron los
 * describes de congestionStyle/predictionStyle junto con esas funciones (el
 * estilo de tramos vive en components/map/edgeStyle.ts y su test).
 */
import { describe, it, expect } from 'vitest';
import {
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
