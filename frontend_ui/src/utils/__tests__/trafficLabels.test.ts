/**
 * Tests del helper de discretización + formateo client-side (TTH-08 Fase 6 / 6g).
 *
 * Cubre los umbrales de congestión (0.30 / 0.70 sobre `mean_occupancy`), la
 * coherencia entre `congestionLabel` y `congestionUiStatus`, y el clamp del
 * `densityPercent`.
 */
import { describe, it, expect } from 'vitest';
import {
    congestionLabel,
    congestionUiStatus,
    densityPercent,
} from '../trafficLabels';

describe('congestionLabel', () => {
    it.each([
        [0.0, 'Bajo'],
        [0.15, 'Bajo'],
        [0.30, 'Bajo'], // boundary: > 0.30 → Moderado, exact 0.30 → Bajo
        [0.31, 'Moderado'],
        [0.50, 'Moderado'],
        [0.70, 'Moderado'], // boundary: > 0.70 → Alto, exact 0.70 → Moderado
        [0.71, 'Alto'],
        [0.95, 'Alto'],
        [1.0, 'Alto'],
    ])('mean_occupancy %f → %s', (occ, expected) => {
        expect(congestionLabel(occ)).toBe(expected);
    });

    it('returns Bajo for NaN', () => {
        expect(congestionLabel(NaN)).toBe('Bajo');
    });

    it('returns Bajo for Infinity (defensive against transport corruption)', () => {
        expect(congestionLabel(Number.POSITIVE_INFINITY)).toBe('Bajo');
    });
});

describe('congestionUiStatus', () => {
    it.each([
        [0.0, 'fluid'],
        [0.30, 'fluid'],
        [0.50, 'moderate'],
        [0.70, 'moderate'],
        [0.85, 'critical'],
    ])('mean_occupancy %f → %s', (occ, expected) => {
        expect(congestionUiStatus(occ)).toBe(expected);
    });

    it('is consistent with congestionLabel', () => {
        const cases = [0.0, 0.15, 0.31, 0.50, 0.71, 1.0] as const;
        for (const occ of cases) {
            const label = congestionLabel(occ);
            const status = congestionUiStatus(occ);
            const expectedStatus =
                label === 'Alto' ? 'critical' : label === 'Moderado' ? 'moderate' : 'fluid';
            expect(status).toBe(expectedStatus);
        }
    });
});

describe('densityPercent', () => {
    it('formats fraction as percentage rounded to integer', () => {
        expect(densityPercent(0.47)).toBe('47%');
        expect(densityPercent(0.0)).toBe('0%');
        expect(densityPercent(1.0)).toBe('100%');
    });

    it('rounds half-up', () => {
        expect(densityPercent(0.475)).toBe('48%');
    });

    it('clamps below 0 to 0%', () => {
        expect(densityPercent(-0.5)).toBe('0%');
    });

    it('clamps above 1 to 100%', () => {
        expect(densityPercent(1.5)).toBe('100%');
    });

    it('returns 0% for NaN', () => {
        expect(densityPercent(NaN)).toBe('0%');
    });
});
