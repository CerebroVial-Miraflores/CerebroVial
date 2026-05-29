/**
 * Discretización y formateo client-side de métricas de tráfico (TTH-08 Fase 6 / 6g).
 *
 * El SSE de visión (§6.2) emite métricas crudas (números). Estas funciones aplican
 * la localización ES (`Bajo`/`Moderado`/`Alto`), umbrales de congestión y el
 * formateo `%` que el broadcaster solía hacer hasta 6d. Movidos al cliente per
 * §9.2 (transporte sin localización) y §5.2 (frontend consume números crudos).
 *
 * Umbrales sobre `mean_occupancy` ∈ [0, 1]: 0.30 / 0.70 (equivalentes a los
 * umbrales 30 / 70 que usaba el broadcaster viejo sobre `density` *100).
 */

export type CongestionLabel = 'Bajo' | 'Moderado' | 'Alto';
export type CongestionUiStatus = 'fluid' | 'moderate' | 'critical';

const OCCUPANCY_MODERATE_THRESHOLD = 0.30;
const OCCUPANCY_HIGH_THRESHOLD = 0.70;

/**
 * Etiqueta ES de congestión a partir de `mean_occupancy` ∈ [0, 1].
 *
 * Umbrales: `> 0.70` → `Alto`; `> 0.30` → `Moderado`; resto → `Bajo`.
 */
export function congestionLabel(meanOccupancy: number): CongestionLabel {
    if (!Number.isFinite(meanOccupancy)) return 'Bajo';
    if (meanOccupancy > OCCUPANCY_HIGH_THRESHOLD) return 'Alto';
    if (meanOccupancy > OCCUPANCY_MODERATE_THRESHOLD) return 'Moderado';
    return 'Bajo';
}

/**
 * Status interno del Dashboard (`fluid`/`moderate`/`critical`), derivado del mismo
 * `mean_occupancy` con los mismos umbrales que `congestionLabel`.
 */
export function congestionUiStatus(meanOccupancy: number): CongestionUiStatus {
    const label = congestionLabel(meanOccupancy);
    if (label === 'Alto') return 'critical';
    if (label === 'Moderado') return 'moderate';
    return 'fluid';
}

/**
 * Formato de densidad como porcentaje (`"47%"`) a partir de `mean_occupancy`.
 *
 * Clamp a [0, 100]: el dominio garantiza [0, 1] pero el cliente debe ser robusto
 * a datos sucios del transporte (e.g. una migración futura).
 */
export function densityPercent(meanOccupancy: number): string {
    if (!Number.isFinite(meanOccupancy)) return '0%';
    const pct = Math.max(0, Math.min(100, Math.round(meanOccupancy * 100)));
    return `${pct}%`;
}
