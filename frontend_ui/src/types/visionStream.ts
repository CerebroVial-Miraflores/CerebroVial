/**
 * Tipos del payload SSE de visión (shape §6.2, TTH-08 Fase 6 / 6d).
 *
 * El broadcaster del módulo `edge_device` emite `event: traffic_update` por cada
 * `TrafficData` agregado de la ventana (cadencia = `vision.persistence.interval_seconds`
 * del backend). Payload puro: números, sin localización, sin formateos `%`, sin
 * umbrales hardcoded — todo eso se computa client-side con `utils/trafficLabels.ts`
 * (§9.2 prohíbe localización/umbrales en el transporte).
 *
 * `camera.street_monitored` viene como `null` en MVP1; ningún consumidor frontend
 * lo lee. Si en el futuro se requiere el nombre humano de la calle/intersección,
 * dos rutas abiertas: (1) registry en frontend, (2) `CameraMetadataProvider`
 * inyectable en el broadcaster (volver a §6.2 al pie). Documentado en handoff
 * de cierre de Fase 6.
 */

export interface VisionStreamWindow {
    start: string; // ISO-8601 UTC
    end: string; // ISO-8601 UTC
    duration_seconds: number;
}

export interface VisionStreamCamera {
    id: string;
    street_monitored: string | null;
}

export interface VisionStreamZone {
    id: string;
}

export interface VisionStreamMetrics {
    unique_vehicles: number;
    vehicles_by_type: Record<string, number>;
    /** km/h; `null` si la cámara carece de calibración pixel→metro. */
    mean_speed_kmh: number | null;
    flow_vehicles_per_hour: number;
    /** Fracción [0.0, 1.0] del área de zona cubierta por bboxes. */
    mean_occupancy: number;
    /** `null` si la zona carece de `segment_length_meters`. */
    density_vehicles_per_km: number | null;
}

export interface VisionStreamPayload {
    schema_version: string;
    event_type: 'traffic_update';
    server_timestamp: string;
    camera: VisionStreamCamera;
    zone: VisionStreamZone;
    window: VisionStreamWindow;
    metrics: VisionStreamMetrics;
}
