"""Función pura de cómputo de `TrafficData` desde una ventana de `FrameAnalysis`.

Privada al paquete `aggregators/`. Sin I/O, sin threads, sin side effects. La
usa el `AsyncTrafficAggregator` (5b) dentro de su worker; los tests la
ejercitan directamente.

Schema canónico: `tth-08-fase1-diseno.md` §5.4. Decisión clave sobre la
interpretación de `mean_occupancy`: ver DHU-025 (unión vs suma + clip).
"""
from collections import Counter, defaultdict
from datetime import datetime

from ...domain.entities import FrameAnalysis, TrafficData
from ...domain.value_objects import CameraId, VehicleId, ZoneId


def compute_traffic_data(
    analyses: list[FrameAnalysis],
    camera_id: CameraId,
    window_start: datetime,
    window_end: datetime,
    zone_segment_lengths: dict[ZoneId, float | None],
) -> list[TrafficData]:
    """Computa un `TrafficData` por cada zona que aparece en `analyses`.

    Args:
        analyses: frames de la ventana (orden arbitrario; el cómputo no lo asume).
        camera_id: identificador de la cámara (todos los frames son de la misma).
        window_start: inicio del intervalo de agregación, tz-aware UTC.
        window_end: fin del intervalo, tz-aware UTC, mayor que `window_start`.
        zone_segment_lengths: mapeo zona → segment_length_meters (None si la zona
            no está calibrada — `density_vehicles_per_km` quedará `None`).

    Returns:
        Lista de `TrafficData`, ordenada por `zone_id.value`. Vacía si
        `analyses` está vacío.

    Raises:
        ValueError: si `window_end <= window_start`.
    """
    window_duration_s = (window_end - window_start).total_seconds()
    if window_duration_s <= 0:
        raise ValueError("window_end debe ser estrictamente posterior a window_start")
    if not analyses:
        return []

    # Por zona: tipos observados por vehículo (para voto mayoritario),
    # pares (count, avg_speed) por frame con velocidades válidas (para
    # mean_speed_kmh count-weighted), y occupancies por frame.
    types_per_vehicle: dict[ZoneId, dict[VehicleId, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    speed_pairs_per_zone: dict[ZoneId, list[tuple[int, float]]] = defaultdict(list)
    occupancies_per_zone: dict[ZoneId, list[float]] = defaultdict(list)

    for analysis in analyses:
        type_by_id = {v.id: v.type for v in analysis.vehicles}
        speed_by_id = {v.id: v.speed for v in analysis.vehicles}

        for zone_id, zvc in analysis.zones.items():
            occupancies_per_zone[zone_id].append(zvc.occupancy)

            for vid in zvc.vehicle_ids:
                vehicle_type = type_by_id.get(vid)
                if vehicle_type is not None:
                    types_per_vehicle[zone_id][vid].append(vehicle_type)

            valid_speeds = [
                speed_by_id[vid]
                for vid in zvc.vehicle_ids
                if speed_by_id.get(vid) is not None and speed_by_id[vid] > 0  # type: ignore[operator]
            ]
            if zvc.count > 0 and valid_speeds:
                frame_avg_speed = sum(valid_speeds) / len(valid_speeds)
                speed_pairs_per_zone[zone_id].append((zvc.count, frame_avg_speed))

    # Conjunto total de zonas que aparecieron en algún frame.
    all_zones = (
        set(occupancies_per_zone)
        | set(types_per_vehicle)
        | set(speed_pairs_per_zone)
    )

    result: list[TrafficData] = []
    for zone_id in sorted(all_zones, key=lambda z: z.value):
        unique_vehicles, vehicles_by_type = _resolve_vehicle_types(
            types_per_vehicle.get(zone_id, {})
        )
        mean_speed_kmh = _count_weighted_speed(speed_pairs_per_zone.get(zone_id, []))
        flow_vph = unique_vehicles / window_duration_s * 3600.0
        mean_occupancy = _mean(occupancies_per_zone.get(zone_id, [])) or 0.0
        density = _density_per_km(
            unique_vehicles, zone_segment_lengths.get(zone_id)
        )
        result.append(
            TrafficData(
                camera_id=camera_id,
                zone_id=zone_id,
                window_start=window_start,
                window_end=window_end,
                window_duration_seconds=window_duration_s,
                unique_vehicles=unique_vehicles,
                vehicles_by_type=vehicles_by_type,
                mean_speed_kmh=mean_speed_kmh,
                flow_vehicles_per_hour=flow_vph,
                mean_occupancy=mean_occupancy,
                density_vehicles_per_km=density,
            )
        )
    return result


def _resolve_vehicle_types(
    types_per_vehicle: dict[VehicleId, list[str]],
) -> tuple[int, dict[str, int]]:
    """Voto mayoritario por vehículo. En empate gana el primero observado."""
    vehicles_by_type: dict[str, int] = defaultdict(int)
    for types in types_per_vehicle.values():
        if not types:
            continue
        # Counter preserva insertion order en ties con `most_common`.
        voted_type = Counter(types).most_common(1)[0][0]
        vehicles_by_type[voted_type] += 1
    return len(types_per_vehicle), dict(vehicles_by_type)


def _count_weighted_speed(
    pairs: list[tuple[int, float]],
) -> float | None:
    """§5.4: Σ(count × avg_speed) / Σ(count) sobre frames válidos.

    `None` si no hay frames con `count > 0` y `avg_speed > 0`.
    """
    if not pairs:
        return None
    total_weight = sum(c for c, _ in pairs)
    if total_weight == 0:
        return None
    weighted_sum = sum(c * s for c, s in pairs)
    return weighted_sum / total_weight


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _density_per_km(
    unique_vehicles: int, segment_length_meters: float | None
) -> float | None:
    """§5.4: `unique_vehicles / segment_length_meters * 1000`.

    `None` si la zona no tiene calibración (`segment_length_meters is None`).
    """
    if segment_length_meters is None or segment_length_meters <= 0:
        return None
    return unique_vehicles / segment_length_meters * 1000.0
