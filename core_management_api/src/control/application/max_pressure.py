"""
Max Pressure controller (Varaiya, 2013) for over-saturated intersections.

Reference: Varaiya, P. (2013). "Max pressure control of a network of signalized
intersections". Transportation Research Part C, 36, 177-195.

Pressure per phase, network form (Varaiya):
    P(φ) = saturation_flow_φ · (queue_φ − Σ τ · queue_down)

donde el término downstream es OPCIONAL y ADITIVO. Si no se pasa ``downstreams``
(o es vacío para una fase), el término es 0 y la fórmula colapsa al per-node
simplificado de SP3:
    P(φ) = saturation_flow_φ · queue_φ

La presión es FIRMADA (puede ser < 0 cuando el link downstream está más lleno
que la cola local). El controlador selecciona ``next_phase`` como argmax(P) sobre
las presiones firmadas (tie-break alfabético). Los verdes se reparten en
proporción al PESO de cada fase, donde ``weight = max(0, P)``: una fase con
presión ≤ 0 recibe peso 0 (verde 0 en el split crudo; la capa MTC lo eleva a
``min_green``). Cuando ningún peso es positivo (todas las presiones ≤ 0, lo que
incluye el caso histórico de todas las colas vacías) el controlador cae a un
round-robin determinista (orden alfabético de phase_id) para mantener el sistema
vivo.

RETROCOMPATIBILIDAD: con ``downstreams=None`` (o un dict de puros ``None``), las
presiones son todas ≥ 0, ``weight == P`` y ``total_weight == total_pressure``, de
modo que ``next_phase`` y ``green_by_phase`` salen BIT-A-BIT idénticos al motor
per-node anterior (ver test_max_pressure::test_no_downstream_matches_current_behavior).
"""
from dataclasses import dataclass


@dataclass
class DownstreamLinkDC:
    """Un movimiento de salida de una fase (Varaiya): cola del link downstream
    al que descarga la fase (``queue``, en vehículos) y la fracción del flujo de
    la fase que descarga a ese link (``turn_ratio`` = τ).

    La presión resta ``Σ τ · queue`` sobre los links de la fase. Para un corredor
    pasante de un sentido, una sola descarga con τ=1.0.
    """
    queue: int
    turn_ratio: float = 1.0


@dataclass
class MaxPressureDecision:
    next_phase: str
    cycle_seconds: float
    green_by_phase: dict[str, float]


class MaxPressureController:
    def __init__(self, default_cycle: float = 60.0) -> None:
        if default_cycle <= 0:
            raise ValueError("default_cycle must be > 0.")
        self.default_cycle = default_cycle

    def decide(
        self,
        queues: dict[str, int],
        saturations: dict[str, float],
        lost_time: float,
        cycle_seconds: float | None = None,
        downstreams: dict[str, list[DownstreamLinkDC] | None] | None = None,
    ) -> MaxPressureDecision:
        if not queues:
            raise ValueError("queues must contain at least one phase.")
        if set(queues) != set(saturations):
            raise ValueError("queues and saturations must share the same phase ids.")
        if lost_time < 0:
            raise ValueError("lost_time must be non-negative.")

        cycle = float(cycle_seconds) if cycle_seconds is not None else self.default_cycle
        usable_green = max(0.0, cycle - lost_time)

        pressures: dict[str, float] = {}
        for phase_id, q in queues.items():
            s = saturations[phase_id]
            if s <= 0:
                raise ValueError(f"saturation_flow for phase {phase_id!r} must be > 0.")
            if q < 0:
                raise ValueError(f"queue for phase {phase_id!r} must be >= 0.")
            # Término downstream OPCIONAL y ADITIVO: Σ τ · queue_down.
            # Sin downstreams (o lista None/vacía para la fase) ⇒ 0 ⇒ P = s·q.
            down_term = 0.0
            if downstreams is not None:
                links = downstreams.get(phase_id)
                if links:
                    for link in links:
                        if link.turn_ratio <= 0:
                            raise ValueError(
                                f"turn_ratio for phase {phase_id!r} must be > 0."
                            )
                        if link.queue < 0:
                            raise ValueError(
                                f"downstream queue for phase {phase_id!r} must be >= 0."
                            )
                        down_term += link.turn_ratio * link.queue
            pressures[phase_id] = s * (q - down_term)

        ordered = sorted(queues)

        # Solo las presiones positivas reparten verde. Una presión ≤ 0 (link
        # downstream más lleno que la cola local) recibe peso 0.
        weights = {p: max(0.0, pressures[p]) for p in pressures}
        total_weight = sum(weights.values())

        if total_weight == 0.0:
            # Round-robin fallback: ninguna fase con presión positiva
            # (incluye el caso histórico de todas las colas vacías).
            next_phase = ordered[0]
            share = usable_green / len(ordered)
            green_by_phase = {p: share for p in ordered}
        else:
            # argmax sobre presiones FIRMADAS, iterando alfabéticamente para que
            # los empates resuelvan al phase_id menor (max() devuelve la primera
            # clave en empate).
            next_phase = max(ordered, key=lambda p: pressures[p])
            green_by_phase = {
                p: usable_green * (weights[p] / total_weight) for p in pressures
            }

        return MaxPressureDecision(
            next_phase=next_phase,
            cycle_seconds=cycle,
            green_by_phase=green_by_phase,
        )
