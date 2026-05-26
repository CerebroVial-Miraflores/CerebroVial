"""
Webster (1958) optimal cycle calculator for under-saturated intersections.

Reference: Webster, F.V. (1958). "Traffic Signal Settings". Road Research
Technical Paper No. 39. London: HMSO.

Formulae:
    Y       = Σ (q_i / s_i)             over all phases (critical-lane simplification: all phases treated as critical)
    C_opt   = (1.5 · L + 5) / (1 − Y)   in seconds
    g_i     = (C_opt − L) · (y_i / Y)   with y_i = q_i / s_i

When Y → 1 the cycle diverges; we treat Y ≥ 0.95 as infeasible.

Yellow and all-red intervals are NOT computed here — they are part of `lost_time`
and are added back as part of `PhaseTimings` by `MTCRestrictionApplier`.
"""
from dataclasses import dataclass


class WebsterInfeasible(Exception):
    """Raised when intersection is over-saturated (Y ≥ Y_max)."""


@dataclass
class WebsterResult:
    cycle_seconds: float
    green_by_phase: dict[str, float]
    y_total: float  # Σ q_i / s_i — exposed so the write-path can persist it.


class WebsterCalculator:
    Y_MAX = 0.95

    def __init__(self, min_cycle: float = 30.0, max_cycle: float = 120.0) -> None:
        if min_cycle <= 0 or max_cycle <= min_cycle:
            raise ValueError("min_cycle and max_cycle must satisfy 0 < min_cycle < max_cycle.")
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle

    def compute(
        self,
        flows: dict[str, float],
        saturations: dict[str, float],
        lost_time: float,
    ) -> WebsterResult:
        if not flows:
            raise ValueError("flows must contain at least one phase.")
        if set(flows) != set(saturations):
            raise ValueError("flows and saturations must share the same phase ids.")
        if lost_time < 0:
            raise ValueError("lost_time must be non-negative.")

        ratios: dict[str, float] = {}
        for phase_id, q in flows.items():
            s = saturations[phase_id]
            if s <= 0:
                raise ValueError(f"saturation_flow for phase {phase_id!r} must be > 0.")
            if q < 0:
                raise ValueError(f"flow for phase {phase_id!r} must be >= 0.")
            ratios[phase_id] = q / s

        y_total = sum(ratios.values())

        if y_total >= self.Y_MAX:
            raise WebsterInfeasible(
                f"Intersection is over-saturated: Y={y_total:.3f} >= Y_max={self.Y_MAX}."
            )

        c_calc = (1.5 * lost_time + 5.0) / (1.0 - y_total)
        cycle = max(self.min_cycle, min(self.max_cycle, c_calc))
        usable_green = max(0.0, cycle - lost_time)

        if y_total == 0.0:
            share = usable_green / len(ratios)
            green_by_phase = {p: share for p in ratios}
        else:
            green_by_phase = {p: usable_green * (y / y_total) for p, y in ratios.items()}

        return WebsterResult(
            cycle_seconds=cycle,
            green_by_phase=green_by_phase,
            y_total=y_total,
        )
