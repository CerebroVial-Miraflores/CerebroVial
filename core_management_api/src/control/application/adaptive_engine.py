"""
Adaptive engine that orchestrates Webster, Max Pressure and the MTC compliance
layer to produce a single ``RecommendationDC`` per request.

Routing:
    flow_total < PEAK_THRESHOLD  → Webster + MTC          (off-peak)
    flow_total ≥ PEAK_THRESHOLD  → Max Pressure + MTC     (peak)

When Webster is infeasible during off-peak the exception propagates. During
peak operation Max Pressure runs with the controller's ``default_cycle`` even
when Webster could not produce a base cycle.

The engine deliberately operates on dataclasses (no Pydantic) so it can be
exercised in tests without spinning up FastAPI.
"""
from dataclasses import dataclass, field
from typing import Optional

from .max_pressure import MaxPressureController
from .mtc_constraints import MTCRestrictionApplier, PhaseTiming
from .webster import WebsterCalculator, WebsterInfeasible


@dataclass
class PhaseFlowDC:
    phase_id: str
    flow: float
    saturation_flow: float
    queue: int = 0
    has_pedestrian: bool = False


@dataclass
class IntersectionStateDC:
    intersection_id: str
    timestamp: str
    phases: list[PhaseFlowDC]
    lost_time: float = 8.0


@dataclass
class RecommendationDC:
    intersection_id: str
    mode: str  # "webster" or "max_pressure"
    cycle_seconds: float
    phase_timings: list[PhaseTiming]
    reasoning: str
    next_phase: Optional[str] = None
    adjustments: list[str] = field(default_factory=list)
    # Internal-only fields persisted to motor_decisions; NOT exposed in the
    # HTTP response (data-model.md §2.1, SDD §4.2). flow_total is always set;
    # y_load_factor is None only when the peak branch falls back to default_cycle
    # because Webster raised WebsterInfeasible.
    flow_total: float = 0.0
    y_load_factor: Optional[float] = None


class AdaptiveEngine:
    def __init__(
        self,
        webster: WebsterCalculator,
        max_pressure: MaxPressureController,
        mtc: MTCRestrictionApplier,
        peak_threshold: float = 1500.0,
    ) -> None:
        self.webster = webster
        self.max_pressure = max_pressure
        self.mtc = mtc
        self.PEAK_THRESHOLD = peak_threshold  # veh/h, sum across phases.

    def recommend(self, state: IntersectionStateDC) -> RecommendationDC:
        if not state.phases:
            raise ValueError("state.phases must contain at least one phase.")

        flows = {p.phase_id: p.flow for p in state.phases}
        saturations = {p.phase_id: p.saturation_flow for p in state.phases}
        queues = {p.phase_id: p.queue for p in state.phases}
        pedestrians = {p.phase_id: p.has_pedestrian for p in state.phases}

        flow_total = sum(flows.values())

        if flow_total < self.PEAK_THRESHOLD:
            return self._webster_branch(
                state=state,
                flows=flows,
                saturations=saturations,
                pedestrians=pedestrians,
                flow_total=flow_total,
            )

        return self._max_pressure_branch(
            state=state,
            flows=flows,
            saturations=saturations,
            queues=queues,
            pedestrians=pedestrians,
            flow_total=flow_total,
        )

    def _webster_branch(
        self,
        state: IntersectionStateDC,
        flows: dict[str, float],
        saturations: dict[str, float],
        pedestrians: dict[str, bool],
        flow_total: float,
    ) -> RecommendationDC:
        webster_result = self.webster.compute(
            flows=flows,
            saturations=saturations,
            lost_time=state.lost_time,
        )
        constrained = self.mtc.apply(
            green_by_phase=webster_result.green_by_phase,
            has_pedestrian=pedestrians,
        )

        reasoning = (
            f"Off-peak (Σ flow = {flow_total:.0f} veh/h < {self.PEAK_THRESHOLD:.0f}). "
            f"Webster optimal cycle = {webster_result.cycle_seconds:.1f}s; "
            "MTC constraints applied."
        )

        return RecommendationDC(
            intersection_id=state.intersection_id,
            mode="webster",
            cycle_seconds=constrained.cycle_seconds,
            phase_timings=constrained.timings,
            reasoning=reasoning,
            next_phase=None,
            adjustments=constrained.adjustments,
            flow_total=flow_total,
            y_load_factor=webster_result.y_total,
        )

    def _max_pressure_branch(
        self,
        state: IntersectionStateDC,
        flows: dict[str, float],
        saturations: dict[str, float],
        queues: dict[str, int],
        pedestrians: dict[str, bool],
        flow_total: float,
    ) -> RecommendationDC:
        y_load_factor: Optional[float] = None
        try:
            webster_base = self.webster.compute(
                flows=flows,
                saturations=saturations,
                lost_time=state.lost_time,
            )
            base_cycle = webster_base.cycle_seconds
            y_load_factor = webster_base.y_total
            cycle_source = "Webster base cycle"
        except WebsterInfeasible:
            base_cycle = self.max_pressure.default_cycle
            cycle_source = "Max Pressure default cycle (Webster infeasible)"

        decision = self.max_pressure.decide(
            queues=queues,
            saturations=saturations,
            lost_time=state.lost_time,
            cycle_seconds=base_cycle,
        )
        constrained = self.mtc.apply(
            green_by_phase=decision.green_by_phase,
            has_pedestrian=pedestrians,
        )

        reasoning = (
            f"Peak (Σ flow = {flow_total:.0f} veh/h ≥ {self.PEAK_THRESHOLD:.0f}). "
            f"Max Pressure on {cycle_source} = {base_cycle:.1f}s; "
            f"next phase = {decision.next_phase}; MTC constraints applied."
        )

        return RecommendationDC(
            intersection_id=state.intersection_id,
            mode="max_pressure",
            cycle_seconds=constrained.cycle_seconds,
            phase_timings=constrained.timings,
            reasoning=reasoning,
            next_phase=decision.next_phase,
            adjustments=constrained.adjustments,
            flow_total=flow_total,
            y_load_factor=y_load_factor,
        )
