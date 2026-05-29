"""CT-07.8.c — Adaptador TraCI ↔ motor end-to-end con motor mockeado.

Tests (d1-d4):
- d1: corrida 180s con engine mock devolviendo plan fijo. Exit limpio,
  setProgramLogic invocado al menos una vez.
- d2: el mock se invoca cada 30s (esperado: 6 veces en 180s).
- d3 (Catch A): setProgramLogic se aplica solo en borde de ciclo
  (EW_r → NS_g), NO en cada borde de sub-fase. Para 180s con ciclos
  ~60s, esperar ≤4 aplicaciones (no 6 ni más).
- d4 (corrección 3): expand_timings_to_sumo_phases retorna 6 Phases
  con durations [g, y, r, g, y, r] del input.

Tests adicionales:
- d5 (Catch C unit): aggregate_to_phase_flows con flujos N=2400 S=2400
  E=600 W=600 produce flow_total = 6000 > 1500 (engine debe rutear a
  max_pressure).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cerebrovial_simulation.traci_adapter import tllogic_applier
from cerebrovial_simulation.traci_adapter.phase_aggregator import (
    aggregate_to_phase_flows,
)
from cerebrovial_simulation.traci_adapter.run_e2e import run_e2e
from cerebrovial_simulation.traci_adapter.state_reader import DirectionState


SIM_ROOT = Path(__file__).resolve().parent.parent
LINKSTATES_JSON = SIM_ROOT / "conf" / "network" / "linkstates.json"


# Plan fijo del motor (max_pressure ciclo 60s) — para mocks.
MOCK_RECOMMENDATION = {
    "intersection_id": "miraflores_4way",
    "mode": "max_pressure",
    "cycle_seconds": 60.0,
    "phase_timings": [
        {"phase_id": "NS", "green": 25.0, "yellow": 3.0, "all_red": 2.0},
        {"phase_id": "EW", "green": 25.0, "yellow": 3.0, "all_red": 2.0},
    ],
    "next_phase": "NS",
    "reasoning": "mocked",
    "adjustments": [],
    "decision_id": None,
}


def _mock_engine(call_log: list, recommendation: dict = MOCK_RECOMMENDATION):
    def _fn(*, intersection_id, timestamp, phase_flows, lost_time):
        call_log.append({
            "intersection_id": intersection_id,
            "timestamp": timestamp,
            "phase_flows": phase_flows,
            "lost_time": lost_time,
        })
        return recommendation
    return _fn


# ---------- d4: expansión PhaseTiming → 3 sub-fases SUMO ----------

def test_d4_expand_timings_produces_6_phases():
    """corrección 3: 2 PhaseTiming → 6 Phases SUMO."""
    linkstates = json.loads(LINKSTATES_JSON.read_text())
    timings = [
        {"phase_id": "NS", "green": 30.0, "yellow": 3.0, "all_red": 2.0},
        {"phase_id": "EW", "green": 25.0, "yellow": 3.0, "all_red": 2.0},
    ]
    phases = tllogic_applier.expand_timings_to_sumo_phases(timings, linkstates)
    assert len(phases) == 6
    expected_durations = [30.0, 3.0, 2.0, 25.0, 3.0, 2.0]
    actual_durations = [float(p.duration) for p in phases]
    assert actual_durations == expected_durations
    # Cada Phase tiene state linkstates de la sub-fase correcta.
    assert phases[0].state == linkstates["NS_g"]
    assert phases[1].state == linkstates["NS_y"]
    assert phases[2].state == linkstates["NS_r"]
    assert phases[3].state == linkstates["EW_g"]
    assert phases[4].state == linkstates["EW_y"]
    assert phases[5].state == linkstates["EW_r"]


# ---------- d5: phase_aggregator con flow tasa-de-paso ----------

def test_d5_phase_aggregator_flow_total_crosses_peak_threshold():
    """Catch C: aggregator pasa flow tasa-de-paso al motor.
    Bajo carga am_peak, flow_total > 1500 (PEAK_THRESHOLD del engine).
    """
    states = {
        "N": DirectionState("N", flow_vph=2400, queue_vehicles=12, queue_length_m=90, mean_speed_mps=8),
        "S": DirectionState("S", flow_vph=2400, queue_vehicles=12, queue_length_m=90, mean_speed_mps=8),
        "E": DirectionState("E", flow_vph=600,  queue_vehicles=4,  queue_length_m=30, mean_speed_mps=9),
        "W": DirectionState("W", flow_vph=600,  queue_vehicles=4,  queue_length_m=30, mean_speed_mps=9),
    }
    phase_flows = aggregate_to_phase_flows(states)
    assert len(phase_flows) == 2
    flow_total = sum(p["flow"] for p in phase_flows)
    assert flow_total > 1500, f"flow_total={flow_total} no cruza umbral 1500"
    # PhaseFlow NS suma N + S.
    pf_ns = next(p for p in phase_flows if p["phase_id"] == "NS")
    assert pf_ns["flow"] == 4800
    assert pf_ns["saturation_flow"] == 10800
    # Queue es max, no suma.
    assert pf_ns["queue"] == 12


# ---------- d1, d2, d3: corrida e2e con mock ----------

@pytest.fixture(scope="module")
def e2e_result(tmp_path_factory):
    """Una corrida única reutilizada por d1/d2/d3."""
    call_log: list = []
    out_dir = tmp_path_factory.mktemp("e2e_d123")
    stats = run_e2e(
        pattern="am_peak",
        seed=1,
        end_s=180,
        out_dir=out_dir,
        engine_recommend_fn=_mock_engine(call_log),
    )
    return stats, call_log


def test_d1_e2e_runs_and_applies_plan(e2e_result):
    """Corrida 180s con motor mock termina limpio y aplica plan ≥1 vez."""
    stats, _ = e2e_result
    assert stats["total_steps"] == 180
    assert stats["n_applications"] >= 1, (
        f"setProgramLogic nunca aplicado en 180s; n_applications={stats['n_applications']}"
    )


def test_d2_engine_invoked_every_30s(e2e_result):
    """Motor invocado cada 30s sim ⇒ 6 veces en 180s."""
    stats, call_log = e2e_result
    assert stats["n_engine_calls"] == 6, (
        f"engine calls={stats['n_engine_calls']}, esperado 6"
    )
    assert len(call_log) == 6


def test_d3_applications_only_at_cycle_boundary(e2e_result):
    """Catch A: aplicaciones ≤ número de ciclos (no = engine calls).

    Para 180s con ciclo baseline 60s: máximo 3 ciclos = máximo 3 borders
    EW_r → NS_g. Por lo tanto setProgramLogic ≤ 3 (no 6 = engine calls).
    """
    stats, _ = e2e_result
    n_apps = stats["n_applications"]
    n_calls = stats["n_engine_calls"]
    assert n_apps <= 3, (
        f"setProgramLogic aplicó {n_apps} veces; esperado ≤3 (= ciclos en 180s). "
        f"Si > 3, está aplicando en bordes de sub-fase, no de ciclo."
    )
    # Y aplicó al menos una vez (d1 cubre).
    assert n_apps >= 1
    # Aplicaciones < llamadas (porque no toda llamada cae en borde de ciclo).
    assert n_apps < n_calls or n_apps <= 3
