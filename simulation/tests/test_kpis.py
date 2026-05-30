"""F5 — Webster offline (Catch B) + KPIs comparativos.

Tests:
- e1: Webster offline (catch B) computa cycle ∈ [30,120], greens ≥
  MIN_GREEN, cycle = green_NS + green_EW + 2*(yellow+all_red) (tras MTC clamp).
- e2: corrida fixed_webster produce KPIs con todas las claves esperadas, finitas.
- e3: Webster compute_from_pattern("am_peak") produce el mismo plan que invocando
  con flujos de YAML manualmente (consistency).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from cerebrovial_simulation.fixed_control import webster_fixed
from cerebrovial_simulation.kpis import run_comparison


SIM_ROOT = Path(__file__).resolve().parent.parent


def test_e1_webster_offline_within_mtc_bounds():
    """Catch B: webster_fixed.compute(flows AM-peak, sat, L=10) cumple MTC bounds."""
    plan = webster_fixed.compute(
        flow_ns_vph=4800,    # 2400 + 2400
        flow_ew_vph=1200,    # 600 + 600
        sat_ns_vph=10800,    # 6 × 1800
        sat_ew_vph=7200,     # 4 × 1800
        lost_time_s=10.0,
    )
    # Cycle dentro de MTC bounds.
    assert webster_fixed.MIN_CYCLE <= plan.cycle_seconds <= webster_fixed.MAX_CYCLE, (
        f"cycle={plan.cycle_seconds} fuera de [{webster_fixed.MIN_CYCLE}, "
        f"{webster_fixed.MAX_CYCLE}]"
    )
    # Greens dentro de MTC bounds.
    assert plan.green_NS >= webster_fixed.MIN_GREEN
    assert plan.green_EW >= webster_fixed.MIN_GREEN
    assert plan.green_NS <= webster_fixed.MAX_GREEN
    assert plan.green_EW <= webster_fixed.MAX_GREEN
    # Suma consistente: cycle ≈ green_NS + green_EW + 2*(yellow + all_red).
    expected_cycle = plan.green_NS + plan.green_EW + 2 * (plan.yellow_s + plan.all_red_s)
    assert math.isclose(plan.cycle_seconds, expected_cycle, abs_tol=0.01), (
        f"cycle inconsistente: {plan.cycle_seconds} vs {expected_cycle}"
    )
    # Y total finito y bajo 0.95.
    assert 0 < plan.y_total < 0.95


def test_e2_fixed_webster_run_produces_finite_kpis(tmp_path):
    """Una corrida fixed_webster (60s) produce KPIs con todas las claves no-NaN."""
    # Precomputar add-file en su ubicación canónica (idempotente).
    if not webster_fixed.ADD_FILE.exists():
        plan = webster_fixed.compute_from_pattern("am_peak")
        webster_fixed.write_addfile(plan)

    out_dir = tmp_path / "fixed_am_peak_seed1"
    kpi = run_comparison._run_fixed_webster(
        pattern="am_peak", seed=1, end_s=120, out_dir=out_dir
    )
    d = kpi.to_dict()
    # Todas las claves esperadas presentes.
    expected_keys = {
        "mode", "pattern", "seed", "sim_duration_s",
        "mean_travel_time_s", "total_delay_s", "throughput_veh_per_h",
        "max_queue_m_N", "max_queue_m_S", "max_queue_m_E", "max_queue_m_W",
        "mean_queue_m_N", "mean_queue_m_S", "mean_queue_m_E", "mean_queue_m_W",
    }
    assert expected_keys.issubset(set(d.keys()))
    # Numéricos finitos no-NaN.
    for k in expected_keys - {"mode", "pattern", "seed"}:
        v = d[k]
        assert v is not None and isinstance(v, (int, float)) and math.isfinite(v), (
            f"{k} no finito: {v}"
        )
    assert d["mode"] == "fixed_webster"
    assert d["sim_duration_s"] > 0
    assert d["throughput_veh_per_h"] > 0


def test_e3_compute_from_pattern_consistent_with_manual():
    """compute_from_pattern lee YAML correctamente y produce el mismo plan."""
    auto = webster_fixed.compute_from_pattern("am_peak")
    manual = webster_fixed.compute(
        flow_ns_vph=4800,
        flow_ew_vph=1200,
        sat_ns_vph=10800,
        sat_ew_vph=7200,
        lost_time_s=10.0,
    )
    assert math.isclose(auto.cycle_seconds, manual.cycle_seconds, abs_tol=0.01)
    assert math.isclose(auto.green_NS, manual.green_NS, abs_tol=0.01)
    assert math.isclose(auto.green_EW, manual.green_EW, abs_tol=0.01)


def test_e4_infeasible_raises():
    """Y_total >= 0.95 lanza WebsterInfeasible."""
    with pytest.raises(webster_fixed.WebsterInfeasible):
        webster_fixed.compute(
            flow_ns_vph=10000,  # 92.6% sat NS
            flow_ew_vph=6000,   # 83% sat EW → Y_total > 1
            sat_ns_vph=10800,
            sat_ew_vph=7200,
            lost_time_s=10.0,
        )
