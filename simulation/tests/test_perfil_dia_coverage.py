"""Cobertura del dataset perfil-día (TTH-11) — criterio nuevo de CT-07.2.

Reemplaza, para el set perfil-día, el criterio "jam ≥3 sostenido" del viejo
test_b2 (que aplicaba a los patrones constantes de 20 min de TTH-07 y sigue
xfail en test_patterns_run.py). Acá el jam 3-4 es TRANSITORIO en las ventanas
pico:

- a (smoke, rápido): cada perfil corre 600 s sin error y produce edgedata.xml +
  lanearea.xml válidos (verifica que sumocfg + rutas fásicas estén bien cableadas).
- b (slow, 24h): coverage_perfil_dia satisfecho para los 4 perfiles con el
  criterio nuevo (pico transitorio en ventana / valle sin congestión persistente /
  jam 0 en valles).
"""
from __future__ import annotations

import pytest
import yaml

from cerebrovial_simulation.dataset.coverage_perfil_dia import (
    PERFIL_DIA_PARAMS,
    check_coverage,
)
from cerebrovial_simulation.coverage_check import run_sumo_for_coverage


PATTERNS = ["laborable", "finde", "feriado", "especial"]


@pytest.fixture(scope="module")
def perfil_cfg():
    with open(PERFIL_DIA_PARAMS) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_a_perfil_runs_and_produces_outputs(tmp_path, pattern):
    """Smoke: cada perfil corre 600 s y produce edgedata.xml + lanearea.xml."""
    out_dir = tmp_path / pattern
    edgedata, lanearea = run_sumo_for_coverage(
        pattern=pattern, seed=42, out_dir=out_dir, end_s=600
    )
    assert edgedata.exists()
    assert lanearea.exists()
    assert edgedata.stat().st_size > 100
    assert lanearea.stat().st_size > 100


@pytest.mark.slow
def test_b_coverage_satisfied_for_all_perfiles(tmp_path, perfil_cfg):
    """Criterio nuevo satisfecho para los 4 perfiles (corrida completa de 24h)."""
    period_s = perfil_cfg["period_s"]
    fail_msgs = []
    for name in PATTERNS:
        params = perfil_cfg["patterns"][name]
        out_dir = tmp_path / name
        report = check_coverage(
            pattern=name,
            seed=42,
            out_dir=out_dir,
            coverage=params["coverage"],
            end_s=period_s,
        )
        if not report.satisfied:
            fail_msgs.append(
                f"{name}: {report.reasons} "
                f"(max_jam={report.max_jam_per_dir}, "
                f"peak_jam_in_window={report.peak_jam_in_window}, "
                f"buckets@target={report.peak_buckets_at_target}, "
                f"jam0_in_valley={report.jam0_in_valley})"
            )
    assert not fail_msgs, "Coverage perfil-día falló:\n" + "\n".join(fail_msgs)
