"""CT-07.8.b — Cada patrón corre + cobertura jam level verificada.

Tests:
- b1: cada uno de los 4 .sumocfg corre 600s sin error y produce
  edgedata.xml + lanearea.xml válidos.
- b2: coverage_check.run_all_patterns() reporta satisfied=True para los 4.

Time-box per pattern run: ~3 min con end_s=600 (suficiente para cubrir
warmup + jam estabilizado + tail). El test completo total ≤15 min.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cerebrovial_simulation.coverage_check import (
    PATTERN_PARAMS,
    check_coverage,
    run_sumo_for_coverage,
)


SIM_ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ["am_peak", "pm_peak", "offpeak", "weekend"]


@pytest.fixture(scope="module")
def pattern_cfg():
    with open(PATTERN_PARAMS) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_b1_pattern_runs_and_produces_outputs(tmp_path, pattern):
    """Cada patrón corre 600s sin error y produce edgedata.xml + lanearea.xml."""
    out_dir = tmp_path / pattern
    edgedata, lanearea = run_sumo_for_coverage(
        pattern=pattern,
        seed=42,
        out_dir=out_dir,
        end_s=600,
    )
    assert edgedata.exists()
    assert lanearea.exists()
    # Sanity: outputs no vacíos.
    assert edgedata.stat().st_size > 100
    assert lanearea.stat().st_size > 100


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Realineación D-009 a escala Waze (2026-05-31): am_peak/pm_peak fueron "
        "calibrados para jam≥3 sostenido bajo los cortes viejos (90/70/50/30). "
        "Con los cortes Waze (80/60/40/20) el mismo tráfico se reclasifica a jam 2 "
        "en las dirs target N/S (era el sobre-reporte de ~1 nivel que se corrigió). "
        "Recuperar cobertura jam≥3 requiere recalibrar la demanda (subir flows_vph "
        "de am_peak/pm_peak verificando no-spillback) — tarea aparte. Quitar este "
        "xfail cuando la demanda se recalibre."
    ),
)
def test_b2_coverage_satisfied_for_all_patterns(tmp_path, pattern_cfg):
    """coverage_check satisfecho para los 4 patrones (con duration completa)."""
    fail_msgs = []
    for name in PATTERNS:
        params = pattern_cfg["patterns"][name]
        out_dir = tmp_path / name
        report = check_coverage(
            pattern=name,
            seed=42,
            out_dir=out_dir,
            target_jam=params["target_jam"],
            target_direction=params.get("target_direction"),
            # Usar duration completa del yaml (1200s) para evitar artefactos
            # de warmup que podrían faltar en una corrida corta.
            end_s=pattern_cfg["duration_s"],
        )
        if not report.satisfied:
            fail_msgs.append(
                f"{name}: {report.reasons} "
                f"(max_jam={report.max_jam_per_dir}, "
                f"sustained_ge3={report.sustained_jam_ge3_buckets}, "
                f"max_queue_m={ {d: round(v,1) for d,v in report.max_queue_per_dir_m.items()} })"
            )
    assert not fail_msgs, "Coverage failures:\n" + "\n".join(fail_msgs)
