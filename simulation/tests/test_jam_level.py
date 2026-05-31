"""Guardas de regresión del mapeo jam_level (D-009, escala Waze 2026-05-31).

Cubre dos cosas que el resto de la suite no fijaba:
- Los cortes nuevos 80/60/40/20 (frontera por frontera), para ratio_to_jam_level
  y sumo_to_jam_level.
- La distinción "sin observación" (calle vacía → jam 0) vs "velocidad 0 con
  vehículos" (atasco real → jam 5), en generate.py y coverage_check.py, usando
  sampledSeconds como señal de presencia.
"""
from __future__ import annotations

from pathlib import Path

from cerebrovial_simulation import coverage_check
from cerebrovial_simulation.dataset import generate as gen_mod
from cerebrovial_simulation.dataset import schema
from cerebrovial_simulation.jam_level import ratio_to_jam_level, sumo_to_jam_level


VMAX = schema.VMAX_MPS  # 13.89


# ───────────────────────── cortes nuevos (Waze 80/60/40/20) ─────────────────────────

def test_ratio_cuts_boundaries():
    """Bandas: jam0 ≥0.80 | jam1 [0.60,0.80) | jam2 [0.40,0.60) | jam3 [0.20,0.40) | jam4 (0,0.20) | jam5 ≤0."""
    assert ratio_to_jam_level(0.81) == 0
    assert ratio_to_jam_level(0.80) == 0
    assert ratio_to_jam_level(0.799) == 1
    assert ratio_to_jam_level(0.60) == 1
    assert ratio_to_jam_level(0.599) == 2
    assert ratio_to_jam_level(0.40) == 2
    assert ratio_to_jam_level(0.399) == 3
    assert ratio_to_jam_level(0.20) == 3
    assert ratio_to_jam_level(0.199) == 4
    assert ratio_to_jam_level(0.0001) == 4
    assert ratio_to_jam_level(0.0) == 5
    assert ratio_to_jam_level(-0.5) == 5


def test_sumo_cuts_boundaries():
    """Mismos cortes vía velocidad/VMAX. mean_speed==0 → 5 (terminal sin cambio)."""
    assert sumo_to_jam_level(0.0, VMAX) == 5
    assert sumo_to_jam_level(VMAX * 0.80, VMAX) == 0
    assert sumo_to_jam_level(VMAX * 0.799, VMAX) == 1
    assert sumo_to_jam_level(VMAX * 0.60, VMAX) == 1
    assert sumo_to_jam_level(VMAX * 0.40, VMAX) == 2
    assert sumo_to_jam_level(VMAX * 0.20, VMAX) == 3
    assert sumo_to_jam_level(VMAX * 0.10, VMAX) == 4


# ───────────────────────── ausente vs cero (Issue 2) ─────────────────────────

# Un solo bucket (end=60) con las 4 direcciones en estados distintos:
#   N_in: vacío (sampledSeconds=0, SIN speed)         → jam 0, ratio 1.0  (calle libre)
#   S_in: vehículos parados (sampled>0, speed=0)      → jam 5             (atasco real)
#   E_in: speed=11.0 (ratio≈0.792)                    → jam 1
#   W_in: speed=2.0  (ratio≈0.144)                    → jam 4
_EDGEDATA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<meandata>
  <interval begin="0.00" end="60.00" id="aggE">
    <edge id="N_in" sampledSeconds="0.00" departed="0" arrived="0" entered="0" left="0"/>
    <edge id="S_in" sampledSeconds="120.00" speed="0.00" departed="5" arrived="0" entered="0" left="0"/>
    <edge id="E_in" sampledSeconds="50.00" speed="11.00" departed="3" arrived="0" entered="0" left="0"/>
    <edge id="W_in" sampledSeconds="50.00" speed="2.00" departed="3" arrived="0" entered="0" left="0"/>
  </interval>
</meandata>
"""

# lanearea mínimo: sin colas relevantes (queue_length=0 para todas).
_LANEAREA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<meandata>
  <interval begin="0.00" end="60.00" id="LA_N_0" maxJamLengthInMeters="0.00"/>
</meandata>
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_generate_distingue_ausente_de_cero(tmp_path):
    """generate._build_rows_for_run: vacío→jam 0/ratio 1.0; speed0 con veh→jam 5."""
    edgedata = _write(tmp_path, "edgedata.xml", _EDGEDATA_XML)
    lanearea = _write(tmp_path, "lanearea.xml", _LANEAREA_XML)
    rows = gen_mod._build_rows_for_run("test", 1, edgedata, lanearea)
    by_dir = {r["direction"]: r for r in rows}

    # Calle vacía: NO jam 5, sino flujo libre.
    assert by_dir["N"]["jam_level"] == 0
    assert by_dir["N"]["ratio"] == 1.0
    assert abs(by_dir["N"]["mean_speed_mps"] - VMAX) < 1e-6
    assert by_dir["N"]["n_vehicles"] == 0

    # Atasco real: vehículos presentes y parados.
    assert by_dir["S"]["jam_level"] == 5
    assert by_dir["S"]["n_vehicles"] == 5

    # Observaciones genuinas mapean por los cortes nuevos.
    assert by_dir["E"]["jam_level"] == 1   # 11.0/13.89 ≈ 0.792 ∈ [0.60,0.80)
    assert by_dir["W"]["jam_level"] == 4   # 2.0/13.89  ≈ 0.144 ∈ (0,0.20)


def test_coverage_check_trata_vacio_como_flujo_libre(tmp_path):
    """coverage_check._parse_edgedata: edge vacío → speed=VMAX → sumo_to_jam_level→0."""
    edgedata = _write(tmp_path, "edgedata.xml", _EDGEDATA_XML)
    per_dir = coverage_check._parse_edgedata(edgedata)

    # N_in vacío: speed sustituido por VMAX (no -1/0 que daría jam 5).
    n_speed = per_dir["N"][0][1]
    assert abs(n_speed - coverage_check.VMAX_MPS) < 1e-6
    assert sumo_to_jam_level(n_speed, coverage_check.VMAX_MPS) == 0

    # S_in parado: speed 0 conservado → jam 5.
    s_speed = per_dir["S"][0][1]
    assert s_speed == 0.0
    assert sumo_to_jam_level(s_speed, coverage_check.VMAX_MPS) == 5
