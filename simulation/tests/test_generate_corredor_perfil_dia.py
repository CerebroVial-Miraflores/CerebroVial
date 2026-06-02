"""Tests del generador perfil-día×OD del corredor (generate_corredor_perfil_dia.py).

El script vive en simulation/scripts/ (no es paquete) → se carga vía importlib desde
su path, así el test es independiente del CWD y corre con el venv raíz (stdlib + PyYAML).
"""
from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


_SIM_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _SIM_ROOT / "scripts" / "generate_corredor_perfil_dia.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_corredor_perfil_dia", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(tmp_path, profile="laborable"):
    mod = _load_module()
    with open(mod.PARAMS) as f:
        demand_cfg = yaml.safe_load(f)
    with open(mod.PERFIL) as f:
        perfil_cfg = yaml.safe_load(f)
    out = tmp_path / "perfil.rou.xml"
    mod.generate(demand_cfg, perfil_cfg, profile=profile, out_file=out)
    return out, demand_cfg, perfil_cfg


def _parse_flows(out_file):
    """Lista de dicts {idx, flow_id, vph, begin, end} desde el .rou.xml."""
    root = ET.parse(out_file).getroot()
    flows = []
    for fl in root.findall("flow"):
        fid = fl.get("id")
        route = fl.get("route")
        idx = int(fid.rsplit("_", 1)[1])
        # el id debe ser exactamente f_{route}_{idx}
        assert fid == f"f_{route}_{idx}", fid
        flows.append({
            "idx": idx,
            "flow_id": route,
            "vph": int(fl.get("vehsPerHour")),
            "begin": int(fl.get("begin")),
            "end": int(fl.get("end")),
        })
    return flows


def test_diez_fases_con_todos_los_flows_positivos(tmp_path):
    out, demand_cfg, perfil_cfg = _generate(tmp_path)
    flows = _parse_flows(out)
    phases = perfil_cfg["profiles"]["laborable"]["phases"]
    base = demand_cfg["flows_vph"]

    assert len(phases) == 10
    # nº de flows esperado = por fase, los OD con round(base*scale) > 0.
    expected_total = sum(
        1
        for ph in phases
        for b in base.values()
        if round(b * ph["scale"]) > 0
    )
    assert len(flows) == expected_total
    # con esta calibración todos los scale dan vph>0 para los 11 OD en las 10 fases.
    assert expected_total == 10 * len(base) == 110

    idxs = sorted({fl["idx"] for fl in flows})
    assert idxs == list(range(10))


def test_begin_end_y_vph_por_fase(tmp_path):
    out, demand_cfg, perfil_cfg = _generate(tmp_path)
    flows = _parse_flows(out)
    phases = perfil_cfg["profiles"]["laborable"]["phases"]
    base = demand_cfg["flows_vph"]

    for fl in flows:
        ph = phases[fl["idx"]]
        # begin/end = h * 3600
        assert fl["begin"] == int(ph["begin_h"] * 3600)
        assert fl["end"] == int(ph["end_h"] * 3600)
        # vph = round(base * scale)
        assert fl["vph"] == round(base[fl["flow_id"]] * ph["scale"])


def test_turn_ratios_preservados_entre_fases(tmp_path):
    """Todos los OD de una misma fuente escalan por el mismo factor → la proporción
    entre dos flows de la misma fuente es constante entre fases (módulo redondeo)."""
    out, demand_cfg, perfil_cfg = _generate(tmp_path)
    flows = _parse_flows(out)
    base = demand_cfg["flows_vph"]

    # Dos OD de la fuente Larco Sur.
    a, b = "larco_S_N", "larco_S_benE"
    base_ratio = base[a] / base[b]
    by_phase = {}
    for fl in flows:
        by_phase.setdefault(fl["idx"], {})[fl["flow_id"]] = fl["vph"]

    seen_ratios = []
    for idx, d in by_phase.items():
        seen_ratios.append(d[a] / d[b])
    # cada proporción se mantiene cerca de la base (tolerancia por redondeo).
    for r in seen_ratios:
        assert abs(r - base_ratio) / base_ratio < 0.05
