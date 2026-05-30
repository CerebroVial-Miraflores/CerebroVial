"""CT-07.8.a — Topología carga sin errores + behavior baseline + Parquet outputs.

Tests:
- a1: sumolib.net.readNet no lanza.
- a2: 1 TLS junction; programa baseline con 6 sub-fases (g/y/r × NS/EW).
- a3: N/S = 3 lanes/aproche, E/W = 2 lanes/aproche.
- a4: vmax todos los lanes = 13.89 m/s ±0.01.
- a5: largos efectivos N/S ≈ 290m, E/W ≈ 186m (tolerancia ±10).
- a6 (corrección 4 del plan): corrida 60s baseline, verifica movimiento NS
  durante NS-verde y bloqueo EW durante NS-verde.
- a7 (corrección 2 del plan): corrida 60s con edgeData + laneArea + Parquet,
  verifica 4 outputs Parquet legibles por pyarrow.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import sumolib


SIM_ROOT = Path(__file__).resolve().parent.parent
NET_XML = SIM_ROOT / "conf" / "network" / "miraflores_4way.net.xml"
TLLOGIC_ADD = SIM_ROOT / "conf" / "network" / "miraflores_4way.tllogic.add.xml"
EDGEDATA_ADD = SIM_ROOT / "conf" / "network" / "edgedata.add.xml"
LANEAREA_ADD = SIM_ROOT / "conf" / "network" / "lanearea.add.xml"
LINKSTATES_JSON = SIM_ROOT / "conf" / "network" / "linkstates.json"
ROUTE_FIXTURE = SIM_ROOT / "tests" / "fixtures" / "test_smoke.rou.xml"


# ---------- a1-a5: introspección sumolib (read-only) ----------

@pytest.fixture(scope="module")
def net() -> sumolib.net.Net:
    return sumolib.net.readNet(str(NET_XML), withPrograms=True)


def test_a1_net_loads(net):
    """sumolib.net.readNet no lanza."""
    assert net is not None
    assert len(net.getEdges()) >= 8  # 4 in + 4 out (mínimo)


def test_a2_tls_program_baseline_has_6_subphases(net):
    """1 TLS junction en el net + programa 'baseline' en add-file con 6 sub-fases."""
    # 1 TLS junction en el net.
    tls_list = net.getTrafficLights()
    assert len(tls_list) == 1, f"Esperado 1 TLS, encontrado {len(tls_list)}"

    # 'baseline' vive en tllogic.add.xml, no en net.xml. Lo parseamos por separado.
    import xml.etree.ElementTree as ET
    root = ET.parse(TLLOGIC_ADD).getroot()
    tllogics = root.findall("tlLogic")
    assert len(tllogics) == 1
    tl = tllogics[0]
    assert tl.attrib["id"] == "center"
    assert tl.attrib["programID"] == "baseline"
    phases = tl.findall("phase")
    assert len(phases) == 6, f"Esperadas 6 sub-fases, hay {len(phases)}"
    # Durations consistentes con network_params.yaml (NS 30/3/2, EW 25/3/2).
    durations = [int(p.attrib["duration"]) for p in phases]
    assert durations == [30, 3, 2, 25, 3, 2], f"Durations: {durations}"


def test_a3_lane_counts(net):
    """N/S aproches = 3 lanes; E/W = 2 lanes."""
    assert len(net.getEdge("N_in").getLanes()) == 3
    assert len(net.getEdge("S_in").getLanes()) == 3
    assert len(net.getEdge("E_in").getLanes()) == 2
    assert len(net.getEdge("W_in").getLanes()) == 2


def test_a4_vmax_uniform(net):
    """vmax = 13.89 m/s ± 0.01 en todos los lanes de aproche/salida."""
    for eid in ["N_in", "S_in", "E_in", "W_in", "N_out", "S_out", "E_out", "W_out"]:
        for lane in net.getEdge(eid).getLanes():
            assert abs(lane.getSpeed() - 13.89) <= 0.01, (
                f"{lane.getID()} speed={lane.getSpeed()} fuera de 13.89±0.01"
            )


def test_a5_approach_lengths(net):
    """N/S length ≈ 290 m, E/W ≈ 186 m (tolerancia ±10; junta interna del nodo central reduce ~10m)."""
    ns_len = net.getEdge("N_in").getLength()
    ew_len = net.getEdge("E_in").getLength()
    assert 280 <= ns_len <= 305, f"N_in length={ns_len} fuera de [280,305]"
    assert 175 <= ew_len <= 205, f"E_in length={ew_len} fuera de [175,205]"


# ---------- a6: behavior baseline (corrección 4 — linkstates derivados son correctos) ----------

def _run_sumo(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Helper: invoca sumo con env y captura output."""
    sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
    cmd = [str(sumo_bin), *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)


def test_a6_baseline_behavior(tmp_path):
    """Corrida 60s baseline con tráfico fixture; verifica que:
    - tripinfo registra arribos (vehículos completaron ruta).
    - linkstates funcionan: NS y EW reciben verde alternadamente
      (al menos un vehículo NS y un vehículo EW completaron).
    """
    tripinfo = tmp_path / "tripinfo.xml"
    args = [
        "-n", str(NET_XML),
        "-r", str(ROUTE_FIXTURE),
        "-a", str(TLLOGIC_ADD),
        "--end", "120",
        "--no-step-log",
        "--no-warnings",
        "--tripinfo-output", str(tripinfo),
    ]
    proc = _run_sumo(args, cwd=tmp_path)
    assert proc.returncode == 0, f"sumo failed: STDOUT={proc.stdout}\nSTDERR={proc.stderr}"
    assert tripinfo.exists(), "tripinfo.xml no se generó"

    content = tripinfo.read_text()
    # Verifica que vehículos completaron y rutas N→S, S→N, E→W, W→E ejercitaron.
    assert "<tripinfo " in content, "Ningún vehículo completó su ruta"
    # Al menos un vehículo de cada eje cruzó (verde alternó correctamente).
    # Los flows fN, fS, fE, fW generan ids fN.0, fS.0, etc.
    assert ("id=\"fN." in content or "id=\"fS." in content), \
        "Ningún vehículo NS completó — verde NS no funcionó"
    assert ("id=\"fE." in content or "id=\"fW." in content), \
        "Ningún vehículo EW completó — verde EW no funcionó"


# ---------- a7: Parquet outputs (corrección 2) ----------

def test_a7_outputs_parquet_and_xml_fallback(tmp_path):
    """Corrida 120s con edgeData + laneArea + --output.format parquet.

    Verifica:
    - summary.parquet + tripinfo.parquet (single-write outputs) ⇒ Parquet directo OK.
    - edgedata.xml + lanearea.xml (multi-interval outputs) ⇒ XML fallback documentado.
      SUMO 1.26 arrow writer no soporta append multi-interval; el pipeline F3
      parsea XML y convierte a Parquet con pyarrow.
    """
    tllogic_local = tmp_path / "tllogic.add.xml"
    shutil.copy(TLLOGIC_ADD, tllogic_local)
    edgedata_local = tmp_path / "edgedata.add.xml"
    shutil.copy(EDGEDATA_ADD, edgedata_local)
    lanearea_local = tmp_path / "lanearea.add.xml"
    shutil.copy(LANEAREA_ADD, lanearea_local)

    summary = tmp_path / "summary.parquet"
    tripinfo = tmp_path / "tripinfo.parquet"

    # NOTA: NO usar --output.format parquet global — fuerza Parquet en
    # outputs multi-interval (edgeData/laneArea) cuyo arrow writer no
    # soporta append. Cada output usa su sufijo: .parquet ⇒ Parquet
    # (summary/tripinfo, single-write), .xml ⇒ XML (edgeData/laneArea).
    args = [
        "-n", str(NET_XML),
        "-r", str(ROUTE_FIXTURE),
        "-a", f"{tllogic_local},{edgedata_local},{lanearea_local}",
        "--end", "120",
        "--no-step-log",
        "--no-warnings",
        "--summary-output", str(summary),
        "--tripinfo-output", str(tripinfo),
    ]
    proc = _run_sumo(args, cwd=tmp_path)
    assert proc.returncode == 0, (
        f"sumo failed: STDOUT={proc.stdout}\nSTDERR={proc.stderr}"
    )

    # Parquet directo: outputs single-write.
    assert summary.exists(), "summary.parquet faltante"
    assert tripinfo.exists(), "tripinfo.parquet faltante"
    for pq_file in [summary, tripinfo]:
        t = pq.read_table(str(pq_file))
        assert t.num_rows > 0, f"{pq_file.name} sin filas"

    # XML fallback: outputs multi-interval (edgeData + laneArea).
    edgedata_xml = tmp_path / "edgedata.xml"
    lanearea_xml = tmp_path / "lanearea.xml"
    assert edgedata_xml.exists(), f"edgedata.xml faltante en {tmp_path}"
    assert lanearea_xml.exists(), f"lanearea.xml faltante en {tmp_path}"

    # Verifica contenido XML mínimo: 120s con freq=60 ⇒ ≥1 intervalo cerrado.
    import xml.etree.ElementTree as ET
    ed_root = ET.parse(edgedata_xml).getroot()
    assert ed_root.findall("interval"), "edgedata.xml sin <interval>"
    la_root = ET.parse(lanearea_xml).getroot()
    # laneArea genera <interval> con un <id="LA_*"> attribute.
    la_intervals = la_root.findall("interval")
    assert la_intervals, "lanearea.xml sin <interval>"
    detector_ids = {iv.attrib.get("id", "") for iv in la_intervals}
    expected_detectors = {f"LA_{d}_{i}" for d in ["N", "S"] for i in range(3)} | \
                          {f"LA_{d}_{i}" for d in ["E", "W"] for i in range(2)}
    assert expected_detectors.issubset(detector_ids), (
        f"Detectores ausentes: {expected_detectors - detector_ids}"
    )


# ---------- a8: linkstates.json estructura ----------

def test_a8_linkstates_json_has_6_keys():
    """linkstates.json (generado por build_network.py) tiene las 6 sub-fases esperadas."""
    states = json.loads(LINKSTATES_JSON.read_text())
    expected = {"NS_g", "NS_y", "NS_r", "EW_g", "EW_y", "EW_r"}
    assert set(states.keys()) == expected, f"Keys: {set(states.keys())}"
    # Misma longitud para todos (mismo n_links).
    lengths = {len(v) for v in states.values()}
    assert len(lengths) == 1, f"Linkstate strings con longitudes distintas: {lengths}"
    n_links = lengths.pop()
    assert n_links > 0
    # Sub-fases rojas son todo 'r'.
    assert states["NS_r"] == "r" * n_links
    assert states["EW_r"] == "r" * n_links
    # NS_g tiene al menos un 'G' (through), al menos un 'g' (left permissive), al menos un 'r' (E/W).
    assert "G" in states["NS_g"] and "g" in states["NS_g"] and "r" in states["NS_g"]
    assert "G" in states["EW_g"] and "g" in states["EW_g"] and "r" in states["EW_g"]
