"""Barrido de CAPACIDAD — caracterización demora vs demanda, fijo vs per-node.

Responde, con número y figura, cuánto de la demora del corredor es ARREGLABLE POR
TEMPORIZADO vs IRREDUCIBLE POR CAPACIDAD. NO es un "controlador perfecto"
(contestable): es un barrido de demanda que revela el techo de capacidad
(throughput) y muestra dónde se para el per-node respecto de él. Caracterización
que refuerza la conclusión cerrada de IE05 (per-node óptimo), no la reabre.

Solo varía demand_scale; NO toca tiempos de control. Dos brazos pareados por
semilla, misma demanda determinista:
  fijo      — TLLogic estático embebido en corredor_larco.net.xml.
  per-node  — Max-Pressure simplificado P=s·q (downstream OFF). Reproduce IE05.

Dos métricas por punto:
  (a) W_RED puerta-a-puerta (w_red_door2door_s_postwarm) — lo arreglable por temporizado.
  (b) THROUGHPUT — techo de capacidad:
        throughput_net (veh/h, arrived×3600/sim_dur) [principal]
        schell_discharge_vph — descarga MEDIDA cruzando Schell al norte
        (Σ nVehEntered del detector e2_schDc post-warmup ×3600/T_pw; NO derivado del
        completion_ratio, que se sesga a la baja en sobre-saturación).

Compuerta de integridad (banco no derivó): a scale=1.0 el per-node fresco reproduce
el W_RED histórico de IE05 (peak_s_n_adaptive_seed{s}) EXACTO al dígito (<1e-9).
Se corre PRIMERO; si falla → HARD STOP, no se corre el resto.

HONESTIDAD DEL TECHO: SUMO teletransporta autos atascados → infla throughput y
deprime demora en sobre-saturación. teleports_total se registra por punto; μ̂ se
estima en plot_capacity SOLO en el régimen teleport≈0 (este script solo mide y reporta).

Uso:
    invoke up      # motor vivo en :8001 con el código de Etapa 1
    # barrido completo (default 0.6..2.0 paso 0.2, semillas 42-46):
    simulation/.venv/bin/python scripts/sweep_capacity.py
    # smoke (1 escala, 1 semilla) — valida brazos + compuerta de integridad:
    simulation/.venv/bin/python scripts/sweep_capacity.py --scales 1.0 --seeds 42
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
CONF = SIM_ROOT / "conf" / "corredor_larco"
CFG = CONF / "peak_s_n.sumocfg"
DET = CONF / "corredor_larco.det.add.xml"
PARAMS = CONF / "demand_params.yaml"
DEFAULT_ROUTES = CONF / "routes" / "peak_s_n.rou.xml"
DATA = SIM_ROOT / "data" / "corredor_larco"
OUT_CSV = DATA / "capacity_sweep" / "capacity_sweep.csv"
SUMO_PKG = SIM_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "sumo"
HEALTH = os.environ.get("ENGINE_HEALTH", "http://localhost:8001/control/health")

END, WARMUP = 1800, 600
T_PW = END - WARMUP  # ventana post-warmup (s) — consistente con cómo se mide W_RED

# Detector de descarga por Schell: e2_schDc cubre 279893875#4 (Schell→Diez Canseco),
# inmediatamente aguas abajo de la línea de pare de Schell. nVehEntered = autos que
# cruzaron Schell al norte. Carril 0 (bus/bici) lee ~0; se suma sobre los 3 carriles.
SCHDC = ["e2_schDc_0", "e2_schDc_1", "e2_schDc_2"]
# Edge que define el movimiento del corredor que cruza Schell al norte (para λ_corr).
CORR_EDGE = "279893875#4"

os.environ.setdefault("SUMO_HOME", str(SUMO_PKG))
os.environ["PATH"] = f"{SUMO_PKG}/bin:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(SIM_ROOT / "src"))
sys.path.insert(0, str(HERE))  # para reusar generate_corredor_routes en proceso

import generate_corredor_routes as gen_routes  # noqa: E402
from cerebrovial_simulation.kpis import collect_corredor  # noqa: E402
from cerebrovial_simulation.traci_adapter import corredor_adaptive  # noqa: E402


def _f(v) -> float:
    return 0.0 if v in (None, "") else float(v)


def _scale_tag(scale: float) -> int:
    return int(round(scale * 100))


# ---------------------------------------------------------------------------
# Demanda
# ---------------------------------------------------------------------------

def _demand_vph(scale: float, corr_only: bool) -> int:
    """veh/h ofrecidos a escala `scale`. corr_only=True → solo rutas que cruzan Schell
    al norte (las que incluyen CORR_EDGE) = λ_corr para el overlay teórico."""
    cfg = yaml.safe_load(PARAMS.read_text())
    routes, flows = cfg["routes"], cfg["flows_vph"]
    total = 0
    for fid, vph in flows.items():
        if vph <= 0:
            continue
        if corr_only and CORR_EDGE not in routes.get(fid, "").split():
            continue
        total += round(vph * scale)
    return total


def _regen_routes(scale: float) -> None:
    """Regenera el routes default a `scale` (ambos brazos lo leen vía el sumocfg)."""
    cfg = yaml.safe_load(PARAMS.read_text())
    gen_routes.generate(cfg, scale=scale, out_file=DEFAULT_ROUTES)


def _schell_discharge_vph(out_dir: Path) -> float:
    """Descarga MEDIDA cruzando Schell (Σ nVehEntered e2_schDc, buckets post-warmup)."""
    path = out_dir / "lanearea.xml"
    if not path.exists():
        return 0.0
    root = ET.parse(path).getroot()
    entered = 0.0
    for iv in root.findall("interval"):
        if iv.attrib.get("id") not in SCHDC:
            continue
        if _f(iv.attrib.get("end")) <= WARMUP:
            continue
        entered += _f(iv.attrib.get("nVehEntered"))
    return entered * 3600.0 / T_PW if T_PW > 0 else 0.0


# ---------------------------------------------------------------------------
# Brazos
# ---------------------------------------------------------------------------

def _run_fixed(seed: int, out_dir: Path):
    """fijo = TLLogic embebido en net.xml (sin additional-file de tlLogic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DET.name).write_bytes(DET.read_bytes())
    env = dict(os.environ, SUMO_HOME=str(SUMO_PKG))
    args = [
        str(SUMO_PKG / "bin" / "sumo"), "-c", str(CFG),
        "--additional-files", str(out_dir / DET.name),
        "--seed", str(seed), "--end", str(END),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
        "--no-step-log", "--no-warnings",
    ]
    p = subprocess.run(args, cwd=out_dir, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"sumo fijo seed={seed} falló:\n{p.stderr[-800:]}")
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS)


def _run_pernode(seed: int, out_dir: Path):
    """per-node = Max-Pressure simplificado (downstream OFF), vía motor :8001."""
    stats = corredor_adaptive.run_adaptive(seed=seed, end_s=END, out_dir=out_dir, downstream=False)
    errs = sum(len(s["errors"]) for s in stats.values())
    if errs:
        print(f"  ⚠️ seed {seed} [per-node]: {errs} errores de motor", file=sys.stderr)
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS)


def _edge_max(kpis, label: str) -> float:
    for eq in kpis.edge_queues:
        if eq.label == label:
            return eq.max_queue_m
    return 0.0


def _row(scale: float, arm: str, seed: int, kpis, out_dir: Path) -> dict:
    return {
        "scale": scale,
        "demand_vph": _demand_vph(scale, corr_only=False),
        "demand_corr_vph": _demand_vph(scale, corr_only=True),
        "arm": arm,
        "seed": seed,
        "w_red": round(kpis.w_red_door2door_s_postwarm, 4),
        "w_red_wait": round(kpis.w_red_wait_s_postwarm, 4),
        "w_red_inside": round(kpis.w_red_inside_s_postwarm, 4),
        "throughput_net": round(kpis.throughput_veh_per_h, 2),
        "schell_discharge_vph": round(_schell_discharge_vph(out_dir), 2),
        "completion": round(kpis.completion_ratio, 4),
        "teleports": kpis.teleports_total,
        "never_inserted": kpis.never_inserted_veh,
        "benSch_max": round(_edge_max(kpis, "benavides_to_schell(INT)"), 2),
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

FIELDS = ["scale", "demand_vph", "demand_corr_vph", "arm", "seed", "w_red",
          "w_red_wait", "w_red_inside", "throughput_net", "schell_discharge_vph",
          "completion", "teleports", "never_inserted", "benSch_max"]


def _write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ CSV: {OUT_CSV.relative_to(SIM_ROOT)}  ({len(rows)} filas)")


def _integrity_check(seed: int, w_fresh: float) -> tuple[bool, float | None]:
    """per-node fresco @1.0 vs histórico IE05, exacto al dígito."""
    hist_dir = DATA / f"peak_s_n_adaptive_seed{seed}_end{END}"
    if not hist_dir.exists():
        return False, None
    w_hist = collect_corredor.collect_corredor(hist_dir, WARMUP, PARAMS).w_red_door2door_s_postwarm
    return abs(w_fresh - w_hist) < 1e-9, w_hist


def main(scales: list[float], seeds: list[int]) -> int:
    # Motor vivo (brazo per-node).
    try:
        assert requests.get(HEALTH, timeout=5).status_code == 200
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: motor no responde 200 en {HEALTH} ({exc}). `invoke up` primero.", file=sys.stderr)
        return 1

    # 1.0 primero → compuerta de integridad ANTES del resto del barrido.
    ordered = ([1.0] + [s for s in scales if s != 1.0]) if 1.0 in scales else list(scales)
    rows: list[dict] = []
    try:
        for scale in ordered:
            tag = _scale_tag(scale)
            _regen_routes(scale)
            print(f"\n{'='*78}\nESCALA {scale:.2f}  (demanda {_demand_vph(scale, False)} veh/h, "
                  f"corredor {_demand_vph(scale, True)} veh/h)\n{'='*78}", file=sys.stderr)
            integ: list[tuple[int, float, float | None, bool]] = []
            for seed in seeds:
                fx_dir = DATA / f"cap_scale{tag}_fijo_seed{seed}_end{END}"
                pn_dir = DATA / f"cap_scale{tag}_pernode_seed{seed}_end{END}"
                fx = _run_fixed(seed, fx_dir)
                pn = _run_pernode(seed, pn_dir)
                rows.append(_row(scale, "fijo", seed, fx, fx_dir))
                rows.append(_row(scale, "pernode", seed, pn, pn_dir))
                w_pn = pn.w_red_door2door_s_postwarm
                print(f"  seed {seed}: W_RED fijo={fx.w_red_door2door_s_postwarm:6.1f}  "
                      f"per-node={w_pn:6.1f}  | thr_net pn={pn.throughput_veh_per_h:6.0f}  "
                      f"descargaSchell pn={_schell_discharge_vph(pn_dir):6.0f}  "
                      f"teleports pn={pn.teleports_total}", file=sys.stderr)
                if scale == 1.0:
                    ok, w_hist = _integrity_check(seed, w_pn)
                    integ.append((seed, w_pn, w_hist, ok))

            if scale == 1.0:
                print("\n[COMPUERTA DE INTEGRIDAD — per-node @1.0 = IE05, exacto al dígito]", file=sys.stderr)
                bad = []
                for seed, w_pn, w_hist, ok in integ:
                    hs = f"{w_hist:.4f}" if w_hist is not None else "FALTA dir histórico"
                    print(f"  seed {seed}: fresco={w_pn:.4f}  histórico={hs}  "
                          f"{'OK' if ok else '✗ MISMATCH'}", file=sys.stderr)
                    if not ok:
                        bad.append(seed)
                if bad:
                    print(f"\n✗ HARD STOP: el banco derivó en semillas {bad}. "
                          f"No se corre el resto del barrido.", file=sys.stderr)
                    _write_csv(rows)
                    return 2
                print("  → integridad OK; sigue el barrido completo.", file=sys.stderr)
    finally:
        # Restaurar el routes default a scale=1.0 (no dejar sucio el conf).
        _regen_routes(1.0)
        print("\n(routes default restaurado a scale=1.0)", file=sys.stderr)

    _write_csv(rows)
    return 0


def _parse_args() -> argparse.Namespace:
    default_scales = [round(0.6 + 0.2 * i, 1) for i in range(8)]  # 0.6..2.0
    ap = argparse.ArgumentParser(description="Barrido de capacidad demanda↔control (fijo vs per-node).")
    ap.add_argument("--scales", type=float, nargs="+", default=default_scales)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    return ap.parse_args()


if __name__ == "__main__":
    ns = _parse_args()
    sys.exit(main(ns.scales, ns.seeds))
