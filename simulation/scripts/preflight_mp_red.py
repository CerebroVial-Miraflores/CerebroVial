"""Pre-flight de Etapa 2 (Max Pressure de red) — 1 semilla, ANTES del barrido 42-51.

Descarta el modo de falla que el análisis de fin de sweep NO puede ver: un Δ≈0
entre MP-red y per-node tiene dos causas indistinguibles desde la tabla —
  (a) null real: Schell es binding constraint, retener no ayuda.
  (b') no-op de cableado: --downstream no enchufó / payload vacío / MP-red corrió
       idéntico al per-node → Δ=0 ARTEFACTO.
La única forma de descartar (b') es verificar EN EJECUCIÓN que Benavides recibió la
cola de #1 y que el término operó. Eso es lo que hace este script.

Compuertas (todas deben pasar para soltar el barrido):
  A0  per-node (downstream OFF) reproduce el W_RED histórico de IE05 de ESA semilla,
      exacto al dígito. Si no → el harness derivó; PARAR (no es regresión del motor:
      Etapa 1 probó el motor bit-idéntico, así que un mismatch acá es no-motor).
  A1  el payload de Benavides-LARCO en el brazo downstream-ON trae downstream.queue
      fiel al halting de #1 (NO None/vacío). Aserción DURA.
  A2  como #1 satura ~99% (IE05, 10/10), al menos una decisión de Benavides difiere
      del per-node. Corolario de A1: si #1 satura y NADA cambia, el cable no enchufa.

Uso:
    invoke up   # motor vivo en :8001 (otra terminal)
    simulation/.venv/bin/python scripts/preflight_mp_red.py [SEED]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
CONF = SIM_ROOT / "conf" / "corredor_larco"
PARAMS = CONF / "demand_params.yaml"
DATA = SIM_ROOT / "data" / "corredor_larco"
SUMO_PKG = SIM_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "sumo"
HEALTH = os.environ.get("ENGINE_HEALTH", "http://localhost:8001/control/health")

END, WARMUP = 1800, 600

os.environ.setdefault("SUMO_HOME", str(SUMO_PKG))
os.environ["PATH"] = f"{SUMO_PKG}/bin:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(SIM_ROOT / "src"))

from cerebrovial_simulation.kpis import collect_corredor  # noqa: E402
from cerebrovial_simulation.traci_adapter import corredor_adaptive  # noqa: E402
from cerebrovial_simulation.traci_adapter.engine_client import recommend as _real_recommend  # noqa: E402

BEN = "larco_benavides"


def _make_recorder(log: list[dict]):
    """Envuelve recommend(): registra (iid, phases_in, rec_out) sin alterar conducta."""
    def _rec(*, intersection_id, timestamp, phase_flows, lost_time=10.0, **kw):
        rec = _real_recommend(
            intersection_id=intersection_id, timestamp=timestamp,
            phase_flows=phase_flows, lost_time=lost_time, **kw,
        )
        log.append({"iid": intersection_id, "phases": phase_flows, "rec": rec})
        return rec
    return _rec


def _ben_larco_decisions(log: list[dict]) -> list[tuple]:
    """Secuencia de decisiones de Benavides: (next_phase, green_LARCO redondeado)."""
    out = []
    for e in log:
        if e["iid"] != BEN:
            continue
        rec = e["rec"]
        g = next((t["green"] for t in rec["phase_timings"] if t["phase_id"] == "LARCO"), None)
        out.append((rec.get("next_phase"), round(g, 3) if g is not None else None))
    return out


def _ben_larco_downstream(log: list[dict]) -> list[int]:
    """queue downstream que viajó en el payload de Benavides-LARCO (None si ausente)."""
    out = []
    for e in log:
        if e["iid"] != BEN:
            continue
        larco = next((p for p in e["phases"] if p["phase_id"] == "LARCO"), None)
        if larco and larco.get("downstream"):
            out.append(larco["downstream"][0]["queue"])
        else:
            out.append(None)
    return out


def _w_red(out_dir: Path) -> float:
    return collect_corredor.collect_corredor(out_dir, WARMUP, PARAMS).w_red_door2door_s_postwarm


def main(seed: int) -> int:
    # Motor vivo.
    try:
        assert requests.get(HEALTH, timeout=5).status_code == 200
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: motor no responde 200 en {HEALTH} ({exc}). `invoke up` primero.", file=sys.stderr)
        return 1

    # Demanda determinista (idéntica a la del barrido y a la histórica de IE05).
    import subprocess
    subprocess.run([str(SIM_ROOT / ".venv/bin/python"), str(HERE / "generate_corredor_routes.py")],
                   check=True, capture_output=True)

    print(f"\n{'='*78}\nPRE-FLIGHT Max Pressure de red — seed {seed} | end={END} warmup={WARMUP}\n{'='*78}")

    # --- A0: per-node fresco reproduce el W_RED histórico exacto ---
    hist_dir = DATA / f"peak_s_n_adaptive_seed{seed}_end{END}"
    w_hist = _w_red(hist_dir) if hist_dir.exists() else None

    pernode_dir = DATA / f"preflight_pernode_seed{seed}_end{END}"
    log_pn: list[dict] = []
    corredor_adaptive.run_adaptive(seed=seed, end_s=END, out_dir=pernode_dir,
                                   engine_recommend_fn=_make_recorder(log_pn), downstream=False)
    w_pn = _w_red(pernode_dir)

    a0_ok = w_hist is not None and abs(w_pn - w_hist) < 1e-9
    print("\n[A0] no-regresión per-node (W_RED door2door):")
    print(f"     histórico IE05 = {w_hist if w_hist is not None else 'FALTA dir histórico'}")
    print(f"     per-node fresco = {w_pn}")
    print(f"     → {'OK exacto al dígito' if a0_ok else 'MISMATCH — el harness derivó, PARAR'}")

    # --- brazo downstream-ON ---
    downon_dir = DATA / f"preflight_downon_seed{seed}_end{END}"
    log_on: list[dict] = []
    corredor_adaptive.run_adaptive(seed=seed, end_s=END, out_dir=downon_dir,
                                   engine_recommend_fn=_make_recorder(log_on), downstream=True)
    w_on = _w_red(downon_dir)

    # --- A1: payload de Benavides-LARCO trae downstream fiel a #1 (no None/vacío) ---
    down_q = _ben_larco_downstream(log_on)
    n_present = sum(1 for q in down_q if q is not None)
    n_positive = sum(1 for q in down_q if q is not None and q > 0)
    max_q = max((q for q in down_q if q is not None), default=None)
    a1_ok = n_present == len(down_q) and n_positive >= 1
    print("\n[A1] payload downstream de Benavides-LARCO (cola de #1 vía TraCI):")
    print(f"     ciclos con clave downstream: {n_present}/{len(down_q)}  | con queue>0: {n_positive}  | max queue={max_q}")
    print(f"     → {'OK payload fiel (no None, #1 con cola real)' if a1_ok else 'NO-OP: payload vacío/None → cable no enchufa'}")

    # Sanidad: el brazo per-node NUNCA debe traer downstream (control byte-idéntico).
    pn_has_down = any(q is not None for q in _ben_larco_downstream(log_pn))
    print(f"     control per-node sin downstream en payload: {'OK' if not pn_has_down else 'FALLA (no es control limpio)'}")

    # --- A2: al menos una decisión de Benavides difiere (corolario con #1 saturado) ---
    dec_pn = _ben_larco_decisions(log_pn)
    dec_on = _ben_larco_decisions(log_on)
    n = min(len(dec_pn), len(dec_on))
    n_diff = sum(1 for i in range(n) if dec_pn[i] != dec_on[i])
    a2_ok = n_diff >= 1 or dec_pn != dec_on
    print("\n[A2] decisiones de Benavides per-node vs downstream-ON:")
    print(f"     ciclos comparados={n}  difieren={n_diff}  (len pn={len(dec_pn)}, on={len(dec_on)})")
    print(f"     → {'OK el término operó (≥1 decisión cambió)' if a2_ok else 'NO-OP: #1 satura pero NINGUNA decisión cambió'}")

    print(f"\n{'-'*78}")
    print(f"W_RED: per-node={w_pn:.3f}s  downstream-ON={w_on:.3f}s  (Δ informativo, NO veredicto: {w_pn - w_on:+.3f}s)")
    all_ok = a0_ok and a1_ok and a2_ok and not pn_has_down
    print(f"PRE-FLIGHT: {'TODO VERDE → soltar barrido 42-51' if all_ok else 'FALLA → PARAR, no correr el barrido'}")
    print(f"{'='*78}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    sys.exit(main(seed))
