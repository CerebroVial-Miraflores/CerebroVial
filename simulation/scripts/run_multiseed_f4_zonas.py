"""Orquestador F4 (DHU-030) — re-corrida multi-seed PAREADA fijo-vs-adaptativo sobre
Miraflores red completa, día completo, CON edgeData sobre los 9 aproches de los 3 cruces
PMU bifásicos saturados.

F4 NO re-valida el titular (el +20,7% global ya está validado por F5, su CSV vive en
``multiseed_54tls_resultado.csv``). F4 ENRIQUECE: re-corre los 10 seeds × 2 brazos para tener
todo consistente + el edgeData que F5 no capturó. De cada corrida saca:
  - tripinfo.parquet  → franjas GLOBALES (net_timeLoss de red entera, filtrado por depart)
  - edgeData zonas    → KPI POR ZONA (los 3 cruces × franjas)

Mismo control que F5 (54 adaptativos / 45 fijos), misma demanda (scale 1.1, pareada por seed).
edgeData es pura medición — no perturba la dinámica; los net_timeLoss globales de F4 deben
reproducir los de F5 (chequeo de sanidad que imprime el reproceso).

Escribe a dirs DEDICADOS ``f4_fijo_seed<NNN>`` / ``f4_adaptivo_seed<NNN>_end86400`` (NO pisa
los crudos sobrevivientes de F5). Reusa los runners de F2/F3b vía env (OUT_DIR_OVERRIDE +
EDGEDATA_ADD). ROBUSTO A CORTES: un brazo cuyo metrics.json + edgeData ya existen se saltea.

Uso:
    python run_multiseed_f4_zonas.py --seeds 42          # SMOKE de 1 seed (gate)
    python run_multiseed_f4_zonas.py --seeds 42-51       # los 10; retoma si cortó
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
EXP_DIR = SIM_ROOT / "conf" / "miraflores_red_completa"
RUNS = EXP_DIR / "runs"
FIJO_SH = HERE / "run_fijo_red.sh"
ADAP_SH = HERE / "run_adaptivo_red.sh"
CONF = EXP_DIR / "zonas_pmu_f4.json"

sys.path.insert(0, str(HERE))
from make_edgedata_zonas import OUT_NAME, build_addfile  # noqa: E402
from run_multiseed_red import _metrics, ensure_motor      # noqa: E402  (REUSO F5)


def _seedp(seed: int) -> str:
    return f"{seed:03d}"


def _fijo_dir(seed: int) -> Path:
    return RUNS / f"f4_fijo_seed{_seedp(seed)}"


def _adap_dir(seed: int) -> Path:
    return RUNS / f"f4_adaptivo_seed{_seedp(seed)}_end86400"


def _arm_done(out_dir: Path) -> bool:
    """Resume: brazo completo sii metrics.json válido Y edgeData no vacío en disco."""
    edgedata = out_dir / OUT_NAME
    return _metrics(out_dir) is not None and edgedata.exists() and edgedata.stat().st_size > 0


def _run_arm(sh: Path, seed: int, out_dir: Path) -> None:
    if _arm_done(out_dir):
        print(f"   [{sh.name}] seed {seed}: reuso (metrics.json + edgeData ya en disco)")
        return
    add_path = build_addfile(CONF, out_dir)          # add-file → out_dir, salida ABS a out_dir
    env = dict(os.environ, OUT_DIR_OVERRIDE=str(out_dir), EDGEDATA_ADD=str(add_path))
    print(f"   [{sh.name}] seed {seed}: corriendo (edgeData zonas PMU) -> {out_dir.name}")
    subprocess.run(["bash", str(sh), str(seed)], check=True, env=env)
    if _metrics(out_dir) is None:
        raise RuntimeError(f"{sh.name} seed {seed} no produjo metrics.json válido")
    if not (out_dir / OUT_NAME).exists():
        raise RuntimeError(f"{sh.name} seed {seed} no produjo edgeData ({OUT_NAME})")


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Orquestador F4 multi-seed con edgeData de zonas PMU.")
    ap.add_argument("--seeds", default="42-51", help="rango 'a-b' o lista 'a,b,c' (default 42-51)")
    args = ap.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    ensure_motor()
    for seed in seeds:
        print(f">> seed {seed}: F4 ambos brazos (con edgeData)")
        _run_arm(FIJO_SH, seed, _fijo_dir(seed))
        _run_arm(ADAP_SH, seed, _adap_dir(seed))
        f = _metrics(_fijo_dir(seed))["net_timeloss_mean_s"]
        a = _metrics(_adap_dir(seed))["net_timeloss_mean_s"]
        rd = (f - a) / f * 100 if f else 0.0
        print(f"   seed {seed} OK: fijo {f:.3f}s  adap {a:.3f}s  RD% {rd:+.2f}  "
              f"(global, sanity vs F5)")
    print(f"\n>> F4 corrida completa para seeds {seeds[0]}-{seeds[-1]}. "
          f"Reproceso: python {HERE.name}/reprocess_f4_zonas.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
