"""Genera el add-file de <edgeData> de F4 sobre los 9 edges-aproche de los 3 cruces PMU
bifásicos saturados (fuente: conf/miraflores_red_completa/zonas_pmu_f4.json).

El add-file emite el SET COMPLETO de atributos por edge (timeLoss, waitingTime, density,
speed, occupancy, entered/left, ...) — SUMO 1.26 los da todos al no restringir `attributes`.
`freq=3600` (por hora): el reproceso agrega las horas a las franjas B2 (más flexible que
emitir por franja). `excludeEmpty=false`: un edge sin tráfico en una hora igual aparece.

La salida de edgeData va a un path ABSOLUTO dentro de out_dir (no depende del CWD del proceso
SUMO). El add-file se escribe en out_dir/edgedata_zonas_pmu.add.xml; ambos brazos (fijo y
adaptativo) cargan el MISMO add-file (mismas zonas, misma freq) para comparar apples-to-apples.

Uso:
    python make_edgedata_zonas.py --out <dir> [--conf <zonas_pmu_f4.json>]
    # imprime el path del add-file generado (lo consumen los runners vía -a)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
DEFAULT_CONF = SIM_ROOT / "conf" / "miraflores_red_completa" / "zonas_pmu_f4.json"

ADD_NAME = "edgedata_zonas_pmu.add.xml"
OUT_NAME = "edgedata_zonas_pmu.xml"
EDGEDATA_ID = "zonas_pmu"
FREQ_S = 3600


def build_addfile(conf_path: Path, out_dir: Path) -> Path:
    conf = json.loads(conf_path.read_text())
    edges: list[str] = []
    for z in conf["zonas"]:
        edges.extend(z["edges_aproche"])
    # dedupe preservando orden (no debería haber solapamiento entre cruces, pero por las dudas)
    seen: set[str] = set()
    edges = [e for e in edges if not (e in seen or seen.add(e))]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (out_dir / OUT_NAME).resolve()      # ABSOLUTO → independiente del CWD de SUMO
    add_path = out_dir / ADD_NAME

    edges_attr = " ".join(edges)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!-- F4: edgeData sobre los 9 aproches de los 3 cruces PMU bifásicos saturados.\n'
        f'     Fuente: {conf_path.name}. Generado por make_edgedata_zonas.py (no editar a mano). -->\n'
        '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n'
        f'    <edgeData id="{EDGEDATA_ID}" freq="{FREQ_S}" file="{out_file}"\n'
        f'             excludeEmpty="false" edges="{edges_attr}"/>\n'
        '</additional>\n'
    )
    add_path.write_text(xml)
    return add_path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Genera el add-file de edgeData de F4 (zonas PMU).")
    ap.add_argument("--out", required=True, type=Path, help="dir de la corrida (runs/...)")
    ap.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    args = ap.parse_args(argv)
    add_path = build_addfile(args.conf, args.out)
    print(str(add_path))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
