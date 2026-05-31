"""Genera el dataset de entrenamiento de TTH-09 (CT-07.3).

Por cada (pattern, seed) en train+valid:
- Invoca sumo headless con seed determinístico.
- Lee edgedata.xml (velocidad/cuenta por edge por bucket 60s, agregado a
  dirección) y lanearea.xml (maxJamLengthInMeters por detector por
  bucket, suma sobre lanes de la dirección).
- Construye filas (seed, pattern, t_sim_s, direction, ...) según schema.
- Persiste a data/{partition}/{pattern}_seed{N}.parquet.

Sin TraCI live (corrección 1). Lee XML fallback de F1 (SUMO 1.26 arrow
writer no soporta append multi-interval) y escribe Parquet final
(single-write OK).

Uso:
    python -m cerebrovial_simulation.dataset.generate         # full (80 train + 20 valid)
    python -m cerebrovial_simulation.dataset.generate --quick # 1 seed × 4 patrones × 2 particiones = 8 corridas
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ..jam_level import ratio_to_jam_level
from . import partitions, schema


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent.parent
SCENARIOS_DIR = SIM_ROOT / "conf" / "scenarios"
NETWORK_DIR = SIM_ROOT / "conf" / "network"
DATA_DIR = SIM_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"


EDGE_TO_DIR = {"N_in": "N", "S_in": "S", "E_in": "E", "W_in": "W"}
DET_BY_DIR = {
    "N": ["LA_N_0", "LA_N_1", "LA_N_2"],
    "S": ["LA_S_0", "LA_S_1", "LA_S_2"],
    "E": ["LA_E_0", "LA_E_1"],
    "W": ["LA_W_0", "LA_W_1"],
}
DET_DIR = {det: d for d, dets in DET_BY_DIR.items() for det in dets}


def _run_sumo(pattern: str, seed: int, out_dir: Path) -> tuple[Path, Path]:
    """Corrida headless en out_dir con seed determinístico.

    Retorna paths a (edgedata.xml, lanearea.xml).
    """
    sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
    cfg = SCENARIOS_DIR / f"{pattern}.sumocfg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copiar add-files a out_dir (SUMO resuelve file= relativo al add-file).
    add_files_local = []
    for add_name in ["miraflores_4way.tllogic.add.xml", "edgedata.add.xml", "lanearea.add.xml"]:
        dst = out_dir / add_name
        shutil.copy(NETWORK_DIR / add_name, dst)
        add_files_local.append(dst)

    args = [
        str(sumo_bin),
        "-c", str(cfg),
        "--additional-files", ",".join(str(p) for p in add_files_local),
        "--seed", str(seed),
        # Outputs single-write OK en Parquet directo (NO los persistimos al
        # dataset final pero los necesitamos para F5 KPIs si se reusan).
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
    ]
    proc = subprocess.run(args, cwd=out_dir, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sumo failed para {pattern} seed={seed}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return out_dir / "edgedata.xml", out_dir / "lanearea.xml"


def _parse_edgedata_buckets(path: Path) -> dict[tuple[float, str], dict[str, float]]:
    """Retorna {(bucket_t, direction): {'speed': X, 'departed': Y, 'sampled': Z}}.

    `departed` cuenta vehículos spawneados en el bucket (apto para approach
    edges alimentados por <flow>). `entered` es 0 para source edges porque
    nadie viene "de otra parte" — la N_in/etc. son nodos priority dead-end
    en este net.

    `sampled` (sampledSeconds) es la señal de presencia: con excludeEmpty="false"
    un edge vacío se emite con sampledSeconds="0.00" y SIN atributo speed (2b).
    Por eso `speed` usa centinela -1 (atributo ausente) y NUNCA se lee cuando
    sampled<=0 — ver _build_rows_for_run.
    """
    root = ET.parse(path).getroot()
    result: dict[tuple[float, str], dict[str, float]] = {}
    for interval in root.findall("interval"):
        bucket_t = float(interval.attrib["end"])
        for edge in interval.findall("edge"):
            eid = edge.attrib["id"]
            d = EDGE_TO_DIR.get(eid)
            if d is None:
                continue
            speed = float(edge.attrib.get("speed", "-1"))  # -1 = atributo ausente
            departed = float(edge.attrib.get("departed", "0"))
            sampled = float(edge.attrib.get("sampledSeconds", "0"))
            result[(bucket_t, d)] = {"speed": speed, "departed": departed, "sampled": sampled}
    return result


def _parse_lanearea_buckets(path: Path) -> dict[tuple[float, str], float]:
    """Retorna {(bucket_t, direction): sum_maxJamLengthInMeters} agregado por dirección."""
    root = ET.parse(path).getroot()
    result: dict[tuple[float, str], float] = {}
    for interval in root.findall("interval"):
        det_id = interval.attrib["id"]
        d = DET_DIR.get(det_id)
        if d is None:
            continue
        bucket_t = float(interval.attrib["end"])
        max_jam = float(interval.attrib.get("maxJamLengthInMeters", "0"))
        key = (bucket_t, d)
        result[key] = result.get(key, 0.0) + max_jam
    return result


def _build_rows_for_run(
    pattern: str,
    seed: int,
    edgedata_xml: Path,
    lanearea_xml: Path,
) -> list[dict]:
    edge_data = _parse_edgedata_buckets(edgedata_xml)
    lane_data = _parse_lanearea_buckets(lanearea_xml)

    # Buckets observados = unión de claves; ordenados.
    buckets = sorted({t for (t, _) in edge_data.keys()})

    rows = []
    for t in buckets:
        for d in schema.DIRECTIONS:
            e = edge_data.get((t, d))
            q = lane_data.get((t, d), 0.0)
            if e is None or e["sampled"] <= 0.0:
                # Sin observación: calle vacía = flujo libre (jam 0), NO vía cerrada.
                # SUMO emite el bucket vacío con sampledSeconds="0.00" y sin atributo
                # speed (medido en 2b); usar sampledSeconds como señal de presencia
                # evita mapear ese hueco a jam 5.
                # SUPUESTO: vacío = sin demanda. Válido en la 4-way sintética (demanda
                # controlada). En red real multi-intersección (TTH-09) un edge puede
                # estar vacío por bloqueo aguas arriba → revisar antes de heredarlo.
                speed, ratio, n_veh, jam = schema.VMAX_MPS, 1.0, 0, 0
            else:
                speed = e["speed"]
                n_veh = int(e["departed"])
                ratio = speed / schema.VMAX_MPS if schema.VMAX_MPS > 0 else 0.0
                # speed=0 con vehículos presentes → ratio 0 → jam 5 (atasco real).
                jam = int(ratio_to_jam_level(ratio))
            rows.append({
                "seed": seed,
                "pattern": pattern,
                "t_sim_s": float(t),
                "direction": d,
                "mean_speed_mps": float(speed),
                "n_vehicles": int(n_veh),
                "queue_length_m": float(q),
                "max_speed_mps": schema.VMAX_MPS,
                "ratio": float(ratio),
                "jam_level": int(jam),
            })
    return rows


def generate_one(pattern: str, seed: int, partition_dir: Path, scratch_root: Path) -> Path:
    """Genera 1 archivo Parquet: data/{partition}/{pattern}_seed{N}.parquet."""
    out_dir = scratch_root / f"{pattern}_seed{seed}"
    edgedata, lanearea = _run_sumo(pattern, seed, out_dir)
    rows = _build_rows_for_run(pattern, seed, edgedata, lanearea)

    # pyarrow Table desde rows.
    cols = {name: [r[name] for r in rows] for name in schema.SCHEMA.names}
    table = pa.Table.from_pydict(cols, schema=schema.SCHEMA)
    schema.validate(table)

    partition_dir.mkdir(parents=True, exist_ok=True)
    out_pq = partition_dir / f"{pattern}_seed{seed}.parquet"
    pq.write_table(table, out_pq)
    return out_pq


def generate_all(quick: bool = False) -> dict[str, list[Path]]:
    """Genera train + valid completos. quick=True ⇒ 1 seed por partición × patrón."""
    train_iter: Iterator[tuple[str, int]]
    valid_iter: Iterator[tuple[str, int]]
    if quick:
        train_iter = iter([(p, partitions.TRAIN_SEEDS[0]) for p in partitions.PATTERNS])
        valid_iter = iter([(p, partitions.VALID_SEEDS[0]) for p in partitions.PATTERNS])
    else:
        train_iter = partitions.iter_train()
        valid_iter = partitions.iter_valid()

    produced: dict[str, list[Path]] = {"train": [], "valid": []}
    with tempfile.TemporaryDirectory(prefix="cerebrovial_dataset_") as scratch:
        scratch_root = Path(scratch)
        for pattern, seed in train_iter:
            print(f"  train: {pattern} seed={seed}")
            produced["train"].append(generate_one(pattern, seed, TRAIN_DIR, scratch_root))
        for pattern, seed in valid_iter:
            print(f"  valid: {pattern} seed={seed}")
            produced["valid"].append(generate_one(pattern, seed, VALID_DIR, scratch_root))
    return produced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="1 seed × patrón en cada partición (smoke).")
    args = parser.parse_args()
    produced = generate_all(quick=args.quick)
    print(f"\nTrain: {len(produced['train'])} archivos en {TRAIN_DIR}")
    print(f"Valid: {len(produced['valid'])} archivos en {VALID_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
