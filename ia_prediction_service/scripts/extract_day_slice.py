"""Fase 3 — pre-extracción offline del slice del día seed051 para servir en el core.

El core sirve la predicción GRU baseline cortando la ventana del lunes 8 jun = **seed051**.
La ventana fiel al training sale del ``.npz`` consolidado de Fase 2
(``miraflores_laborable_60d.npz`` ``[60,1440,1660,1]`` + máscara), NO del Parquet (que
exigiría reimplementar el builder con riesgo de orden de nodos). Pero el ``.npz`` completo
pesa 574 MB descomprimido — excede el límite de 512 MB del core y está gitignored, así que
no se hornea a la imagen.

Este script extrae UN día (``day_idx=9`` ↔ seed051) a un artefacto chico, auto-contenido
(~1.8 MB comprimido, ~12 MB en RAM), que SÍ se versiona y se copia a la imagen del core
(``core_management_api/models/day_seed051_tensor.npz``). El core lo carga al startup y corta
la ventana ``t-29..t`` en memoria.

PROCEDENCIA — ``day_idx=9 ↔ seed051`` (verificado, no asumido):
- ``metadata.json`` del tensor: ``seed_by_day_index["9"] == 51``.
- Regla del builder: día N ↔ seed 42+N ⇒ seed 51 = día 9.
- Este script RE-VERIFICA en runtime que ``seeds[9] == 51`` antes de extraer (fail-fast).

Auto-contenido: guarda ``node_order`` (los 1660 ``sumo_edge`` en orden ``node_index``) DENTRO
del slice, para que el core mapee ``node_index → edge_id`` sin depender del LCC mapping JSON
(que no está en la imagen del core). El stage-gate verifica que ese ``node_order`` es
byte-idéntico al del ``.npz`` de training (cierra el círculo entrada→salida del mismo orden).

STAGE-GATE 1 (incluido acá, fail-fast): el slice extraído es byte-idéntico al slice del
``.npz`` completo — ``np.array_equal`` de tensor, máscara y ``node_order``. Si no coincide,
aborta con código != 0 y NO escribe (o invalida) el artefacto.

Uso (desde la raíz del repo, con el root .venv que tiene numpy/pyarrow):
  .venv/bin/python ia_prediction_service/scripts/extract_day_slice.py
  .venv/bin/python ia_prediction_service/scripts/extract_day_slice.py --check-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# scripts → ia_prediction_service → CerebroVial
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC_NPZ = (
    REPO_ROOT
    / "simulation"
    / "data"
    / "datasets"
    / "miraflores_laborable_60d"
    / "tensors"
    / "miraflores_laborable_60d.npz"
)
DEFAULT_OUT_NPZ = REPO_ROOT / "core_management_api" / "models" / "day_seed051_tensor.npz"

DAY_IDX = 9      # ↔ seed051 (ver PROCEDENCIA en el docstring)
SEED = 51


def extract_slice(src_npz: Path) -> dict:
    """Extrae el slice ``day_idx=9`` del ``.npz`` completo. Fail-fast si seed no coincide."""
    z = np.load(src_npz, allow_pickle=True)
    seeds = z["seeds"].astype(int)
    if int(seeds[DAY_IDX]) != SEED:
        raise ValueError(
            f"PROCEDENCIA ROTA: seeds[{DAY_IDX}] == {int(seeds[DAY_IDX])}, esperado {SEED}. "
            "El mapeo day_idx↔seed cambió; abortar antes de extraer un día equivocado."
        )
    tensor = np.ascontiguousarray(z["tensor"][DAY_IDX])  # [1440, 1660, 1]
    mask = np.ascontiguousarray(z["mask"][DAY_IDX])       # [1440, 1660]
    node_order = z["node_order"]                          # [1660] object (sumo_edge)
    channels = z["channels"]                              # ['timeLoss']
    timesteps = z["timesteps"].astype(np.int32)           # [1440] begin (s)
    return {
        "tensor": tensor,
        "mask": mask,
        "node_order": node_order,
        "channels": channels,
        "timesteps": timesteps,
        "seed": np.int32(SEED),
        "day_idx": np.int32(DAY_IDX),
    }


def gate1(slice_data: dict, src_npz: Path) -> None:
    """STAGE-GATE 1: el slice es byte-idéntico al slice del .npz completo. Aborta si no."""
    z = np.load(src_npz, allow_pickle=True)
    # equal_nan=True: las celdas vacías son NaN (~78.5%) y NaN != NaN; sin esto la
    # igualdad byte-a-byte daría falso negativo. Comparamos valores tratando NaN==NaN.
    checks = {
        "tensor": np.array_equal(slice_data["tensor"], z["tensor"][DAY_IDX], equal_nan=True),
        "mask": np.array_equal(slice_data["mask"], z["mask"][DAY_IDX]),
        "node_order": list(slice_data["node_order"]) == list(z["node_order"]),
    }
    print("STAGE-GATE 1 — slice byte-idéntico al .npz completo (day_idx=9 ↔ seed051):")
    for k, ok in checks.items():
        print(f"  {k:11s}: {'OK' if ok else 'FALLA'}")
    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        raise SystemExit(f"STAGE-GATE 1 FALLA en {failed} — abortado, no se escribe el artefacto.")
    print("  → byte-idéntico. node_order del slice == node_order de training (círculo cerrado).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(DEFAULT_SRC_NPZ), help="ruta al .npz completo")
    parser.add_argument("--out", default=str(DEFAULT_OUT_NPZ), help="ruta de salida del slice")
    parser.add_argument(
        "--check-only", action="store_true",
        help="extrae en memoria y corre stage-gate 1 sin escribir el artefacto",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(
            f"No existe el .npz completo: {src}\n"
            "Es gitignored/regenerable. Regenéralo con miraflores_dataset_builder antes de extraer."
        )

    slice_data = extract_slice(src)
    print(
        f"Slice extraído: tensor={tuple(slice_data['tensor'].shape)} "
        f"mask={tuple(slice_data['mask'].shape)} nodos={len(slice_data['node_order'])} "
        f"seed={int(slice_data['seed'])} day_idx={int(slice_data['day_idx'])}"
    )
    gate1(slice_data, src)

    if args.check_only:
        print("--check-only: stage-gate 1 verde, no se escribió el artefacto.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tensor=slice_data["tensor"],
        mask=slice_data["mask"],
        node_order=slice_data["node_order"],
        channels=slice_data["channels"],
        timesteps=slice_data["timesteps"],
        seed=slice_data["seed"],
        day_idx=slice_data["day_idx"],
    )
    size_mb = out.stat().st_size / 1e6
    print(f"Escrito: {out} ({size_mb:.2f} MB)")

    # Re-verificación: lo escrito al disco también pasa el gate (no solo lo que está en RAM).
    written = np.load(out, allow_pickle=True)
    reread_ok = (
        np.array_equal(written["tensor"], slice_data["tensor"], equal_nan=True)
        and np.array_equal(written["mask"], slice_data["mask"])
        and list(written["node_order"]) == list(slice_data["node_order"])
    )
    if not reread_ok:
        raise SystemExit("El artefacto escrito NO coincide con el slice en RAM — abortado.")
    print("Re-lectura del artefacto: byte-idéntica al slice en RAM. OK.")


if __name__ == "__main__":
    main()
