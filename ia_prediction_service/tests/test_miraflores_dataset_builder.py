"""Tests del dataset builder por-arista de Miraflores (miraflores_dataset_builder.py).

Verifica: shape consolidado + máscara; alineación por ``node_index`` (NO por orden
de aparición); regla de máscara de 3 estados (vacío → False + target NaN); que el
target es el ``timeLoss`` TOTAL del Parquet (NO un promedio derivado); fail-fast por
arista/seed faltante; determinismo; y exclusión de las aristas de la islita.

Fixtures chicas en ``tmp_path``. Dos tests adicionales corren contra el dataset real
(60 Parquet en disco) para el shape exacto [60,1440,375,1] y ~82% de máscara vacía;
se saltan si el dataset no está presente.

Corre con el venv raíz:
``cd ia_prediction_service && .venv/bin/python -m pytest tests/test_miraflores_dataset_builder.py``
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.miraflores_dataset_builder import (
    DEFAULT_LCC_MAPPING,
    DEFAULT_PARQUET_DIR,
    DEFAULT_SEEDS,
    build_miraflores_dataset,
    load_node_order,
    save_dataset,
)

# --- Fixture sintética: 3 aristas LCC (A,B,C en orden node_index) + 1 islita -------
LCC_EDGES = ["A_edge", "B_edge", "C_edge"]   # node_index 0, 1, 2
ISLAND_EDGES = ["ISL_x", "ISL_y"]
TIMESTEPS = [0, 60]
NAN = float("nan")

# (edge, timestep): (timeLoss, density, speed, flow)
# A_edge t0: tráfico con n_veh=3 (flow=180) y timeLoss TOTAL=30 → NO debe verse 10 (mean).
# A_edge t60: VACÍO (density=0, speed NaN).
# C_edge t0: VACÍO.
_CELLS_S42 = {
    ("A_edge", 0): (30.0, 2.0, 8.0, 180.0),
    ("A_edge", 60): (0.0, 0.0, NAN, 0.0),
    ("B_edge", 0): (5.0, 1.0, 9.0, 60.0),
    ("B_edge", 60): (7.0, 1.5, 9.5, 120.0),
    ("C_edge", 0): (0.0, 0.0, NAN, 0.0),
    ("C_edge", 60): (2.0, 0.5, 10.0, 60.0),
}


def _write_mapping(tmp_path, edges=LCC_EDGES, island=ISLAND_EDGES, scramble=True):
    """LCC mapping con ``nodes`` (node_index 0..N-1) — desordenado a propósito para
    probar que el orden sale de node_index, no del orden del archivo."""
    nodes = [{"node_index": i, "sumo_edge": e} for i, e in enumerate(edges)]
    if scramble:
        nodes = list(reversed(nodes))
    p = tmp_path / "lcc_mapping.json"
    p.write_text(json.dumps({"nodes": nodes, "dropped_edges": list(island)}))
    return p


def _write_parquet(parquet_dir, seed, cells, *, include_island=True, drop_edge=None):
    """Escribe day_seed0NN.parquet. Filas en orden ARBITRARIO (no node_index)."""
    rows = []
    for (edge, ts), (tl, dens, spd, flow) in cells.items():
        if edge == drop_edge:
            continue
        rows.append({"edge_id": edge, "timestep": ts, "timeLoss": tl,
                     "density": dens, "speed": spd, "flow": flow})
    if include_island:
        for isl in ISLAND_EDGES:
            for ts in TIMESTEPS:
                rows.append({"edge_id": isl, "timestep": ts, "timeLoss": 99.0,
                             "density": 9.0, "speed": 1.0, "flow": 600.0})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    df["edge_id"] = df["edge_id"].astype("string")
    df["timestep"] = df["timestep"].astype("int32")
    for c in ("timeLoss", "density", "speed", "flow"):
        df[c] = df[c].astype("float32")
    df.to_parquet(parquet_dir / f"day_seed{seed:03d}.parquet", index=False)


def _build_small(tmp_path, seeds=(42, 43)):
    mp = _write_mapping(tmp_path)
    pdir = tmp_path / "ds"
    pdir.mkdir()
    for s in seeds:
        cells = {k: v for k, v in _CELLS_S42.items()}
        if s != 42:  # variar valores por día (no afecta estructura)
            cells = {k: (tl + s, d, sp, fl) for k, (tl, d, sp, fl) in cells.items()}
        _write_parquet(pdir, s, cells)
    return mp, pdir


# ---------------------------------------------------------------- shape + máscara --
def test_shape_general(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42, 43))
    r = build_miraflores_dataset(pdir, mp, seeds=(42, 43))
    assert r["tensor"].shape == (2, len(TIMESTEPS), len(LCC_EDGES), 1)
    assert r["mask"].shape == (2, len(TIMESTEPS), len(LCC_EDGES))
    assert r["channels"] == ["timeLoss"]
    assert list(r["seeds"]) == [42, 43]
    assert r["node_order"] == LCC_EDGES


# -------------------------------------------------- alineación por node_index ------
def test_alignment_by_node_index_not_order(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42,))
    r = build_miraflores_dataset(pdir, mp, seeds=(42,))
    node = {e: i for i, e in enumerate(r["node_order"])}
    ts = {t: i for i, t in enumerate(r["timesteps"])}
    # A_edge es node_index 0 pese a estar al final del archivo y barajado en el Parquet.
    assert node["A_edge"] == 0 and node["B_edge"] == 1 and node["C_edge"] == 2
    # tensor[día, t, node_index, 0] == timeLoss crudo del Parquet para ese sumo_edge+ts.
    for (edge, t), (tl, dens, spd, _flow) in _CELLS_S42.items():
        val = r["tensor"][0, ts[t], node[edge], 0]
        empty = (dens == 0.0) and np.isnan(spd)
        if empty:
            assert np.isnan(val)
        else:
            assert val == pytest.approx(tl)


# ------------------------------------------------------- máscara 3 estados ---------
def test_mask_three_state(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42,))
    r = build_miraflores_dataset(pdir, mp, seeds=(42,))
    node = {e: i for i, e in enumerate(r["node_order"])}
    ts = {t: i for i, t in enumerate(r["timesteps"])}
    # Celda vacía conocida (A_edge t60): máscara False + target NaN.
    assert r["mask"][0, ts[60], node["A_edge"]] == False  # noqa: E712
    assert np.isnan(r["tensor"][0, ts[60], node["A_edge"], 0])
    # Celda con tráfico (B_edge t0): máscara True + target finito.
    assert r["mask"][0, ts[0], node["B_edge"]] == True  # noqa: E712
    assert np.isfinite(r["tensor"][0, ts[0], node["B_edge"], 0])
    # En la fixture, 2 de 6 celdas son vacías.
    assert r["mask"].sum() == 4


# --------------------------------------- target = TOTAL (no promedio derivado) -----
def test_target_is_total_not_mean(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42,))
    r = build_miraflores_dataset(pdir, mp, seeds=(42,))
    node = {e: i for i, e in enumerate(r["node_order"])}
    ts = {t: i for i, t in enumerate(r["timesteps"])}
    # A_edge t0: flow=180 → n_veh≈3, timeLoss TOTAL=30. El builder NO debe dividir.
    val = r["tensor"][0, ts[0], node["A_edge"], 0]
    assert val == pytest.approx(30.0)          # total, tal cual el Parquet
    assert val != pytest.approx(10.0)          # NO el mean (30/3)


# ------------------------------------------------------------- fail-fast -----------
def test_failfast_missing_edge(tmp_path):
    mp = _write_mapping(tmp_path)
    pdir = tmp_path / "ds"
    pdir.mkdir()
    _write_parquet(pdir, 42, _CELLS_S42, drop_edge="C_edge")
    with pytest.raises(ValueError, match="C_edge"):
        build_miraflores_dataset(pdir, mp, seeds=(42,))


def test_failfast_missing_seed(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42, 43))
    with pytest.raises(ValueError, match="seed 99"):
        build_miraflores_dataset(pdir, mp, seeds=(42, 43, 99))


def test_failfast_missing_feature_column(tmp_path):
    # Pedir una feature inexistente → ValueError claro (no traceback crudo de pyarrow),
    # validado contra el schema antes del read.
    mp, pdir = _build_small(tmp_path, seeds=(42,))
    with pytest.raises(ValueError, match="columna_inexistente"):
        build_miraflores_dataset(pdir, mp, seeds=(42,), feature_columns=["columna_inexistente"])


# ------------------------------------------------------------- determinismo --------
def test_determinism(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42, 43))
    a = build_miraflores_dataset(pdir, mp, seeds=(42, 43))
    b = build_miraflores_dataset(pdir, mp, seeds=(42, 43))
    np.testing.assert_array_equal(a["tensor"], b["tensor"])  # equal_nan por defecto
    np.testing.assert_array_equal(a["mask"], b["mask"])


# --------------------------------------------------------- islita excluida ---------
def test_island_edges_excluded(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42,))
    r = build_miraflores_dataset(pdir, mp, seeds=(42,))
    for isl in ISLAND_EDGES:
        assert isl not in r["node_order"]
    assert len(r["node_order"]) == len(LCC_EDGES)


# --------------------------------------------------------- persistencia ------------
def test_save_dataset_writes_npz_and_metadata(tmp_path):
    mp, pdir = _build_small(tmp_path, seeds=(42, 43))
    r = build_miraflores_dataset(pdir, mp, seeds=(42, 43))
    out = tmp_path / "tensors"
    written = save_dataset(r, out, lcc_mapping_path=mp)
    assert written["npz"].exists() and written["metadata"].exists() and written["readme"].exists()
    loaded = np.load(written["npz"], allow_pickle=True)
    np.testing.assert_array_equal(loaded["tensor"], r["tensor"])
    np.testing.assert_array_equal(loaded["mask"], r["mask"])
    assert list(loaded["seeds"]) == [42, 43]
    meta = json.loads(written["metadata"].read_text())
    assert meta["channels"] == ["timeLoss"]
    assert "D-013" in meta["target_authority"]
    assert meta["shape"]["tensor"] == [2, len(TIMESTEPS), len(LCC_EDGES), 1]


# =========================== TESTS CONTRA EL DATASET REAL (skip si ausente) ========
def _real_dataset_present() -> bool:
    return all(
        (DEFAULT_PARQUET_DIR / f"day_seed{s:03d}.parquet").exists() for s in DEFAULT_SEEDS
    ) and DEFAULT_LCC_MAPPING.exists()


real_only = pytest.mark.skipif(
    not _real_dataset_present(), reason="dataset real (60 Parquet) no presente en disco"
)


@real_only
def test_real_shape_and_island_exclusion():
    r = build_miraflores_dataset()
    assert r["tensor"].shape == (60, 1440, 375, 1)
    assert r["mask"].shape == (60, 1440, 375)
    # las 6 aristas de la islita (dropped_edges del LCC) no aparecen.
    dropped = json.loads(DEFAULT_LCC_MAPPING.read_text())["dropped_edges"]
    assert dropped, "el mapping debería declarar dropped_edges"
    assert set(dropped).isdisjoint(set(r["node_order"]))


@real_only
def test_real_mask_fraction_about_82pct():
    r = build_miraflores_dataset()
    empty_frac = 1.0 - float(r["mask"].mean())
    assert 0.78 <= empty_frac <= 0.86, f"fracción vacía={empty_frac:.3f}, esperaba ≈0.82"
    # celdas vacías → NaN; celdas válidas → finitas (timeLoss total, sin imputar).
    assert np.isnan(r["tensor"][~r["mask"]]).all()
    assert np.isfinite(r["tensor"][r["mask"]]).all()


def test_load_node_order_real_lcc():
    if not DEFAULT_LCC_MAPPING.exists():
        pytest.skip("LCC mapping ausente")
    order = load_node_order(DEFAULT_LCC_MAPPING)
    assert len(order) == 375
    assert len(set(order)) == 375  # sin duplicados
