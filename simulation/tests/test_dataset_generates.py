"""F3 — Dataset Parquet + particiones train/valid sin overlap.

Tests:
- c1: partitions.iter_train() y iter_valid() no comparten ningún (pattern, seed).
- c2: corrida 1 seed × 1 patrón produce Parquet legible por pyarrow con
  todas las columnas del schema.
- c3: ranges — jam_level ∈ [0,5]; ratio ∈ [0,1.05]; queue_length_m ≥ 0;
  n_vehicles ≥ 0.
- c4: am_peak seed=1 cumple invariantes (4 dirs × 20 buckets = 80 rows;
  n_vehicles consistente con flujos del patrón).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from cerebrovial_simulation.dataset import generate as gen_mod
from cerebrovial_simulation.dataset import partitions, schema


def test_c1_train_valid_no_overlap():
    """No hay (pattern, seed) compartido entre train y valid."""
    train_set = set(partitions.iter_train())
    valid_set = set(partitions.iter_valid())
    overlap = train_set & valid_set
    assert overlap == set(), f"Overlap detectado: {overlap}"
    assert len(train_set) == 80, f"Train debería tener 80 corridas, tiene {len(train_set)}"
    assert len(valid_set) == 20, f"Valid debería tener 20 corridas, tiene {len(valid_set)}"


def test_c2_generate_one_produces_valid_parquet(tmp_path):
    """1 seed × 1 patrón produce Parquet legible con schema completo."""
    partition_dir = tmp_path / "train"
    with tempfile.TemporaryDirectory() as scratch:
        out = gen_mod.generate_one(
            pattern="am_peak",
            seed=1,
            partition_dir=partition_dir,
            scratch_root=Path(scratch),
        )
    assert out.exists()
    table = pq.read_table(str(out))
    # Schema válido.
    schema.validate(table)
    # Tamaño esperado: 1200s / 60s = 20 buckets × 4 dirs = 80 filas.
    assert table.num_rows == 80, f"num_rows={table.num_rows}, esperado 80"


def test_c3_ranges_within_bounds(tmp_path):
    """Todas las columnas numéricas en rangos válidos."""
    partition_dir = tmp_path / "train"
    with tempfile.TemporaryDirectory() as scratch:
        out = gen_mod.generate_one(
            pattern="am_peak", seed=1,
            partition_dir=partition_dir, scratch_root=Path(scratch),
        )
    table = pq.read_table(str(out))
    cols = table.to_pydict()

    assert all(0 <= j <= 5 for j in cols["jam_level"]), "jam_level fuera de [0,5]"
    assert all(0 <= r <= 1.05 for r in cols["ratio"]), "ratio fuera de [0, 1.05]"
    assert all(q >= 0 for q in cols["queue_length_m"]), "queue_length_m negativo"
    assert all(n >= 0 for n in cols["n_vehicles"]), "n_vehicles negativo"
    assert all(s >= 0 for s in cols["mean_speed_mps"]), "mean_speed negativo"
    # max_speed_mps constante.
    assert all(abs(m - schema.VMAX_MPS) < 0.01 for m in cols["max_speed_mps"])


def test_c4_am_peak_seed1_invariants(tmp_path):
    """am_peak seed=1: 80 rows, n_vehicles N+S ~ 40/bucket (2400 v/h × 60/3600)."""
    partition_dir = tmp_path / "train"
    with tempfile.TemporaryDirectory() as scratch:
        out = gen_mod.generate_one(
            pattern="am_peak", seed=1,
            partition_dir=partition_dir, scratch_root=Path(scratch),
        )
    table = pq.read_table(str(out)).to_pydict()
    # 4 direcciones × 20 buckets = 80.
    assert len(table["seed"]) == 80
    # n_vehicles para N + S por bucket inicial: ~40 cada uno (2400 v/h × 1/60 h = 40).
    # Filtra rows direction=='N' y t_sim_s == 60.
    n_first_bucket = [
        n for n, d, t in zip(table["n_vehicles"], table["direction"], table["t_sim_s"])
        if d == "N" and t == 60.0
    ]
    assert n_first_bucket, "no encontró fila N t=60"
    assert 30 <= n_first_bucket[0] <= 50, (
        f"n_vehicles[N, t=60]={n_first_bucket[0]} fuera de [30,50] (esperado ~40 para 2400 v/h)"
    )


def test_c5_generate_all_quick(tmp_path, monkeypatch):
    """generate_all(quick=True) produce 4 train + 4 valid Parquet legibles."""
    # Redirigir DATA_DIR/TRAIN_DIR/VALID_DIR del módulo al tmp_path.
    monkeypatch.setattr(gen_mod, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(gen_mod, "VALID_DIR", tmp_path / "valid")

    produced = gen_mod.generate_all(quick=True)
    assert len(produced["train"]) == 4
    assert len(produced["valid"]) == 4
    for path in produced["train"] + produced["valid"]:
        assert path.exists()
        t = pq.read_table(str(path))
        schema.validate(t)
        assert t.num_rows > 0
