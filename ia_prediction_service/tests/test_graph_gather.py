"""Test de paridad del gather de grafo (Fase 4 STGNN) contra el baseline (Fase 3).

Blinda la invariante de la que depende toda la validez de Fase 5: el gather de grafo
debe compartir byte-a-byte con el baseline el **índice de ventanas**, los **valores** y el
**scaler fijo**. La ÚNICA diferencia permitida es que los 375 nodos van en el eje N en vez
de colapsarse en batch.

Integración: usa el ``.npz`` real de Fase 2, el split compartido (``miraflores_split``) y el
scaler guardado en ``miraflores_baseline_metadata.json``. Si el ``.npz``/metadata no están
presentes (p. ej. clon fresco sin el dataset generado), los casos se saltean.

Corre con el venv de training (tsl no es necesario aquí, pero es el venv del track):
  cd ia_prediction_service && .venv/bin/python -m pytest tests/test_graph_gather.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.miraflores_split import get_split
from src.data.miraflores_windower import (
    DEFAULT_NPZ,
    HORIZON,
    LOOKBACK,
    build_graph_window_index,
    build_window_index,
    compute_timeloss_scaler,
    gather_batch,
    gather_graph_batch,
    load_npz,
)

EVAL_STRIDE = 12             # stride registrado del baseline (metadata)
EXPECTED_SNAPSHOTS = 1160    # 116 starts/día × 10 días de test
EXPECTED_STARTS_PER_DAY = 116
EXPECTED_TEST_DAYS = 10

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_METADATA = (
    REPO_ROOT / "ia_prediction_service" / "scripts" / "miraflores_baseline_metadata.json"
)


@pytest.fixture(scope="module")
def real_data():
    """``(timeloss, mask, seeds)`` del ``.npz`` real de Fase 2; saltea si está ausente."""
    if not Path(DEFAULT_NPZ).exists():
        pytest.skip(f".npz de Fase 2 ausente: {DEFAULT_NPZ}")
    data = load_npz(DEFAULT_NPZ)
    return data["timeloss"], data["mask"], data["seeds"]


def test_paridad_indice_1160_snapshots(real_data):
    """(1) Índice: mismos (day, start_t) y mismo conteo que el baseline → 1.160 snapshots."""
    timeloss, _mask, seeds = real_data
    n_t, n_nodes = timeloss.shape[1], timeloss.shape[2]
    _, _, test_seeds = get_split()

    base = build_window_index(seeds, test_seeds, n_nodes=n_nodes, n_timesteps=n_t,
                              lookback=LOOKBACK, horizon=HORIZON, stride=EVAL_STRIDE)
    graph = build_graph_window_index(seeds, test_seeds, n_timesteps=n_t,
                                     lookback=LOOKBACK, horizon=HORIZON, stride=EVAL_STRIDE)

    # Conteo: 116 starts × 10 días = 1.160 snapshots de grafo; baseline = 1.160 × n_nodes.
    assert graph.shape == (EXPECTED_SNAPSHOTS, 2)
    assert base.shape[0] == EXPECTED_SNAPSHOTS * n_nodes

    last_start = n_t - LOOKBACK - HORIZON
    starts = np.arange(0, last_start + 1, EVAL_STRIDE)
    assert len(starts) == EXPECTED_STARTS_PER_DAY
    assert len(set(int(s) for s in test_seeds)) == EXPECTED_TEST_DAYS

    # El baseline ordena día→start→nodo, repitiendo cada (day,start) n_nodes veces seguidas.
    # Tomar 1 de cada n_nodes filas debe reconstruir EXACTAMENTE el índice de grafo (mismo orden).
    base_daystart = base[::n_nodes][:, :2]
    assert base_daystart.shape == graph.shape
    assert np.array_equal(base_daystart, graph)


def test_paridad_valores_grafo_vs_baseline(real_data):
    """(2) Valores: gather_graph_batch == gather_batch por nodo (byte-igual), eje N vs batch."""
    timeloss, mask, seeds = real_data
    n_t, n_nodes = timeloss.shape[1], timeloss.shape[2]
    _, _, test_seeds = get_split()
    graph = build_graph_window_index(seeds, test_seeds, n_timesteps=n_t,
                                     lookback=LOOKBACK, horizon=HORIZON, stride=EVAL_STRIDE)

    # Subconjunto determinista de snapshots: primero, medio y último.
    sample = graph[[0, len(graph) // 2, len(graph) - 1]]
    Xg, yg, mg = gather_graph_batch(timeloss, mask, sample)
    assert Xg.shape == (3, LOOKBACK, n_nodes, 2)
    assert yg.shape == (3, HORIZON, n_nodes)
    assert mg.shape == (3, HORIZON, n_nodes)

    rng_nodes = np.arange(n_nodes, dtype=np.int32)
    for bi, (d, s) in enumerate(sample):
        # Filas baseline equivalentes: el mismo (day, start_t) para los n_nodes nodos.
        rows = np.stack([np.full(n_nodes, d), np.full(n_nodes, s), rng_nodes],
                        axis=1).astype(np.int32)
        Xb, yb, mb = gather_batch(timeloss, mask, rows)  # [N,L,2], [N,H], [N,H]
        # gather_graph: [L,N,*] → transponer a [N,L,*] para comparar contra el layout por-nodo.
        assert np.array_equal(np.transpose(Xg[bi], (1, 0, 2)), Xb), f"X difiere snapshot {bi}"
        assert np.array_equal(np.transpose(yg[bi], (1, 0)), yb), f"y difiere snapshot {bi}"
        assert np.array_equal(np.transpose(mg[bi], (1, 0)), mb), f"ymask difiere snapshot {bi}"


def test_scaler_fijo_desde_metadata(real_data):
    """(3) Scaler: el aplicado es el train-only FIJO {6.820, 26.581} de metadata, no recomputado."""
    if not BASELINE_METADATA.exists():
        pytest.skip(f"metadata baseline ausente: {BASELINE_METADATA}")
    timeloss, mask, seeds = real_data
    train_seeds, _, _ = get_split()
    meta_scaler = json.loads(BASELINE_METADATA.read_text())["scaler"]

    # El scaler que aplica el gather de grafo es el train-only compartido con el baseline
    # (determinista sobre el mismo TRAIN) → idéntico al guardado en metadata, NO recomputado
    # sobre el split de test ni sobre los batches del STGNN.
    scaler = compute_timeloss_scaler(timeloss, mask, seeds, train_seeds)
    assert abs(scaler["mean"] - meta_scaler["mean"]) < 1e-4
    assert abs(scaler["std"] - meta_scaler["std"]) < 1e-4
    # Valores fijos documentados (universo v2, N=1660; B4 retrain).
    assert round(scaler["mean"], 3) == 6.82
    assert round(scaler["std"], 3) == 26.581
