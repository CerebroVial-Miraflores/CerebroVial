"""Gate de Fase 1 — el loader de inferencia reproduce el eval offline existente.

La condición de cierre de Fase 1 es honesta: el loader (``src/inference``) debe producir
las MISMAS predicciones que el ``predict_collect`` de los scripts de training, sobre la
misma entrada, dentro de tolerancia numérica estricta. Si no reproduce, hay bug (probable
orden de nodos o scaler) y el test FALLA crudo —no se afloja la tolerancia.

Blindajes (acordados con Cesar en plan mode):
- **Orden de nodos del oráculo GRU**: el ``GruAdapter`` recorre las 1660 aristas en orden
  ``node_index`` (transpone el window); el oráculo ``predict_collect`` recorre el orden del
  índice que se le pasa. Se construye el índice GRU EXPLÍCITAMENTE (snapshot-major,
  node_index-minor) y, ANTES del ``allclose``, se asierta que las entradas de ambos caminos
  son idénticas (``gather_batch`` == transpuesta del window de grafo). Si ese assert falla,
  el test revienta ahí, localizando el desalineo en vez de esconderlo en el allclose.
- **CPU en ambos lados**: el loader corre en CPU; el oráculo TAMBIÉN (se fuerza
  ``DEV='cpu'`` en el script GRU —usa un global de módulo— y se pasa ``dev='cpu'`` al STGNN).
  Así la tolerancia 1e-5 es honesta y no se persigue un falso negativo CPU-vs-MPS.

Correr: ``cd ia_prediction_service && .venv/bin/python -m pytest tests/test_inference_loader.py -v``
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from src.data.miraflores_split import get_split
from src.data.miraflores_windower import (
    DEFAULT_NPZ,
    build_graph_window_index,
    compute_timeloss_scaler,
    gather_batch,
    gather_graph_batch,
    load_lcc_edge_index,
    load_npz,
)
from src.inference import (
    GruAdapter,
    StgnnAdapter,
    load_inference_adapter,
    resolve,
)
from src.models.miraflores_gru_baseline import MirafloresGRUBaseline
from src.models.time_then_space import TimeThenSpaceModel

# --- Rutas (tests/ -> ia_prediction_service -> CerebroVial) ---
IA_ROOT = Path(__file__).resolve().parents[1]
GRU_CKPT = IA_ROOT / "models" / "miraflores_gru_baseline.pt"
STGNN_CKPT = IA_ROOT / "models" / "miraflores_stgnn.pt"
SCRIPTS = IA_ROOT / "scripts"
MAPPING = IA_ROOT / "src" / "data" / "artifacts" / "miraflores_graph_lcc_mapping.json"

N_NODES = 1660
HORIZON = 30
LOOKBACK = 30
K_SNAPSHOTS = 8          # snapshots deterministas para el gate (rápido, ~segundos)
ATOL, RTOL = 1e-5, 1e-4  # CPU en ambos lados -> tolerancia estricta honesta

_npz_missing = not Path(DEFAULT_NPZ).exists()
_gru_missing = not GRU_CKPT.exists()
_stgnn_missing = not STGNN_CKPT.exists()


# --------------------------------------------------------------------------- #
# Carga del oráculo: predict_collect de los scripts de training (sin efectos
# colaterales — todo lo demás está bajo `if __name__ == "__main__"`).
# --------------------------------------------------------------------------- #
def _load_script(path: Path, mod_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # corre imports + defs de módulo; NO main()
    return mod


@pytest.fixture(scope="module")
def real_data():
    if _npz_missing:
        pytest.skip(f".npz de Fase 2 ausente: {DEFAULT_NPZ}")
    data = load_npz(DEFAULT_NPZ)
    return data["timeloss"], data["mask"], data["seeds"]


@pytest.fixture(scope="module")
def test_snapshots(real_data):
    """Primeros K snapshots (day, start) del fold TEST + scaler train-only + windows."""
    timeloss, mask, seeds = real_data
    n_t = timeloss.shape[1]
    train_seeds, _val, test_seeds = get_split()  # scaler train-only; gate sobre TEST
    scaler = compute_timeloss_scaler(timeloss, mask, seeds, train_seeds)
    gidx = build_graph_window_index(seeds, test_seeds, n_timesteps=n_t, stride=12)
    assert gidx.shape[0] >= K_SNAPSHOTS, "fold test demasiado chico para el gate"
    gidx_k = gidx[:K_SNAPSHOTS].astype(np.int32)          # [K, 2] (day, start)
    # Window de grafo por snapshot: [K, 30, 1660, 2] (entrada común a ambos adaptadores).
    Xg, _y, _ym = gather_graph_batch(timeloss, mask, gidx_k)
    return {
        "timeloss": timeloss, "mask": mask, "seeds": seeds,
        "scaler": scaler, "gidx_k": gidx_k, "Xg": Xg,
    }


# --------------------------------------------------------------------------- #
# 1. GATE — reproducción del eval (condición de cierre de Fase 1)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_gru_missing or _npz_missing, reason="ckpt GRU o npz ausente")
def test_gate_gru_reproduce_eval(test_snapshots):
    """GruAdapter.predict == predict_collect del baseline, CPU en ambos lados.

    Incluye el BLINDAJE de orden de nodos: assert explícito de que el índice GRU del
    oráculo y el window del adaptador recorren las 1660 aristas en el mismo orden.
    """
    s = test_snapshots
    timeloss, mask = s["timeloss"], s["mask"]
    mean, std = s["scaler"]["mean"], s["scaler"]["std"]
    gidx_k, Xg = s["gidx_k"], s["Xg"]
    K = gidx_k.shape[0]

    # --- Índice GRU EXPLÍCITO: snapshot-major, node_index-minor (orden node_index) ---
    rows = np.empty((K * N_NODES, 3), dtype=np.int32)
    for k, (day, start) in enumerate(gidx_k):
        sl = slice(k * N_NODES, (k + 1) * N_NODES)
        rows[sl, 0] = day
        rows[sl, 1] = start
        rows[sl, 2] = np.arange(N_NODES, dtype=np.int32)   # node_index 0..1659 ascendente

    # --- BLINDAJE: entradas idénticas y alineadas en orden de nodos ---
    # gather_batch del índice GRU (orden de filas = node_index por snapshot) debe igualar
    # la transpuesta del window de grafo [K,30,1660,2] -> [K,1660,30,2] -> [K*1660,30,2].
    Xb, _yb, _ymb = gather_batch(timeloss, mask, rows)                 # [K*1660, 30, 2]
    Xg_t = np.transpose(Xg, (0, 2, 1, 3)).reshape(K * N_NODES, LOOKBACK, 2)
    assert np.array_equal(Xb, Xg_t), (
        "DESALINEO de orden de nodos entre el índice GRU del oráculo y el window del "
        "adaptador: las entradas no coinciden. El gate NO es válido hasta resolver esto."
    )

    # --- Oráculo: predict_collect del baseline en CPU ---
    baseline = _load_script(SCRIPTS / "train_miraflores_baseline.py", "oracle_baseline")
    baseline.DEV = "cpu"   # forzar CPU (el script usa un global de módulo "mps" si disponible)
    ckpt = torch.load(GRU_CKPT, map_location="cpu", weights_only=False)
    oracle_model = MirafloresGRUBaseline(**ckpt["arch"]).to("cpu")
    oracle_model.load_state_dict(ckpt["state_dict"])
    pred_o, _y, _ym = baseline.predict_collect(
        oracle_model, timeloss, mask, rows, mean, std, batch=512
    )
    pred_oracle = pred_o.numpy().reshape(K, N_NODES, HORIZON)          # [K,1660,30]

    # --- Loader bajo prueba ---
    adapter = load_inference_adapter(GRU_CKPT, device="cpu")
    loader_out = np.stack([adapter.predict(Xg[k]) for k in range(K)])  # [K,1660,30]

    assert loader_out.shape == pred_oracle.shape
    assert np.allclose(loader_out, pred_oracle, atol=ATOL, rtol=RTOL), (
        f"GRU: el loader NO reproduce el eval. max|Δ|={np.abs(loader_out - pred_oracle).max():.3e}"
    )


@pytest.mark.skipif(_stgnn_missing or _npz_missing, reason="ckpt STGNN o npz ausente")
def test_gate_stgnn_reproduce_eval(test_snapshots):
    """StgnnAdapter.predict == predict_collect del STGNN, CPU en ambos lados."""
    s = test_snapshots
    timeloss, mask = s["timeloss"], s["mask"]
    mean, std = s["scaler"]["mean"], s["scaler"]["std"]
    gidx_k, Xg = s["gidx_k"], s["Xg"]
    K = gidx_k.shape[0]

    # --- Oráculo: predict_collect del STGNN en CPU (dev es parámetro) ---
    stgnn = _load_script(SCRIPTS / "train_miraflores_stgnn.py", "oracle_stgnn")
    ckpt = torch.load(STGNN_CKPT, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    oracle_model = TimeThenSpaceModel(
        input_size=arch["input_size"], n_nodes=arch["n_nodes"], horizon=arch["horizonte"],
        emb_size=arch["emb"], hidden_size=arch["hidden"], rnn_layers=arch["rnn_layers"],
        gnn_kernel=arch["gnn_kernel"], output_size=arch["output_size"],
    ).to("cpu")
    oracle_model.load_state_dict(ckpt["state_dict"])
    ei_np, ew_np = load_lcc_edge_index()
    ei = torch.from_numpy(ei_np)
    ew = torch.from_numpy(ew_np)
    pred_o, _y, _ym = stgnn.predict_collect(
        oracle_model, timeloss, mask, gidx_k, mean, std, 16, ei, ew, "cpu"
    )
    # predict_collect STGNN devuelve [K, 30, 1660] -> a [K, 1660, 30]
    pred_oracle = pred_o.numpy().transpose(0, 2, 1)                    # [K,1660,30]

    # --- Loader bajo prueba ---
    adapter = load_inference_adapter(STGNN_CKPT, device="cpu")
    loader_out = np.stack([adapter.predict(Xg[k]) for k in range(K)])  # [K,1660,30]

    assert loader_out.shape == pred_oracle.shape
    assert np.allclose(loader_out, pred_oracle, atol=ATOL, rtol=RTOL), (
        f"STGNN: el loader NO reproduce el eval. max|Δ|={np.abs(loader_out - pred_oracle).max():.3e}"
    )


# --------------------------------------------------------------------------- #
# 2. (opt-in `slow`) métricas completas vs json persistido — confirmación agregada
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW") or _gru_missing or _npz_missing,
    reason="opt-in: set RUN_SLOW=1 (y requiere ckpt GRU + npz)",
)
def test_gru_full_metrics_vs_json(real_data):
    """Corre el loader sobre TODO el test set y compara métricas vs el json persistido.

    Confirmación agregada del gate per-predicción (test 1). Pesa ~minutos en CPU -> `slow`.
    """
    from src.evaluation.miraflores_metrics import DEFAULT_CUTS, masked_metrics_by_horizon

    metrics_json = SCRIPTS / "miraflores_baseline_metrics.json"
    if not metrics_json.exists():
        pytest.skip("json de métricas baseline ausente")
    timeloss, mask, seeds = real_data
    n_t = timeloss.shape[1]
    train_seeds, _v, test_seeds = get_split()
    gidx = build_graph_window_index(seeds, test_seeds, n_timesteps=n_t, stride=12)
    adapter = load_inference_adapter(GRU_CKPT, device="cpu")

    preds, ys, ms = [], [], []
    for row in gidx:
        Xg, y, ym = gather_graph_batch(timeloss, mask, row[None, :])  # [1,30,1660,2]
        out = adapter.predict(Xg[0])                                  # [1660,30]
        preds.append(out)
        ys.append(y[0].transpose(1, 0))                              # [1660,30]
        ms.append(ym[0].transpose(1, 0))
    pred_m = np.concatenate(preds, axis=0)
    y_m = np.concatenate(ys, axis=0)
    mask_m = np.concatenate(ms, axis=0)
    metrics = masked_metrics_by_horizon(pred_m, y_m, mask_m, DEFAULT_CUTS)

    stored = json.loads(metrics_json.read_text())["test_all"]
    for metric in ("MAE", "RMSE"):
        for c in DEFAULT_CUTS:
            got = metrics[metric][c]
            exp = stored[metric][str(c)]
            assert abs(got - exp) < 1e-2, f"{metric}@{c}: loader={got:.4f} json={exp:.4f}"


# --------------------------------------------------------------------------- #
# 3. Orden de nodos — invariante sagrada
# --------------------------------------------------------------------------- #
def test_mapping_node_order_integrity():
    """node_index == posición en la lista, y la arista 0 conocida es '-1098384939'."""
    if not MAPPING.exists():
        pytest.skip("mapping LCC ausente")
    data = json.loads(MAPPING.read_text())
    nodes = data["nodes"]
    assert len(nodes) == N_NODES
    assert data["counts"]["nodes"] == N_NODES
    assert len(data["edges"]) == 2948
    for i, n in enumerate(nodes):
        assert n["node_index"] == i, f"node_index roto en posición {i}"
    assert nodes[0]["sumo_edge"] == "-1098384939"


@pytest.mark.skipif(_gru_missing or _npz_missing, reason="ckpt GRU o npz ausente")
def test_gru_output_order_equivariance(test_snapshots):
    """GRU nodo-agnóstico: permutar nodos del window permuta filas de salida idénticamente.

    Prueba directa de que la fila ``i`` de la salida corresponde al nodo ``i`` de la
    entrada (orden node_index), sin depender del oráculo.
    """
    s = test_snapshots
    adapter = load_inference_adapter(GRU_CKPT, device="cpu")
    window = s["Xg"][0]                                   # [30,1660,2]
    base = adapter.predict(window)                        # [1660,30]
    perm = np.random.default_rng(0).permutation(N_NODES)
    permuted_window = window[:, perm, :]                  # permuta el eje de nodos
    out_perm = adapter.predict(permuted_window)           # [1660,30]
    assert np.allclose(out_perm, base[perm], atol=1e-6), (
        "GRU no es order-equivariant: la fila i de la salida NO sigue al nodo i de la entrada."
    )


# --------------------------------------------------------------------------- #
# 4. Unitarios de carga (registry / remapeo de arch / errores / shapes)
# --------------------------------------------------------------------------- #
def test_registry_resolves_and_rejects_unknown():
    assert resolve("MirafloresGRUBaseline") is GruAdapter
    assert resolve("TimeThenSpaceModel") is StgnnAdapter
    with pytest.raises(ValueError, match="modelo desconocido"):
        resolve("ModeloInexistente")


@pytest.mark.skipif(_stgnn_missing, reason="ckpt STGNN ausente")
def test_stgnn_arch_remap_loads_state_dict_strict():
    """El remapeo de arch STGNN reconstruye un modelo cuyo state_dict carga estricto."""
    ckpt = torch.load(STGNN_CKPT, map_location="cpu", weights_only=False)
    adapter = StgnnAdapter.from_checkpoint(ckpt, device="cpu")  # load_state_dict estricto
    assert set(adapter.model.state_dict().keys()) == set(ckpt["state_dict"].keys())


@pytest.mark.skipif(_gru_missing, reason="ckpt GRU ausente")
def test_gru_arch_direct_loads_state_dict_strict():
    ckpt = torch.load(GRU_CKPT, map_location="cpu", weights_only=False)
    adapter = GruAdapter.from_checkpoint(ckpt, device="cpu")
    assert set(adapter.model.state_dict().keys()) == set(ckpt["state_dict"].keys())


@pytest.mark.skipif(_gru_missing, reason="ckpt GRU ausente")
def test_contract_shapes_and_bad_window():
    adapter = load_inference_adapter(GRU_CKPT, device="cpu")
    good = np.zeros((LOOKBACK, N_NODES, 2), dtype=np.float32)
    out = adapter.predict(good)
    assert out.shape == (N_NODES, HORIZON)
    assert out.dtype == np.float32
    for bad in (np.zeros((LOOKBACK, N_NODES)), np.zeros((30, 100, 2)), np.zeros((1, 30, 1660, 2))):
        with pytest.raises(ValueError, match="window debe ser"):
            adapter.predict(bad)


def test_loader_rejects_ckpt_without_model_key(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, bad)
    with pytest.raises(ValueError, match="no tiene la llave 'model'"):
        load_inference_adapter(bad, device="cpu")
