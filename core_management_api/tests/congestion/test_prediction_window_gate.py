"""STAGE-GATE 2 (Fase 3) — la ventana servida en el core es fiel a la del training.

Es el gate central: si la entrada de inferencia no es la que el modelo vio, todo lo demás
da igual. Dos verificaciones:

1. **Ventana byte-igual**: ``build_window`` (core) sobre el slice == ``gather_graph_batch``
   (windower de training, oráculo) sobre el ``.npz`` completo, para el mismo
   ``(day_idx=9, start=t-29)``. ``np.array_equal`` (con ``equal_nan`` no hace falta: los
   canales ya tienen vacío→0). Si falla → desalineo (probable orden de nodos).
2. **Reproducción de predict**: el camino vendorizado del core (``load_gru_adapter`` +
   ``GruAdapter.predict``) reproduce el ORIGINAL de Fase 1, sobre la MISMA ventana,
   ``np.allclose(atol=1e-5, rtol=1e-4)``. El oráculo se carga desde los archivos ORIGINALES
   de ``ia_prediction_service`` vía ``importlib`` (evita la colisión del paquete ``src``
   core-vs-ia y el import de ``tsl`` que arrastra ``adapters.py``): así se prueba que las
   COPIAS vendorizadas reproducen al original byte-a-byte en comportamiento.

DEPENDE DEL ``.npz`` COMPLETO (gitignored/regenerable, 574 MB) → NO corre en CI ni en un
clon limpio. El skip es RUIDOSO a propósito (warning + mensaje explícito): un gate skipeado
que parece verde es peor que uno que falla. Deuda registrada en
``documentation/ESTADO_Y_PROXIMOS_PASOS.md``: materializar un fixture mínimo es trabajo
posterior. Correr local (con el ``.npz`` presente):
  cd core_management_api && PYTHONPATH=. ../.venv/bin/python -m pytest \
    tests/congestion/test_prediction_window_gate.py -v
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from src.congestion.infrastructure.gru_inference_adapter import load_gru_adapter
from src.congestion.infrastructure.gru_window_builder import build_window

_REPO = Path(__file__).resolve().parents[3]
_FULL_NPZ = _REPO / "simulation/data/datasets/miraflores_laborable_60d/tensors/miraflores_laborable_60d.npz"
_IA = _REPO / "ia_prediction_service"
_CORE_CKPT = _REPO / "core_management_api/models/miraflores_gru_baseline.pt"
_IA_CKPT = _IA / "models/miraflores_gru_baseline.pt"
_SLICE = _REPO / "core_management_api/models/day_seed051_tensor.npz"

DAY_IDX = 9      # ↔ seed051
T_CUT = 1155     # 19:15 (minuto del día) — el t de la demo
LOOKBACK = 30


def _load_module(path: Path, name: str) -> ModuleType:
    """Carga un .py original (torch/numpy-puro, sin imports relativos) vía importlib."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Skip RUIDOSO si falta el .npz completo: la FIDELIDAD de la ventana NO queda verificada.
if not _FULL_NPZ.exists():
    _MSG = (
        "STAGE-GATE 2 SKIPPED: falta el .npz completo (gitignored/regenerable) en "
        f"{_FULL_NPZ}. La FIDELIDAD de la ventana NO está verificada en este entorno "
        "(no corre en CI ni en clon limpio). Regenerar con miraflores_dataset_builder "
        "para ejercer el gate. Deuda: materializar un fixture mínimo versionado."
    )
    warnings.warn(_MSG, stacklevel=2)
    pytest.skip(_MSG, allow_module_level=True)


@pytest.fixture(scope="module")
def windows():
    """Ventana del core (desde el slice) y del oráculo (gather_graph_batch sobre el .npz)."""
    # Oráculo: el windower de training, cargado por archivo (numpy-puro, sin colisión src).
    windower = _load_module(_IA / "src/data/miraflores_windower.py", "oracle_windower")
    z = np.load(_FULL_NPZ, allow_pickle=False)
    full_tl = np.ascontiguousarray(z["tensor"][..., 0].astype(np.float32))  # [D, T, N]
    full_mask = np.ascontiguousarray(z["mask"].astype(bool))                # [D, T, N]
    gidx = np.array([[DAY_IDX, T_CUT - LOOKBACK + 1]], dtype=np.int32)      # (day, start)
    Xg, _y, _ym = windower.gather_graph_batch(full_tl, full_mask, gidx)     # [1, 30, N, 2]
    oracle_window = Xg[0]                                                    # [30, 1660, 2]

    # Core: build_window sobre el slice auto-contenido.
    s = np.load(_SLICE, allow_pickle=True)
    slice_tl = s["tensor"][..., 0].astype(np.float32)   # [1440, 1660]
    slice_mask = s["mask"].astype(bool)                  # [1440, 1660]
    core_window = build_window(slice_tl, slice_mask, T_CUT)  # [30, 1660, 2]
    return core_window, oracle_window


def test_window_byte_equal(windows):
    """La ventana del core es byte-igual a la de gather_graph_batch (misma que el training)."""
    core_window, oracle_window = windows
    assert core_window.shape == (LOOKBACK, 1660, 2)
    assert np.array_equal(core_window, oracle_window), (
        "STAGE-GATE 2 FALLA: la ventana del core NO es byte-igual a la del training "
        f"(t={T_CUT}, day_idx={DAY_IDX}). max|Δ|={np.abs(core_window - oracle_window).max():.3e}. "
        "Probable desalineo de orden de nodos."
    )


def test_predict_reproduces_phase1(windows):
    """El camino vendorizado del core reproduce el ORIGINAL de Fase 1 sobre la misma ventana."""
    core_window, oracle_window = windows
    assert np.array_equal(core_window, oracle_window)  # precondición del gate anterior

    # Core: camino vendorizado completo.
    core_adapter = load_gru_adapter(_CORE_CKPT, device="cpu")
    core_pred = core_adapter.predict(core_window)   # [1660, 30] segundos

    # Oráculo: modelo + preprocessing ORIGINALES de ia_prediction_service (por archivo).
    model_mod = _load_module(_IA / "src/models/miraflores_gru_baseline.py", "oracle_gru_model")
    prep_mod = _load_module(_IA / "src/inference/preprocessing.py", "oracle_prep")
    ckpt = torch.load(_IA_CKPT, map_location="cpu", weights_only=False)
    o_model = model_mod.MirafloresGRUBaseline(**ckpt["arch"]).to("cpu")
    o_model.load_state_dict(ckpt["state_dict"])
    o_model.eval()
    mean, std = ckpt["scaler"]["mean"], ckpt["scaler"]["std"]
    x = torch.from_numpy(np.ascontiguousarray(oracle_window, dtype=np.float32))
    x = prep_mod.standardize_window(x, mean, std).permute(1, 0, 2).contiguous()  # [1660,30,2]
    with torch.no_grad():
        o_std = o_model(x)
    o_pred = prep_mod.destandardize(o_std, mean, std).cpu().numpy().astype(np.float32)

    assert np.allclose(core_pred, o_pred, atol=1e-5, rtol=1e-4), (
        "STAGE-GATE 2 FALLA: el camino vendorizado del core NO reproduce el de Fase 1. "
        f"max|Δ|={np.abs(core_pred - o_pred).max():.3e}"
    )


def test_build_window_rango_invalido():
    """build_window valida el rango de t (la ventana t-29..t debe caer en el día)."""
    s = np.load(_SLICE, allow_pickle=True)
    tl = s["tensor"][..., 0].astype(np.float32)
    mask = s["mask"].astype(bool)
    with pytest.raises(ValueError):
        build_window(tl, mask, 28)     # < LOOKBACK-1
    with pytest.raises(ValueError):
        build_window(tl, mask, 1440)   # > N_TIMESTEPS-1
