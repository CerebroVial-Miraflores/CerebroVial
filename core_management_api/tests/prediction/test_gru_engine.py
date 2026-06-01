"""Test de paridad del andamiaje GRU vendorizado (TTH-09 Fase 3a, contrato §4).

Verifica que la clase vendorizada en el core (``gru_model.GRUMultiOutput``) es
compatible con los state_dicts reales entrenados en ``ia_prediction_service`` y que
produce exactamente la misma salida que la clase de origen. Tres comprobaciones por
dirección (N/S/E/W):

1. ``load_state_dict(strict=True)`` exacto: sin claves faltantes ni sobrantes.
2. Forward produce el shape del contrato ``(batch, 30, 6)`` = (batch, horizonte, n_classes).
3. Paridad real vs la clase ORIGINAL (cargada por path con importlib, sin necesidad
   de que ``ia_prediction_service`` sea instalable): mismo state_dict + mismo input
   => ``torch.allclose``.

Skip limpio (NO fallo) cuando el entorno no tiene los recursos:
- torch no instalado            -> pytest.importorskip
- .pt ausente                   -> pytest.skip
- .pt es puntero LFS sin pull   -> pytest.skip  (caso CI sin ``git lfs pull``)
- clase original ausente        -> pytest.skip  (solo afecta la paridad cruzada)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# torch es dependencia CPU-only del core (D-010). Si no está, se skipea toda la suite
# de este archivo antes de importar la clase vendorizada (que importa torch).
torch = pytest.importorskip("torch")

from cerebrovial_shared.lfs_check import LFSPointerError, assert_real_binary  # noqa: E402
from src.prediction.infrastructure.gru_model import GRUMultiOutput  # noqa: E402

# test_gru_engine.py -> prediction[0] -> tests[1] -> core_management_api[2] -> repo[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _REPO_ROOT / "ia_prediction_service" / "models"
_ORIGINAL_CLASS_FILE = (
    _REPO_ROOT / "ia_prediction_service" / "src" / "models" / "gru_multioutput.py"
)

DIRECTIONS = ("N", "S", "E", "W")

# Longitud de la secuencia de entrada (lookback). Por metadata.json: lookback_min ==
# horizonte_min == 30. El shape de salida NO depende de la longitud de entrada (la
# cabeza proyecta el último estado oculto), pero se fija a 30 para reflejar el uso real.
_LOOKBACK = 30


def _load_checkpoint(direction: str) -> dict:
    """Carga el ckpt de una dirección, o skip limpio si no está disponible."""
    path = _MODELS_DIR / f"gru_{direction}.pt"
    if not path.is_file():
        pytest.skip(f"checkpoint GRU {direction} no disponible en {path}")
    try:
        assert_real_binary(path)
    except LFSPointerError:
        pytest.skip(
            f"checkpoint GRU {direction} es puntero LFS sin materializar "
            "(correr 'git lfs pull')"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_vendored(ckpt: dict) -> GRUMultiOutput:
    """Instancia la clase vendorizada con los hiperparámetros del ckpt."""
    return GRUMultiOutput(
        hidden=int(ckpt["hyperparams"]["hidden"]),
        horizonte=int(ckpt["horizonte"]),
        n_classes=int(ckpt["n_classes"]),
    )


def _fixed_input() -> "torch.Tensor":
    """Input determinista (batch=1, lookback, 1) normalizado /5.0. Sin aleatoriedad."""
    return torch.arange(_LOOKBACK, dtype=torch.float32).reshape(1, _LOOKBACK, 1) / 5.0


def _load_original_class():
    """Carga la clase GRUMultiOutput ORIGINAL por path (importlib), sin instalar el módulo."""
    if not _ORIGINAL_CLASS_FILE.is_file():
        pytest.skip(f"clase original no disponible en {_ORIGINAL_CLASS_FILE}")
    spec = importlib.util.spec_from_file_location(
        "gru_multioutput_original", _ORIGINAL_CLASS_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GRUMultiOutput


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_vendored_loads_real_state_dict_strict(direction: str):
    """La clase vendorizada carga el state_dict real exacto (sin claves faltantes/sobrantes)."""
    ckpt = _load_checkpoint(direction)
    model = _build_vendored(ckpt)
    result = model.load_state_dict(ckpt["state_dict"], strict=True)
    # strict=True ya lanza ante mismatch; verificación explícita por claridad.
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_forward_output_shape(direction: str):
    """El forward produce el shape del contrato §4: (batch, 30, 6)."""
    ckpt = _load_checkpoint(direction)
    model = _build_vendored(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        out = model(_fixed_input())
    assert tuple(out.shape) == (1, int(ckpt["horizonte"]), int(ckpt["n_classes"]))
    assert tuple(out.shape) == (1, 30, 6)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_parity_vendored_vs_original(direction: str):
    """Paridad real: la clase vendorizada y la original dan la MISMA salida.

    Paridad POR CONSTRUCCIÓN: mismo state_dict + mismo input fijo + ``.eval()`` (GRU de
    1 capa, sin dropout) => forward determinista. NO se usa ``torch.manual_seed``: si la
    paridad dependiera de la seed, habría aleatoriedad indebida en el forward y este
    test debería fallar para delatarla.
    """
    ckpt = _load_checkpoint(direction)
    original_cls = _load_original_class()

    kwargs = dict(
        hidden=int(ckpt["hyperparams"]["hidden"]),
        horizonte=int(ckpt["horizonte"]),
        n_classes=int(ckpt["n_classes"]),
    )
    vendored = GRUMultiOutput(**kwargs)
    original = original_cls(**kwargs)
    vendored.load_state_dict(ckpt["state_dict"])
    original.load_state_dict(ckpt["state_dict"])
    vendored.eval()
    original.eval()

    x = _fixed_input()
    with torch.no_grad():
        out_vendored = vendored(x)
        out_original = original(x)
    assert torch.allclose(out_vendored, out_original)
