"""Tests del modelo, métricas y loss enmascarado del baseline (STGNN Fase 3).

- Modelo: forward de un batch → salida [B,30] sin NaN con input válido.
- Métricas: MAE/RMSE/R² enmascarados ignoran pasos inválidos (máscara conocida);
  la evaluación por horizonte devuelve los 6 cortes.
- Loss: MSE enmascarado ignora pasos de target inválidos.

Corre con el venv raíz:
``cd ia_prediction_service && ../.venv/bin/python -m pytest tests/test_miraflores_baseline.py``
"""
from __future__ import annotations

import math

import torch

from src.evaluation.miraflores_metrics import (
    DEFAULT_CUTS,
    masked_metrics_by_horizon,
    masked_mse_loss,
)
from src.models.miraflores_gru_baseline import MirafloresGRUBaseline


# ----------------------------- Modelo -----------------------------------------
def test_modelo_forward_shape_sin_nan():
    torch.manual_seed(0)
    model = MirafloresGRUBaseline(hidden=64, horizonte=30, input_size=2)
    x = torch.randn(8, 30, 2)
    x[..., 1] = (x[..., 1] > 0).float()  # canal 1 como indicador 0/1
    out = model(x)
    assert out.shape == (8, 30)
    assert not torch.isnan(out).any()


def test_modelo_horizonte_configurable():
    model = MirafloresGRUBaseline(hidden=16, horizonte=12, input_size=2)
    out = model(torch.randn(4, 30, 2))
    assert out.shape == (4, 12)


# ----------------------------- Métricas ---------------------------------------
def test_metricas_devuelve_seis_cortes():
    B, H = 5, 30
    pred = torch.randn(B, H)
    target = torch.randn(B, H)
    mask = torch.ones(B, H)
    m = masked_metrics_by_horizon(pred, target, mask)
    for name in ("MAE", "RMSE", "R2"):
        assert set(m[name].keys()) == set(DEFAULT_CUTS)
        assert len(m[name]) == 6


def test_metricas_enmascaradas_ignoran_pasos_invalidos():
    # pred == target en TODAS las posiciones válidas; corrompemos sólo inválidas.
    B, H = 4, 30
    target = torch.arange(B * H, dtype=torch.float32).reshape(B, H)
    pred = target.clone()
    mask = torch.ones(B, H)
    # Posición inválida con predicción absurda: NO debe afectar las métricas.
    mask[0, 4] = 0.0           # corte 5 (j=4)
    pred[0, 4] = 1e9
    m = masked_metrics_by_horizon(target=target, pred=pred, mask=mask)
    # Error 0 en todas las posiciones válidas → MAE=RMSE=0 en cada corte.
    for c in DEFAULT_CUTS:
        assert abs(m["MAE"][c]) < 1e-6, f"MAE corte {c}"
        assert abs(m["RMSE"][c]) < 1e-6, f"RMSE corte {c}"
        # pred==target con target variando entre ventanas → R²=1.
        assert abs(m["R2"][c] - 1.0) < 1e-6, f"R2 corte {c}"


def test_metricas_valor_mae_conocido():
    # Error constante de 2.0 en todas las posiciones válidas → MAE=2, RMSE=2.
    B, H = 3, 30
    target = torch.arange(B * H, dtype=torch.float32).reshape(B, H)
    pred = target + 2.0
    mask = torch.ones(B, H)
    m = masked_metrics_by_horizon(pred, target, mask)
    for c in DEFAULT_CUTS:
        assert abs(m["MAE"][c] - 2.0) < 1e-5
        assert abs(m["RMSE"][c] - 2.0) < 1e-5


def test_metricas_corte_sin_validos_es_nan():
    B, H = 4, 30
    pred = torch.randn(B, H)
    target = torch.randn(B, H)
    mask = torch.ones(B, H)
    mask[:, 9] = 0.0  # corte 10 (j=9) sin ninguna ventana válida
    m = masked_metrics_by_horizon(pred, target, mask)
    assert math.isnan(m["MAE"][10])
    assert math.isnan(m["RMSE"][10])
    assert math.isnan(m["R2"][10])
    # Otros cortes siguen siendo finitos.
    assert not math.isnan(m["MAE"][5])


# ----------------------------- Loss enmascarado -------------------------------
def test_loss_ignora_target_invalido():
    # pred == target en válidos; basura en inválidos → loss == 0.
    B, H = 6, 30
    target = torch.randn(B, H)
    pred = target.clone()
    mask = torch.ones(B, H)
    mask[1, 7] = 0.0
    mask[3, 20] = 0.0
    pred[1, 7] = 12345.0   # inválido: debe ignorarse
    pred[3, 20] = -9999.0
    loss = masked_mse_loss(pred, target, mask)
    assert loss.item() < 1e-10


def test_loss_valor_conocido():
    # Error constante 3.0 en todos los pasos válidos → MSE = 9.0.
    B, H = 2, 30
    target = torch.zeros(B, H)
    pred = torch.full((B, H), 3.0)
    mask = torch.ones(B, H)
    loss = masked_mse_loss(pred, target, mask)
    assert abs(loss.item() - 9.0) < 1e-6


def test_loss_diferenciable():
    pred = torch.randn(4, 30, requires_grad=True)
    target = torch.randn(4, 30)
    mask = torch.ones(4, 30)
    mask[:, 0] = 0.0
    loss = masked_mse_loss(pred, target, mask)
    loss.backward()
    assert pred.grad is not None
    # El paso enmascarado (col 0) no recibe gradiente.
    assert torch.allclose(pred.grad[:, 0], torch.zeros(4))
