"""Métricas y loss enmascarados del baseline GRU (STGNN Fase 3). torch puro, sin tsl.

Todo se evalúa **enmascarado**: los pasos de target inválidos (celdas vacías, sin
señal) se ignoran vía máscara — nunca cuentan como demora 0.

- ``masked_mse_loss``: MSE enmascarado, diferenciable (loss de entrenamiento, en el
  espacio en que se le pase — el trainer lo usa estandarizado).
- ``masked_metrics_by_horizon``: MAE / RMSE / R² enmascarados **por horizonte** en
  los cortes de paso 5/10/15/20/25/30 (el h-ésimo paso predicho, single-step).
  Recibe predicciones y targets **des-estandarizados (segundos)**. Devuelve una
  tabla ``métrica × horizonte``.
"""
from __future__ import annotations

import math

import torch

DEFAULT_CUTS: tuple[int, ...] = (5, 10, 15, 20, 25, 30)


def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE promediado SÓLO sobre los pasos válidos (``mask`` > 0). Diferenciable.

    ``pred``/``target``/``mask`` shape ``[B, H]``. Si no hay ningún paso válido en el
    batch, devuelve 0 (denominador piso-limitado a 1) para no romper el backward.
    """
    m = mask.to(pred.dtype)
    sq = (pred - target) ** 2 * m
    denom = m.sum().clamp_min(1.0)
    return sq.sum() / denom


def _as_2d_tensor(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return t.to(torch.float64)  # float64 para estabilidad de R²/sumas


def masked_metrics_by_horizon(
    pred, target, mask, cuts: tuple[int, ...] = DEFAULT_CUTS
) -> dict[str, dict[int, float]]:
    """MAE/RMSE/R² enmascarados por horizonte (en las unidades de entrada: segundos).

    En el corte ``c`` se evalúa la columna ``c-1`` (el c-ésimo paso predicho) sobre
    todas las ventanas cuyo target en ese paso es válido. R² = ``1 - SS_res/SS_tot``
    con SS sobre entradas válidas y media de los targets válidos a ese horizonte.
    Si un corte no tiene entradas válidas (o SS_tot==0 para R²), devuelve ``nan``.
    """
    p = _as_2d_tensor(pred)
    t = _as_2d_tensor(target)
    m = _as_2d_tensor(mask) > 0.5
    if p.shape != t.shape or p.shape != m.shape:
        raise ValueError(f"shapes deben coincidir: pred{tuple(p.shape)} "
                         f"target{tuple(t.shape)} mask{tuple(m.shape)}")
    horizon = p.shape[1]

    out: dict[str, dict[int, float]] = {"MAE": {}, "RMSE": {}, "R2": {}}
    for c in cuts:
        if c < 1 or c > horizon:
            raise ValueError(f"corte {c} fuera de rango [1,{horizon}]")
        j = c - 1
        mj = m[:, j]
        nvalid = int(mj.sum().item())
        if nvalid == 0:
            out["MAE"][c] = float("nan")
            out["RMSE"][c] = float("nan")
            out["R2"][c] = float("nan")
            continue
        pj = p[mj, j]
        tj = t[mj, j]
        err = pj - tj
        mae = float(err.abs().mean().item())
        rmse = float(torch.sqrt((err ** 2).mean()).item())
        ss_res = float((err ** 2).sum().item())
        tbar = float(tj.mean().item())
        ss_tot = float(((tj - tbar) ** 2).sum().item())
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        out["MAE"][c] = mae
        out["RMSE"][c] = rmse
        out["R2"][c] = r2
    return out


def format_metrics_table(
    metrics: dict[str, dict[int, float]], cuts: tuple[int, ...] = DEFAULT_CUTS
) -> str:
    """Render legible de la tabla ``métrica × horizonte`` (horizonte en minutos)."""
    header = "  métrica │ " + " ".join(f"{c:>8d}min" for c in cuts)
    sep = "  ────────┼" + "─" * (len(cuts) * 12)
    lines = [header, sep]
    for name in ("MAE", "RMSE", "R2"):
        row = metrics.get(name, {})
        cells = []
        for c in cuts:
            v = row.get(c, float("nan"))
            cells.append("     nan   " if (v != v or math.isinf(v)) else f"{v:>10.3f} ")
        lines.append(f"  {name:>6} │ " + "".join(cells))
    return "\n".join(lines)
