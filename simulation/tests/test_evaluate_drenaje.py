"""Test de aceptación del evaluador de drenaje (D-014, B3.2.a).

Ancla a la **clasificación cualitativa** del C3 (no a números frágiles): el evaluador debe
clasificar el fixture s11 (scale 1.1, C3 fila 1.1) como **DRENA** y s13 (scale 1.3, C3 fila 1.3)
como **COLAPSA**, computando las tres señales de D-014 sobre los fixtures versionados
(``fixtures/drenaje_c3/``: stats enteros + edgeData recortado-stripeado, metric-equivalente al
crudo — ver ``make_drenaje_fixtures.py``).

Señal #3 = fracción de intervalos sub-20 km/h por ventana de pico > 10 % (enmienda D-014 2026-06-03;
ver § D-014). La operacionalización vieja ("racha > 15 min consecutivos") era inerte: el dip de v2 es
intermitente-frecuente, no sostenido.

Los fixtures traen margen ±30 min alrededor de los picos. Estos tests verifican implícitamente que
el evaluador **acota el dip a las ventanas de pico** ``[25200,32400)`` / ``[64800,72000)`` y no se
confunde con el margen: si computara la fracción sobre todo el archivo, los intervalos del margen
contaminarían el cómputo. Que s11 dé DRENA (su veredicto exige dip OK) es la verificación de que el
acotamiento funciona.

Gate de B3.2.a: este test verde + los números del dip (fracción sub-20 + mean_kmh mín por ventana)
mostrados al revisar. La separación s11/s13 en la fracción debe respetar los márgenes fijados
(s11 < 10 %, s13 > 10 %); si no separa, se para y se vuelve al chat sin tocar el umbral.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# evaluate_drenaje vive en simulation/scripts/, que NO es paquete ni está en el path de tests
# (los tests de simulation importan cerebrovial_simulation.*; este es el primero que toca scripts/).
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import evaluate_drenaje as ed  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "drenaje_c3"


def _eval(tag: str) -> dict:
    return ed.evaluate_day(_FIXTURES / f"stats_{tag}.xml", _FIXTURES / f"edgedata_{tag}.xml")


def test_s11_drena():
    """C3 fila 1.1: 11 teleports, +2% duración, fracción sub-20 < 10% en ambas ventanas → DRENA."""
    r = _eval("s11")
    s = r["signals"]
    assert s["teleports"]["value"] == 11               # C3 fila 1.1
    assert s["duration_s"]["value"] == pytest.approx(259.30, abs=0.5)
    assert s["teleports"]["ok"]
    assert s["duration_s"]["ok"]
    # señal #3 (enmienda D-014): fracción sub-20 por debajo del corte de 10% en AM y PM.
    for w in s["dip"]["windows"].values():
        assert w["frac_sub20"] < ed.DIP_FRAC_MAX
    assert s["dip"]["ok"]
    assert r["drains"] is True
    assert r["verdict"] == "drena"


def test_s13_colapsa():
    """C3 fila 1.3 (onset colapso): 137 teleports, +15% duración, fracción sub-20 > 10% → COLAPSA."""
    r = _eval("s13")
    s = r["signals"]
    assert s["teleports"]["value"] == 137              # C3 fila 1.3
    assert s["duration_s"]["value"] == pytest.approx(292.0, abs=1.0)
    assert not s["teleports"]["ok"]                    # 137 > 50
    assert not s["duration_s"]["ok"]                   # 292 > 280
    # señal #3: fracción sub-20 supera el corte de 10% en ambas ventanas de pico.
    for w in s["dip"]["windows"].values():
        assert w["frac_sub20"] > ed.DIP_FRAC_MAX
    assert not s["dip"]["ok"]
    assert r["drains"] is False
    assert r["verdict"] == "colapsa"


def test_dip_acotado_a_ventana_de_pico():
    """El dip solo mira las ventanas de pico; el margen del fixture no entra al cómputo.

    Verificación estructural: las ventanas reportadas son exactamente las de pico (sin margen),
    la fracción se mide solo dentro de [begin, end), y con tráfico completo el denominador es 120
    intervalos por ventana (si el margen entrara, el denominador sería mayor)."""
    for tag in ("s11", "s13"):
        dip = _eval(tag)["signals"]["dip"]
        assert dip["windows"]["AM"]["begin"] == 25200
        assert dip["windows"]["AM"]["end"] == 32400
        assert dip["windows"]["PM"]["begin"] == 64800
        assert dip["windows"]["PM"]["end"] == 72000
        assert dip["frac_max"] == pytest.approx(0.10)
        assert dip["threshold_kmh"] == 20.0
        for w in dip["windows"].values():
            assert w["n_con_trafico"] == 120
