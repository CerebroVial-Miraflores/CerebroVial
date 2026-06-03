"""Split compartido seed→fold del track STGNN (baseline GRU / STGNN), universo v2.

FUENTE ÚNICA DE VERDAD del split. El baseline GRU y el STGNN importan este módulo
para evaluar sobre EXACTAMENTE el mismo train/val/test, de modo que la comparación
aísle la contribución de la vecindad del grafo (D-011).

El dataset v2 tiene 60 días, uno por seed entera 42..101 (eje 0 del tensor;
día N ↔ seed 42+N).

Split **estratificado por régimen de congestión** (B4, 2026-06-03). El gate de
drenaje D-014 (``calibracion/DRENAJE_GATE_60D_RESULTS.md``) caracterizó el dataset v2
como un **gradiente de congestión de pico PM**, con 12/60 días de **congestión severa**
(cola superior de la distribución, NO días rotos — señal necesaria para un predictor
de demora). El criterio de partición es: **los 12 severos se distribuyen a conciencia
entre los tres folds, cubriendo el gradiente de severidad**, NO al azar.

Por qué importa (gate D-014 §"Consecuencia para B4"): si los severos caen todos en
test, el modelo se entrena ciego a la congestión severa; si caen todos en train, se
evalúa sin ella (métrica engañosamente buena). El split v1 forzaba seed-081 a test
como "el día extremo"; en v2 el extremo es otro (seed-085, 92 teleports / PM 25.8%)
y son **12, no 1** — el ancla de un solo día se reemplaza por **cobertura de régimen**.

Severidad por seed (del gate D-014; señal vinculante = dip PM, secundaria = teleports):
  - DUROS (las 3 señales del gate): 55 (PM 27.5%/70 tel), 62 (26.7%/80), 85 (25.8%/92),
    83 (17.5%/60). Los días genuinamente más severos.
  - MEDIOS (dip PM claramente >10%): 63 (15.8%), 58 (15.0%), 71 (15.0%), 97 (14.2%),
    60 (13.3%), 99 (13.3%).
  - MARGINALES (apenas sobre el corte 10%): 53 (12.5%), 90 (10.8%).

Reparto (cada fold ve el espectro completo; los 4 duros NO se concentran en un fold;
test sobre-representa el régimen severo —es donde se mide D-011— e incluye el peor día):
  - test  (3 severos): 85 (duro/peor por teleports) + 71 (medio) + 90 (marginal).
  - val   (2 severos): 62 (duro/2º) + 58 (medio).
  - train (7 severos): 55 (mayor dip PM) + 83 (duro) + 63/97/60/99 (medios) + 53 (marginal).

Proporciones: 42 train / 8 val / 10 test. Severos por fold: 7 / 2 / 3 (16.7% / 25% / 30%;
test sobre-representa el régimen severo a propósito).
"""
from __future__ import annotations

# --- Régimen de congestión (fuente: DRENAJE_GATE_60D_RESULTS.md, gate D-014 v2) -----
# Los 12 días de congestión severa del dataset v2 (cola superior del gradiente PM).
SEVERE_SEEDS: list[int] = [53, 55, 58, 60, 62, 63, 71, 83, 85, 90, 97, 99]
# Colapsos "duros" (disparan las 3 señales del gate): los más severos. Se reparten
# para que ningún fold los concentre (invariante de estratificación).
SEVERE_DUROS: list[int] = [55, 62, 83, 85]

# --- Split estratificado explícito (seeds enteras 42..101) -------------------------
# 10 días: 3 severos (85 duro/peor + 71 medio + 90 marginal) + 7 moderados/suaves.
TEST_SEEDS: list[int] = [45, 51, 69, 71, 78, 85, 88, 90, 93, 100]
# 8 días: 2 severos (62 duro + 58 medio) + 6 moderados/suaves.
VAL_SEEDS: list[int] = [58, 62, 66, 75, 84, 94, 96, 101]
# 42 días restantes: incluye 7 severos (55, 83, 63, 97, 60, 99, 53).
TRAIN_SEEDS: list[int] = sorted(
    set(range(42, 102)) - set(TEST_SEEDS) - set(VAL_SEEDS)
)

# Rango completo del dataset (60 días, seeds 42..101).
ALL_SEEDS: list[int] = list(range(42, 102))


def _validate() -> None:
    """Invariantes del split (se ejecuta al importar; falla rápido si se rompe).

    Estructurales (suma/solapamiento/proporciones) + **estratificación por régimen**:
    los 12 severos repartidos sin pérdida, cada fold ve congestión severa, y los duros
    NO se concentran en un solo fold. Si alguien reordena los folds sin respetar la
    estratificación, esto rebota.
    """
    train, val, test = set(TRAIN_SEEDS), set(VAL_SEEDS), set(TEST_SEEDS)
    # --- Estructurales ---
    assert not (train & val), f"train∩val no vacío: {sorted(train & val)}"
    assert not (train & test), f"train∩test no vacío: {sorted(train & test)}"
    assert not (val & test), f"val∩test no vacío: {sorted(val & test)}"
    assert train | val | test == set(ALL_SEEDS), "la unión de folds != seeds 42..101"
    assert len(train) + len(val) + len(test) == 60, "los folds no suman 60 (¿solapamiento?)"
    assert (len(train), len(val), len(test)) == (42, 8, 10), (
        f"proporciones esperadas 42/8/10, got "
        f"{len(train)}/{len(val)}/{len(test)}"
    )
    # --- Estratificación por régimen de congestión (B4) ---
    severe = set(SEVERE_SEEDS)
    assert severe <= set(ALL_SEEDS), "SEVERE_SEEDS fuera del rango 42..101"
    # 1) Los 12 severos se reparten sin pérdida (cada uno en exactamente un fold).
    assert severe <= (train | val | test), "algún severo no está en ningún fold"
    # 2) Cada fold ve congestión severa (ninguno queda ciego al régimen).
    assert severe & train, "train sin ningún día severo (se entrenaría ciego al régimen)"
    assert severe & val, "val sin ningún día severo"
    assert severe & test, "test sin ningún día severo (se evaluaría sobre el caso fácil)"
    # 3) Test contiene varios severos (cubre el régimen que decide D-011).
    assert len(severe & test) >= 3, (
        f"test debe contener ≥3 severos (cobertura de régimen), got {sorted(severe & test)}"
    )
    # 4) Los duros NO se concentran: cada fold ve ≥1 duro (gradiente repartido).
    duros = set(SEVERE_DUROS)
    assert duros & train, "ningún duro en train"
    assert duros & val, "ningún duro en val"
    assert duros & test, "ningún duro en test (test no evaluaría el caso más severo)"


_validate()


def get_split() -> tuple[list[int], list[int], list[int]]:
    """Devuelve ``(train_seeds, val_seeds, test_seeds)``, cada uno ordenado.

    Determinista: copias nuevas de las listas módulo-nivel en cada llamada (dos
    llamadas → mismo split, sin estado mutable compartido).
    """
    return sorted(TRAIN_SEEDS), sorted(VAL_SEEDS), sorted(TEST_SEEDS)
