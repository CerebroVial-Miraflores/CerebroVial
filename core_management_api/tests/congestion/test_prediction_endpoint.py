"""Tests del endpoint GET /congestion/prediction (Fase 3 — predicción servida).

Cubren el contrato (30 pasos × 1660 aristas, niveles 0-4), el círculo ``node_order``
entrada→salida (la respuesta etiqueta las aristas con el MISMO orden con el que el modelo
fue entrenado), la resolución de ``t`` (param explícito vs derivado del feed vivo), el
invariante de coherencia (``t`` sale del step del feed, no del reloj de pared) y RBAC.

Usan el slice + checkpoint reales versionados en ``core_management_api/models/`` (no
dependen del ``.npz`` completo gitignored — esos artefactos SÍ están en el repo).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import src.congestion.presentation.api.routes as routes_module
from src.congestion.application.congestion_prediction_service import (
    CongestionPredictionService,
)
from src.congestion.infrastructure.gru_inference_adapter import load_gru_adapter
from src.congestion.infrastructure.prediction_day_source import PredictionDaySource

_MODELS = Path(__file__).resolve().parents[2] / "models"
_SLICE = _MODELS / "day_seed051_tensor.npz"
_CKPT = _MODELS / "miraflores_gru_baseline.pt"

T_DEMO = 1155  # 19:15


@pytest.fixture
def prediction_client(congestion_client):
    """congestion_client con el servicio de predicción inicializado (slice + ckpt reales)."""
    day_source = PredictionDaySource(slice_path=_SLICE)
    day_source.load()
    adapter = load_gru_adapter(_CKPT, device="cpu")
    routes_module.init_prediction_service(
        CongestionPredictionService(day_source, adapter)
    )
    congestion_client._day_source = day_source  # para asserts del círculo node_order
    try:
        yield congestion_client
    finally:
        routes_module._prediction_service_instance = None


def _seed_feed_at(session, ts: datetime, edge_id: str = "-1098384939") -> None:
    """Inserta un snapshot en waze_jams para que latest_timestamp() devuelva ts."""
    from src.congestion.infrastructure import WazeJamsRepo, WazeJamRow
    repo = WazeJamsRepo(session)
    repo.insert_one(
        WazeJamRow(
            event_uuid=f"evt-{ts.isoformat()}",
            snapshot_timestamp=ts,
            edge_id=edge_id,
            speed_mps=10.0,
            delay_seconds=5,
            congestion_level=2,
            jam_length_m=-1,
            road_type=0,
        )
    )
    session.commit()


def test_contrato_30_pasos_1660_aristas(prediction_client):
    """200, 1660 aristas, cada una 30 niveles, todos en {0..4}, base_timestep=t."""
    prediction_client.as_role("operator")
    r = prediction_client.get("/congestion/prediction", params={"t": T_DEMO})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["base_timestep"] == T_DEMO
    assert body["horizon"] == 30
    assert body["count"] == 1660
    assert len(body["edges"]) == 1660
    for e in body["edges"]:
        assert len(e["levels"]) == 30
        assert all(0 <= lv <= 4 for lv in e["levels"]), f"nivel fuera de 0-4 en {e['edge_id']}"


def test_source_date_expuesto(prediction_client):
    """El response trae ``source_date`` = la fecha-calendario del día-fuente (seed051 ↔ 2026-06-08).

    Gate 3a-bis: el frontend (modo ``pinned``) la usa para pedir la base observada coherente
    vía ``getSeries(source_date)``. Es deuda hermana del ``source`` hardcodeado (constante en
    el endpoint, coherente-por-convención con el calendario de siembra), no una fecha resuelta
    del slice — ver el comentario de ``_PREDICTION_SOURCE_DATE`` en routes.py.
    """
    prediction_client.as_role("operator")
    r = prediction_client.get("/congestion/prediction", params={"t": T_DEMO})
    assert r.status_code == 200, r.text
    assert r.json()["source_date"] == "2026-06-08"  # Pydantic serializa date → ISO 8601


def test_circulo_node_order_entrada_salida(prediction_client):
    """La arista i de la respuesta == node_order[i] (mismo orden que vio el modelo).

    Cierra el lazo entrada→salida: el slice guarda node_order verificado byte-idéntico al
    de training (stage-gate 1); acá se confirma que ESE orden etiqueta la respuesta. Sigue
    una arista conocida desde el slice hasta la respuesta.
    """
    prediction_client.as_role("operator")
    node_order = prediction_client._day_source.node_order
    r = prediction_client.get("/congestion/prediction", params={"t": T_DEMO})
    edges = r.json()["edges"]
    # Mismo orden, posición a posición (descarta mapeo cruzado silencioso).
    for i in (0, 1, 10, 829, 1659):
        assert edges[i]["edge_id"] == node_order[i], (
            f"desalineo node_order→respuesta en i={i}: "
            f"{edges[i]['edge_id']!r} != {node_order[i]!r}"
        )


def test_t_fallback_al_feed_no_reloj_de_pared(prediction_client):
    """Sin t explícito, base_timestep sale del max(snapshot_timestamp) del feed (minuto del día)."""
    # Feed anclado a las 19:15 de una fecha cualquiera (epoch-agnóstico: importa la hora).
    _seed_feed_at(prediction_client.session, datetime(2026, 6, 8, 19, 15, 0))
    prediction_client.as_role("operator")
    r = prediction_client.get("/congestion/prediction")  # sin t
    assert r.status_code == 200, r.text
    assert r.json()["base_timestep"] == T_DEMO  # 19*60+15 = 1155


def test_t_param_tiene_precedencia_sobre_feed(prediction_client):
    """Con t explícito se respeta ese t, ignorando el step del feed."""
    _seed_feed_at(prediction_client.session, datetime(2026, 6, 8, 19, 15, 0))  # feed en 1155
    prediction_client.as_role("operator")
    r = prediction_client.get("/congestion/prediction", params={"t": 600})  # 10:00
    assert r.status_code == 200, r.text
    assert r.json()["base_timestep"] == 600


def test_coherencia_feed_step_define_la_ventana(prediction_client):
    """Invariante: dado un step del feed, el corte t corresponde a ese step (mismo día-fuente).

    Anclamos el feed a un step conocido (10:00 → t=600) y verificamos que la predicción se
    arma sobre ESE t — el mismo que GET /congestion/state mostraría (max snapshot_timestamp),
    no datetime.now(). La fuente es seed051 en ambos lados.
    """
    _seed_feed_at(prediction_client.session, datetime(2026, 6, 8, 10, 0, 0))
    prediction_client.as_role("operator")
    r = prediction_client.get("/congestion/prediction")
    assert r.status_code == 200, r.text
    assert r.json()["base_timestep"] == 600
    assert r.json()["source"].startswith("seed051")


def test_feed_vacio_sin_t_da_503(prediction_client):
    """Sin t y sin feed (waze_jams vacío) → 503 con mensaje claro (no se inventa un t)."""
    prediction_client.as_role("operator")
    r = prediction_client.get("/congestion/prediction")
    assert r.status_code == 503
    assert "waze_jams" in r.json()["detail"]


def test_t_fuera_de_rango_da_422(prediction_client):
    """t < 29 o > 1439 → 422 (la ventana t-29..t debe caer en el día)."""
    prediction_client.as_role("operator")
    assert prediction_client.get("/congestion/prediction", params={"t": 10}).status_code == 422
    assert prediction_client.get("/congestion/prediction", params={"t": 2000}).status_code == 422


def test_rbac_rol_no_autorizado_da_403(prediction_client):
    """Un rol fuera de {operator, admin} recibe 403 (mismo guard que el resto de congestion/)."""
    prediction_client.as_role("manager")
    r = prediction_client.get("/congestion/prediction", params={"t": T_DEMO})
    assert r.status_code == 403


def test_servicio_no_inicializado_da_503(congestion_client):
    """Sin init_prediction_service (singleton None) → 503, no 500 (boot tolerante)."""
    routes_module._prediction_service_instance = None
    congestion_client.as_role("operator")
    r = congestion_client.get("/congestion/prediction", params={"t": T_DEMO})
    assert r.status_code == 503
