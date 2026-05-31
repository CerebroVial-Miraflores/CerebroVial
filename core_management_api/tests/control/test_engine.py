"""
End-to-end tests for ``AdaptiveEngine`` and the ``/control/recommend`` endpoint.

The suite avoids importing ``src.main`` (which pulls torch via the prediction
module). Instead, ``conftest.app`` builds a minimal FastAPI app that only
mounts the control router.
"""


def test_recommend_off_peak_routes_to_webster(engine, build_state):
    state = build_state(flow_total=1000.0)

    rec = engine.recommend(state)

    assert rec.mode == "webster"
    assert rec.next_phase is None
    assert {t.phase_id for t in rec.phase_timings} == {"NS", "EW"}
    # MTC guarantees yellow + all_red appear on every phase.
    assert all(t.yellow == 3.0 and t.all_red == 2.0 for t in rec.phase_timings)
    # Cycle stays within the Webster bounds plus MTC ceiling (max_cycle = 120).
    assert 30.0 <= rec.cycle_seconds <= 120.0 + 1e-6
    assert "Off-peak" in rec.reasoning


def test_recommend_peak_routes_to_max_pressure(engine, build_state):
    state = build_state(flow_total=2000.0, queues=(15, 4))

    rec = engine.recommend(state)

    assert rec.mode == "max_pressure"
    # NS has the dominant queue → it should be the next phase.
    assert rec.next_phase == "NS"
    assert "Peak" in rec.reasoning
    assert all(t.green >= 7.0 for t in rec.phase_timings)


def test_recommend_peak_downstream_flips_next_phase(engine):
    """Max Pressure de red end-to-end: NS tiene la cola dominante, pero su link
    downstream casi-lleno baja su presión por debajo de EW → next_phase = EW.
    Recorre recommend() + MTC (verdes ≥ min_green)."""
    from src.control.application.adaptive_engine import (
        IntersectionStateDC,
        PhaseFlowDC,
    )
    from src.control.application.max_pressure import DownstreamLinkDC

    state = IntersectionStateDC(
        intersection_id="I-001",
        timestamp="2026-05-09T10:00:00Z",
        lost_time=10.0,
        phases=[
            PhaseFlowDC(
                phase_id="NS",
                flow=1600.0,
                saturation_flow=1800.0,
                queue=15,
                downstream=[DownstreamLinkDC(queue=14, turn_ratio=1.0)],  # net = 1
            ),
            PhaseFlowDC(phase_id="EW", flow=400.0, saturation_flow=1800.0, queue=4),
        ],
    )

    rec = engine.recommend(state)

    assert rec.mode == "max_pressure"
    assert rec.next_phase == "EW"  # 1800 < 7200 → la transversal gana
    assert all(t.green >= 7.0 for t in rec.phase_timings)  # MTC piso intacto


def test_recommend_at_threshold_uses_max_pressure(engine, build_state):
    # PEAK_THRESHOLD is exclusive on Webster (< threshold). Equal goes to MP.
    state = build_state(flow_total=engine.PEAK_THRESHOLD)

    rec = engine.recommend(state)

    assert rec.mode == "max_pressure"


def test_recommend_off_peak_oversaturated_propagates_webster_infeasible(engine, build_state):
    from src.control.application.webster import WebsterInfeasible

    # Off-peak (flow_total < 1500) but ratios near 1 → Y >= 0.95.
    state = build_state(
        flow_total=1400.0,
        saturation=750.0,  # ratios ≈ 770/750 + 630/750 = 1.87 → infeasible.
    )

    import pytest

    with pytest.raises(WebsterInfeasible):
        engine.recommend(state)


def test_endpoint_off_peak_returns_webster_recommendation(client_with_db, http_payload):
    response = client_with_db.post(
        "/control/recommend", json=http_payload(flow_total=1000.0)
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data"]["mode"] == "webster"
    assert body["data"]["next_phase"] is None
    assert {p["phase_id"] for p in body["data"]["phase_timings"]} == {"NS", "EW"}


def test_endpoint_peak_returns_max_pressure_recommendation(client_with_db, http_payload):
    response = client_with_db.post(
        "/control/recommend",
        json=http_payload(flow_total=2000.0, queues=(15, 4)),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data"]["mode"] == "max_pressure"
    assert body["data"]["next_phase"] == "NS"


def test_endpoint_accepts_downstream_and_flips_next_phase(client_with_db):
    """El schema HTTP acepta ``downstream`` por fase; el link interno lleno baja
    la presión de NS y EW gana el next_phase (Max Pressure de red end-to-end)."""
    payload = {
        "intersection_id": "larco_schell",
        "timestamp": "2026-05-09T10:00:00Z",
        "lost_time": 10.0,
        "phases": [
            {
                "phase_id": "NS",
                "flow": 1600.0,
                "saturation_flow": 1800.0,
                "queue": 15,
                "downstream": [{"queue": 14, "turn_ratio": 1.0}],
            },
            {"phase_id": "EW", "flow": 400.0, "saturation_flow": 1800.0, "queue": 4},
        ],
    }

    response = client_with_db.post("/control/recommend", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data"]["mode"] == "max_pressure"
    assert body["data"]["next_phase"] == "EW"


def test_endpoint_rejects_non_positive_saturation_flow(client_with_db):
    payload = {
        "intersection_id": "larco_schell",
        "timestamp": "2026-05-09T10:00:00Z",
        "lost_time": 8.0,
        "phases": [
            {"phase_id": "NS", "flow": 500, "saturation_flow": 0, "queue": 0},
        ],
    }

    response = client_with_db.post("/control/recommend", json=payload)

    assert response.status_code == 422


def test_endpoint_returns_422_on_oversaturated_off_peak(client_with_db, http_payload, motor_db_session):
    response = client_with_db.post(
        "/control/recommend",
        json=http_payload(flow_total=1400.0, saturation=750.0),
    )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["code"] == "webster_infeasible"
    # No row persisted on the 422 path (write-path runs only when the engine
    # produces a valid recommendation). Verifying the invariant explicitly.
    from cerebrovial_shared.database.models import MotorDecisionDB
    assert motor_db_session.query(MotorDecisionDB).count() == 0


def test_endpoint_returns_503_when_engine_not_initialized():
    """If init_engine is not called, get_engine raises 503. We replicate the
    state by mounting the router on a fresh app without calling init_engine.

    Depends ordering matters: get_engine is declared before get_db in the
    handler, so the 503 fires before FastAPI tries to resolve the DB session.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.control.presentation.api import routes as control_routes

    saved = control_routes._engine_instance
    control_routes._engine_instance = None
    try:
        app = FastAPI()
        app.include_router(control_routes.router)
        with TestClient(app) as ephemeral:
            response = ephemeral.post(
                "/control/recommend",
                json={
                    "intersection_id": "larco_schell",
                    "timestamp": "2026-05-09T10:00:00Z",
                    "phases": [
                        {"phase_id": "NS", "flow": 100, "saturation_flow": 1800},
                    ],
                },
            )
        assert response.status_code == 503
    finally:
        control_routes._engine_instance = saved
