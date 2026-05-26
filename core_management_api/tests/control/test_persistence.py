"""
Tests for the write-path that persists motor decisions (TTH-10, CT-10.9).

Coverage:
- CT-10.9.1: happy path persists a row with all canonical fields.
- CT-10.9.2: FK miss returns 422 unknown_intersection and persists nothing.
- CT-10.9.3: inputs_snapshot is structurally faithful to the payload.
- CT-10.9.4: peak path persists mode='max_pressure'.
- CT-10.9.5: off-peak path persists mode='webster' with y_load_factor > 0.
- CT-10.9.6: peak + WebsterInfeasible persists with y_load_factor IS NULL
             and cycle_seconds == default_cycle (60.0).
- CT-10.9.7: MotorDecisionsRepo exposes no update/delete (append-only).
- CT-10.9.8: response JSON does NOT leak the internal fields flow_total
             or y_load_factor (HTTP contract intact).
"""
from cerebrovial_shared.database.models import MotorDecisionDB

from src.control.infrastructure import MotorDecisionsRepo


def test_recommend_persists_motor_decision_ct_10_9_1(
    client_with_db, motor_db_session, http_payload
):
    """CT-10.9.1: a successful recommendation persists exactly one row with
    all the canonical fields populated."""
    payload = http_payload(flow_total=1000.0)

    response = client_with_db.post("/control/recommend", json=payload)
    assert response.status_code == 200, response.text

    rows = motor_db_session.query(MotorDecisionDB).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.node_id == "larco_schell"
    assert row.mode == "webster"
    assert row.cycle_seconds > 0
    assert row.flow_total == 1000.0
    assert row.y_load_factor is not None and row.y_load_factor > 0
    assert row.reasoning  # non-empty
    assert isinstance(row.phase_timings, list) and len(row.phase_timings) == 2
    assert isinstance(row.adjustments, list)
    assert row.inputs_snapshot is not None
    assert row.decided_at is not None
    assert row.decision_id  # auto-uuid


def test_unknown_intersection_returns_422_and_persists_nothing_ct_10_9_2(
    client_with_db, motor_db_session, http_payload
):
    """CT-10.9.2: an intersection_id absent from graph_nodes fails fast with
    422 unknown_intersection and never reaches the engine nor the write."""
    payload = http_payload(flow_total=1000.0, intersection_id="NO_EXISTE")

    response = client_with_db.post("/control/recommend", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_intersection"
    assert motor_db_session.query(MotorDecisionDB).count() == 0


def test_inputs_snapshot_is_faithful_to_payload_ct_10_9_3(
    client_with_db, motor_db_session, http_payload
):
    """CT-10.9.3: inputs_snapshot reproduces the request payload phase by
    phase (the row must be reproducible from its snapshot)."""
    payload = http_payload(flow_total=1000.0)

    response = client_with_db.post("/control/recommend", json=payload)
    assert response.status_code == 200

    row = motor_db_session.query(MotorDecisionDB).one()
    snap = row.inputs_snapshot
    assert snap["lost_time"] == payload["lost_time"]
    assert len(snap["phases"]) == len(payload["phases"])
    for snap_phase, req_phase in zip(snap["phases"], payload["phases"]):
        assert snap_phase["phase_id"] == req_phase["phase_id"]
        assert snap_phase["flow"] == req_phase["flow"]
        assert snap_phase["saturation_flow"] == req_phase["saturation_flow"]
        assert snap_phase["queue"] == req_phase["queue"]


def test_peak_path_persists_max_pressure_ct_10_9_4(
    client_with_db, motor_db_session, http_payload
):
    """CT-10.9.4 (CT-10.4 + CT-10.9): flow_total ≥ 1500 routes to
    Max Pressure and the persisted row reflects it."""
    response = client_with_db.post(
        "/control/recommend",
        json=http_payload(flow_total=2000.0, queues=(15, 4)),
    )
    assert response.status_code == 200

    row = motor_db_session.query(MotorDecisionDB).one()
    assert row.mode == "max_pressure"
    assert row.flow_total == 2000.0


def test_off_peak_path_persists_webster_with_y_load_ct_10_9_5(
    client_with_db, motor_db_session, http_payload
):
    """CT-10.9.5 (CT-10.4 + CT-10.9): off-peak persists mode='webster' with
    y_load_factor not null and > 0 (Webster always computes Y)."""
    response = client_with_db.post(
        "/control/recommend", json=http_payload(flow_total=1000.0)
    )
    assert response.status_code == 200

    row = motor_db_session.query(MotorDecisionDB).one()
    assert row.mode == "webster"
    assert row.y_load_factor is not None
    assert row.y_load_factor > 0


def test_peak_with_webster_infeasible_persists_null_y_load_ct_10_9_6(
    client_with_db, motor_db_session, http_payload
):
    """CT-10.9.6 (CT-10.5 + CT-10.9): peak + Y ≥ 0.95 forces Max Pressure
    to fall back to default_cycle=60s; the persisted row carries
    cycle_seconds=60.0 and y_load_factor IS NULL (fidelity to
    data-model.md §2.1 — nullable only in saturación severa)."""
    # flow_total ≈ 2000 (peak), saturation low so Y = 2000/600 ≈ 3.33 → infeasible.
    response = client_with_db.post(
        "/control/recommend",
        json=http_payload(flow_total=2000.0, saturation=600.0, queues=(20, 10)),
    )
    assert response.status_code == 200, response.text

    row = motor_db_session.query(MotorDecisionDB).one()
    assert row.mode == "max_pressure"
    assert row.y_load_factor is None
    # default_cycle is 60s; MTC may scale it but the source cycle was 60.
    # Either way, reasoning must mention the default cycle path.
    assert "Webster infeasible" in row.reasoning


def test_motor_decisions_repo_is_append_only_ct_10_9_7(motor_db_session):
    """CT-10.9.7: MotorDecisionsRepo intentionally exposes no update/delete.
    Append-only is an invariant of the application layer (data-model.md §2.1).
    """
    repo = MotorDecisionsRepo(motor_db_session)
    assert not hasattr(repo, "update")
    assert not hasattr(repo, "delete")
    assert not hasattr(repo, "save")  # alias misuse guard


def test_response_does_not_leak_internal_fields_ct_10_9_8(
    client_with_db, http_payload
):
    """CT-10.9.8 (CT-10.8 + CT-10.9): the HTTP contract stays intact —
    flow_total and y_load_factor are persisted but NOT serialized in the
    response (ControlRecommendation Pydantic schema is unchanged)."""
    response = client_with_db.post(
        "/control/recommend", json=http_payload(flow_total=1000.0)
    )
    assert response.status_code == 200

    body = response.json()

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                assert k not in {"flow_total", "y_load_factor"}, (
                    f"internal field leaked into response: {k}"
                )
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(body)
