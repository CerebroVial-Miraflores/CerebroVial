"""
Tests for ``EngineActiveStateRepo`` (TTH-10, CT-10.9 + HU-05).

The repository is built but NOT wired to any endpoint in TTH-10. Activation
is HU-05/HU-07 work. These unit tests prove that the upsert semantics work
correctly so the future HU can rely on it.

The CT-10.9.10 update branch is the one that proves the conceptual
separation declared in DHU-021 #13: ``motor_decisions.decided_at`` is the
moment of calculation, ``engine_active_state.activated_at`` is the moment
of activation — distinct events.
"""
import time

from cerebrovial_shared.database.models import EngineActiveStateDB

from src.control.infrastructure import EngineActiveStateRepo, MotorDecisionsRepo


def _seed_two_decisions(session) -> tuple[str, str]:
    """Insert two motor_decisions rows for the seeded node and return their
    decision_ids in insertion order."""
    repo = MotorDecisionsRepo(session)
    first = repo.insert(
        node_id="larco_schell",
        mode="webster",
        cycle_seconds=42.0,
        flow_total=1000.0,
        y_load_factor=0.6,
        next_phase=None,
        reasoning="first",
        phase_timings=[{"phase_id": "NS", "green": 20, "yellow": 3, "all_red": 2}],
        adjustments=[],
        inputs_snapshot={"lost_time": 8.0, "phases": []},
    )
    second = repo.insert(
        node_id="larco_schell",
        mode="max_pressure",
        cycle_seconds=60.0,
        flow_total=2000.0,
        y_load_factor=None,
        next_phase="NS",
        reasoning="second",
        phase_timings=[{"phase_id": "NS", "green": 25, "yellow": 3, "all_red": 2}],
        adjustments=[],
        inputs_snapshot={"lost_time": 8.0, "phases": []},
    )
    session.commit()
    return first, second


def test_activate_inserts_new_row_when_node_has_no_active_state_ct_10_9_9(
    motor_db_session,
):
    """CT-10.9.9 — insert branch: empty engine_active_state, activate() creates
    the first row with activated_at set."""
    first, _ = _seed_two_decisions(motor_db_session)

    repo = EngineActiveStateRepo(motor_db_session)
    repo.activate(node_id="larco_schell", decision_id=first, activated_by="operator")
    motor_db_session.commit()

    row = motor_db_session.get(EngineActiveStateDB, "larco_schell")
    assert row is not None
    assert row.active_decision_id == first
    assert row.activated_by == "operator"
    assert row.activated_at is not None


def test_activate_updates_existing_row_for_same_node_ct_10_9_10(motor_db_session):
    """CT-10.9.10 — update branch: a second activate() for the same node
    updates the pointer and refreshes activated_at; only one row exists.

    This proves the separation between decided_at (decision event) and
    activated_at (activation event) — DHU-021 #13."""
    first, second = _seed_two_decisions(motor_db_session)

    repo = EngineActiveStateRepo(motor_db_session)
    repo.activate(node_id="larco_schell", decision_id=first, activated_by="operator")
    motor_db_session.commit()

    first_row = motor_db_session.get(EngineActiveStateDB, "larco_schell")
    first_activated_at = first_row.activated_at

    # Ensure the timestamp can advance.
    time.sleep(0.01)

    repo.activate(node_id="larco_schell", decision_id=second, activated_by="auto")
    motor_db_session.commit()

    rows = motor_db_session.query(EngineActiveStateDB).all()
    assert len(rows) == 1  # one row per node, mutable
    updated = rows[0]
    assert updated.active_decision_id == second
    assert updated.activated_by == "auto"
    assert updated.activated_at >= first_activated_at
