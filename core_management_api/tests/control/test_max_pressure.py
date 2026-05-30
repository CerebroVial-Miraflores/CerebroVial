import math

import pytest

from src.control.application.max_pressure import (
    DownstreamLinkDC,
    MaxPressureController,
    MaxPressureDecision,
)


def test_decide_picks_phase_with_largest_pressure():
    controller = MaxPressureController()

    decision = controller.decide(
        queues={"NS": 2, "EW": 12},
        saturations={"NS": 1800, "EW": 1800},
        lost_time=8.0,
    )

    assert isinstance(decision, MaxPressureDecision)
    assert decision.next_phase == "EW"
    assert decision.cycle_seconds == controller.default_cycle
    assert decision.green_by_phase["EW"] > decision.green_by_phase["NS"]
    assert math.isclose(
        sum(decision.green_by_phase.values()),
        decision.cycle_seconds - 8.0,
        rel_tol=1e-9,
    )


def test_decide_falls_back_to_round_robin_when_queues_empty():
    controller = MaxPressureController()

    decision = controller.decide(
        queues={"WE": 0, "NS": 0, "AB": 0},
        saturations={"WE": 1800, "NS": 1800, "AB": 1800},
        lost_time=4.0,
    )

    # Alphabetical: AB < NS < WE → fallback selects "AB".
    assert decision.next_phase == "AB"
    # Greens are evenly shared.
    expected = (decision.cycle_seconds - 4.0) / 3
    for green in decision.green_by_phase.values():
        assert math.isclose(green, expected, rel_tol=1e-9)


def test_decide_breaks_pressure_ties_alphabetically():
    controller = MaxPressureController()

    decision = controller.decide(
        queues={"BB": 10, "AA": 10},
        saturations={"BB": 1800, "AA": 1800},
        lost_time=4.0,
    )

    assert decision.next_phase == "AA"


def test_decide_honours_custom_cycle():
    controller = MaxPressureController()

    decision = controller.decide(
        queues={"NS": 5, "EW": 3},
        saturations={"NS": 1800, "EW": 1800},
        lost_time=8.0,
        cycle_seconds=90.0,
    )

    assert decision.cycle_seconds == 90.0
    assert math.isclose(
        sum(decision.green_by_phase.values()),
        82.0,
        rel_tol=1e-9,
    )


def test_decide_rejects_mismatched_phase_ids():
    controller = MaxPressureController()

    with pytest.raises(ValueError):
        controller.decide(
            queues={"NS": 1},
            saturations={"EW": 1800},
            lost_time=4.0,
        )


# ---------------------------------------------------------------------------
# Max Pressure de red (término downstream) — retrocompatibilidad bit-a-bit.
#
# GOLDEN: comportamiento per-node actual (P = s·q) congelado en valores
# explícitos. Es la red de seguridad de IE05: si estos números cambian, el
# brazo per-node (downstream-off) deja de reproducir el +15.7% histórico.
# Se evalúa con la firma ACTUAL de decide() (sin downstream) para que sea el
# primer verde antes de tocar la lógica, y luego con downstreams=None para
# probar la equivalencia bit-a-bit tras la extensión.
# ---------------------------------------------------------------------------

# Cada caso: (queues, saturations, lost_time, cycle_seconds|None,
#             next_phase esperado, green_by_phase esperado).
_GOLDEN_PER_NODE = [
    # Asimétrico, split limpio (default cycle 60): usable=52, ratio 1:3.
    ({"NS": 1, "EW": 3}, {"NS": 1800, "EW": 1800}, 8.0, None,
     "EW", {"NS": 13.0, "EW": 39.0}),
    # Round-robin: todas las colas en 0 → ordered[0]="A", verde igual.
    ({"A": 0, "B": 0, "C": 0}, {"A": 1800, "B": 1800, "C": 1800}, 6.0, None,
     "A", {"A": 18.0, "B": 18.0, "C": 18.0}),
    # Empate de presión → tie-break alfabético "AA"; verde 50/50.
    ({"BB": 5, "AA": 5}, {"BB": 1800, "AA": 1800}, 4.0, None,
     "AA", {"AA": 28.0, "BB": 28.0}),
    # Empate con ciclo custom 90: ordered[0]="EW", verde 50/50 (usable=82).
    ({"NS": 2, "EW": 2}, {"NS": 1800, "EW": 1800}, 8.0, 90.0,
     "EW", {"NS": 41.0, "EW": 41.0}),
    # Asimétrico con ciclo custom 100: usable=92, ratio 1:3.
    ({"NS": 1, "EW": 3}, {"NS": 1800, "EW": 1800}, 8.0, 100.0,
     "EW", {"NS": 23.0, "EW": 69.0}),
]


def _assert_matches_golden(decision, exp_next, exp_green, exp_cycle):
    assert decision.next_phase == exp_next
    assert math.isclose(decision.cycle_seconds, exp_cycle, rel_tol=1e-9)
    assert set(decision.green_by_phase) == set(exp_green)
    for phase, green in exp_green.items():
        assert math.isclose(decision.green_by_phase[phase], green, rel_tol=1e-9), (
            f"green[{phase}]={decision.green_by_phase[phase]} != {green}"
        )


@pytest.mark.parametrize(
    "queues, saturations, lost_time, cycle, exp_next, exp_green", _GOLDEN_PER_NODE
)
def test_no_downstream_matches_current_behavior(
    queues, saturations, lost_time, cycle, exp_next, exp_green
):
    """Compuerta de retrocompat: la firma actual (sin downstream) produce los
    valores golden per-node. Verde antes y después de la extensión."""
    controller = MaxPressureController()
    exp_cycle = cycle if cycle is not None else controller.default_cycle

    decision = controller.decide(
        queues=queues,
        saturations=saturations,
        lost_time=lost_time,
        cycle_seconds=cycle,
    )
    _assert_matches_golden(decision, exp_next, exp_green, exp_cycle)


@pytest.mark.parametrize(
    "queues, saturations, lost_time, cycle, exp_next, exp_green", _GOLDEN_PER_NODE
)
def test_downstream_none_and_dict_of_none_match_no_downstream(
    queues, saturations, lost_time, cycle, exp_next, exp_green
):
    """downstreams=None y un dict de puros None producen el resultado golden
    bit-a-bit (no hay término downstream que restar)."""
    controller = MaxPressureController()
    exp_cycle = cycle if cycle is not None else controller.default_cycle

    base = dict(queues=queues, saturations=saturations, lost_time=lost_time, cycle_seconds=cycle)

    d_none = controller.decide(**base, downstreams=None)
    d_dict_none = controller.decide(**base, downstreams={p: None for p in queues})

    for decision in (d_none, d_dict_none):
        _assert_matches_golden(decision, exp_next, exp_green, exp_cycle)


def test_downstream_lowers_phase_pressure_and_flips_winner():
    """Escenario Benavides→Schell: el link interno lleno baja la presión de la
    fase pasante → menos verde y le cede el next_phase a la transversal."""
    controller = MaxPressureController()
    base = dict(
        queues={"NS": 10, "EW": 8},
        saturations={"NS": 1800, "EW": 1800},
        lost_time=8.0,
    )

    without = controller.decide(**base)
    assert without.next_phase == "NS"  # 18000 > 14400

    # NS descarga a un link casi-lleno (queue 9) → net_NS = 10-9 = 1.
    with_down = controller.decide(
        **base, downstreams={"NS": [DownstreamLinkDC(queue=9, turn_ratio=1.0)]}
    )
    assert with_down.next_phase == "EW"  # 1800 < 14400 → se invierte
    assert with_down.green_by_phase["NS"] < without.green_by_phase["NS"]
    # El ciclo no cambia; el verde se reparte por presión positiva.
    assert math.isclose(
        sum(with_down.green_by_phase.values()),
        with_down.cycle_seconds - 8.0,
        rel_tol=1e-9,
    )


def test_downstream_turn_ratio_scales_the_subtracted_term():
    """τ escala el término downstream: con τ=0.5 se resta la mitad y el ganador
    cambia respecto a τ=1.0 (net = q − τ·x_down)."""
    controller = MaxPressureController()
    base = dict(
        queues={"NS": 10, "EW": 4},
        saturations={"NS": 1800, "EW": 1800},
        lost_time=8.0,
    )

    # τ=1.0: net_NS = 10-8 = 2 (3600) < EW=4 (7200) → next EW.
    full = controller.decide(
        **base, downstreams={"NS": [DownstreamLinkDC(queue=8, turn_ratio=1.0)]}
    )
    assert full.next_phase == "EW"

    # τ=0.5: net_NS = 10-4 = 6 (10800) > EW=4 (7200) → next NS.
    half = controller.decide(
        **base, downstreams={"NS": [DownstreamLinkDC(queue=8, turn_ratio=0.5)]}
    )
    assert half.next_phase == "NS"


def test_all_negative_pressures_fall_back_to_round_robin():
    """Si todos los downstream superan a las colas locales (todas las presiones
    ≤ 0) → round-robin determinista (liveness), no verde forzado a un net<0."""
    controller = MaxPressureController()
    decision = controller.decide(
        queues={"NS": 2, "EW": 3},
        saturations={"NS": 1800, "EW": 1800},
        lost_time=8.0,
        downstreams={
            "NS": [DownstreamLinkDC(queue=5)],
            "EW": [DownstreamLinkDC(queue=5)],
        },
    )
    # ordered[0] = "EW" (E < N).
    assert decision.next_phase == "EW"
    expected = (decision.cycle_seconds - 8.0) / 2
    for green in decision.green_by_phase.values():
        assert math.isclose(green, expected, rel_tol=1e-9)


def test_one_negative_one_positive_split():
    """Una fase con net ≤ 0 recibe peso 0 (verde 0 crudo → MTC lo subirá a
    min_green); la positiva se lleva todo el usable_green."""
    controller = MaxPressureController()
    decision = controller.decide(
        queues={"NS": 10, "EW": 2},
        saturations={"NS": 1800, "EW": 1800},
        lost_time=8.0,
        downstreams={"EW": [DownstreamLinkDC(queue=5)]},  # net_EW = 2-5 = -3
    )
    assert decision.next_phase == "NS"
    assert math.isclose(decision.green_by_phase["EW"], 0.0, abs_tol=1e-9)
    assert math.isclose(
        decision.green_by_phase["NS"], decision.cycle_seconds - 8.0, rel_tol=1e-9
    )


@pytest.mark.parametrize(
    "bad_link",
    [
        DownstreamLinkDC(queue=5, turn_ratio=0.0),   # τ no positivo
        DownstreamLinkDC(queue=5, turn_ratio=-0.5),
        DownstreamLinkDC(queue=-1, turn_ratio=1.0),  # cola negativa
    ],
)
def test_downstream_validation_rejects_bad_links(bad_link):
    controller = MaxPressureController()
    with pytest.raises(ValueError):
        controller.decide(
            queues={"NS": 5, "EW": 3},
            saturations={"NS": 1800, "EW": 1800},
            lost_time=8.0,
            downstreams={"NS": [bad_link]},
        )
