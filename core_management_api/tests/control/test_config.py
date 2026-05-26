"""
Tests for ``src.control.config.ControlSettings`` — extraction without
recalibration (CT-10.4.1 + CT-10.6.1).

These constants are normative (CT-10.4 cites SUMO calibration; CT-10.6 cites
Manual MTC peruano R.D. N.° 26-2024-MTC/18). TTH-10 extracts them to a
config layer for deployment-time override; the defaults must remain
identical to the previous hardcoded values until validation evidence
justifies recalibration.
"""
from src.control.application.adaptive_engine import AdaptiveEngine
from src.control.application.max_pressure import MaxPressureController
from src.control.application.mtc_constraints import MTCConstants, MTCRestrictionApplier
from src.control.application.webster import WebsterCalculator
from src.control.config import ControlSettings


def test_control_settings_defaults_match_previous_hardcoded_values_ct_10_4_1():
    """CT-10.4.1 + CT-10.6.1: ControlSettings() defaults are byte-identical
    to the legacy hardcoded constants."""
    s = ControlSettings()

    # CT-10.4: peak threshold (SUMO calibrated).
    assert s.adaptive.peak_threshold == 1500.0

    # Webster cycle bounds.
    assert s.webster.min_cycle == 30.0
    assert s.webster.max_cycle == 120.0

    # Max Pressure default cycle.
    assert s.max_pressure.default_cycle == 60.0


def test_mtc_constants_match_normative_values_ct_10_6_1():
    """CT-10.6.1: the five MTC constants persist their Manual MTC values."""
    s = ControlSettings()

    assert s.mtc.min_green == 7
    assert s.mtc.max_green == 60
    assert s.mtc.min_yellow == 3
    assert s.mtc.all_red == 2
    assert s.mtc.min_pedestrian == 7
    assert s.mtc.max_cycle == 120.0


def test_default_constructed_components_match_settings_defaults():
    """The legacy no-arg constructors and ControlSettings() defaults produce
    the same operating numbers — the engine instantiated from settings is
    behaviourally indistinguishable from one with hardcoded defaults."""
    s = ControlSettings()

    legacy_webster = WebsterCalculator()
    legacy_mp = MaxPressureController()
    legacy_mtc_applier = MTCRestrictionApplier()
    legacy_mtc_constants = MTCConstants()
    legacy_engine = AdaptiveEngine(
        webster=legacy_webster, max_pressure=legacy_mp, mtc=legacy_mtc_applier
    )

    assert legacy_webster.min_cycle == s.webster.min_cycle
    assert legacy_webster.max_cycle == s.webster.max_cycle
    assert legacy_mp.default_cycle == s.max_pressure.default_cycle
    assert legacy_mtc_applier.max_cycle == s.mtc.max_cycle
    assert legacy_mtc_constants.min_green == s.mtc.min_green
    assert legacy_mtc_constants.max_green == s.mtc.max_green
    assert legacy_mtc_constants.min_yellow == s.mtc.min_yellow
    assert legacy_mtc_constants.all_red == s.mtc.all_red
    assert legacy_mtc_constants.min_pedestrian == s.mtc.min_pedestrian
    assert legacy_engine.PEAK_THRESHOLD == s.adaptive.peak_threshold
