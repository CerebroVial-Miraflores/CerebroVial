"""Tests del mapeo del adaptador de replay SUMO → waze_jams (TTH-12 Fase 2).

Cubre la REGLA CRÍTICA del NaN (sin observación → nivel 0 + flujo libre, NO la
función), la derivación con observación (reusa ratio_to_jam_level), los centinelas
y la determinación del event_uuid. Inyecta arrays sintéticos para no depender del
Parquet real. Alinea con CT-12.9 (c, d parciales).
"""
import numpy as np

from src.congestion.infrastructure.sumo_replay_adapter import (
    DAY_EPOCH,
    SENTINEL_JAM_LENGTH_M,
    SENTINEL_ROAD_TYPE,
    SumoReplayAdapter,
    _event_uuid,
)


def _synthetic_adapter() -> SumoReplayAdapter:
    a = SumoReplayAdapter(day="synthetic")
    a._universe = ["e1", "e2"]
    a._freeflow = {"e1": 10.0, "e2": 20.0}
    a._edge_ids = ["e1", "e2", "e1", "e2"]
    a._timestep = np.array([0, 0, 60, 60])
    #              e1@0 obs   e2@0 NaN   e1@60 obs   e2@60 NaN
    a._speed = np.array([8.0, np.nan, 2.0, np.nan])
    a._speed_rel = np.array([0.8, np.nan, 0.2, np.nan])
    a._timeloss = np.array([1.4, 0.0, 30.7, 0.0])
    a._loaded = True
    return a


def test_nan_bucket_maps_to_free_flow_level_zero():
    a = _synthetic_adapter()
    r = a._row_at(1)  # e2@0, speedRelative NaN
    assert r.congestion_level == 0           # flujo libre, NO 4 (la función cruda daría 4)
    assert r.speed_mps == 20.0               # velocidad de flujo libre de e2
    assert r.delay_seconds == 0


def test_observed_bucket_uses_ratio_to_jam_level():
    a = _synthetic_adapter()
    r0 = a._row_at(0)   # sr=0.8 -> 0 ; speed=8.0 ; delay=round(1.4)=1
    assert (r0.congestion_level, r0.speed_mps, r0.delay_seconds) == (0, 8.0, 1)
    r2 = a._row_at(2)   # sr=0.2 -> 3 ; speed=2.0 ; delay=round(30.7)=31
    assert (r2.congestion_level, r2.speed_mps, r2.delay_seconds) == (3, 2.0, 31)


def test_derivation_matches_real_function_across_range():
    from cerebrovial_simulation.jam_level import ratio_to_jam_level
    a = _synthetic_adapter()
    for sr, expected in [(0.95, 0), (0.7, 1), (0.5, 2), (0.3, 3), (0.1, 4), (0.0, 5)]:
        lvl, _, _ = a._level_speed_delay(sr, 5.0, 0.0, "e1")
        assert lvl == expected == ratio_to_jam_level(sr)


def test_sentinels_documented():
    a = _synthetic_adapter()
    r = a._row_at(0)
    assert r.jam_length_m == SENTINEL_JAM_LENGTH_M == -1
    assert r.road_type == SENTINEL_ROAD_TYPE == 0


def test_event_uuid_is_deterministic():
    a = _synthetic_adapter()
    r_a = a._row_at(0)
    r_b = a._row_at(0)
    assert r_a.event_uuid == r_b.event_uuid          # mismo (edge, ts) -> mismo uuid
    assert a._row_at(0).event_uuid != a._row_at(2).event_uuid  # distinto ts -> distinto


def test_snapshot_timestamp_maps_timestep_to_epoch():
    a = _synthetic_adapter()
    assert a._row_at(0).snapshot_timestamp == DAY_EPOCH                       # ts=0
    assert _event_uuid("e1", DAY_EPOCH) == a._row_at(0).event_uuid
