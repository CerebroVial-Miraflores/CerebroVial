"""Tests de la interfaz de feed de congestión (CT-12.3 / CT-12.9 d parcial)."""
import numpy as np

from src.congestion.application.feed import CongestionFeed, EdgeCongestion
from src.congestion.infrastructure.sumo_replay_adapter import SumoReplayAdapter


def _synthetic_adapter() -> SumoReplayAdapter:
    a = SumoReplayAdapter(day="synthetic")
    a._universe = ["e1", "e2"]
    a._freeflow = {"e1": 10.0, "e2": 20.0}
    a._edge_ids = ["e1", "e2", "e1", "e2"]
    a._timestep = np.array([0, 0, 60, 60])
    a._speed = np.array([8.0, np.nan, 2.0, np.nan])
    a._speed_rel = np.array([0.8, np.nan, 0.2, np.nan])
    a._timeloss = np.array([1.4, 0.0, 30.7, 0.0])
    a._loaded = True
    return a


def test_adapter_satisfies_congestion_feed_protocol():
    assert isinstance(_synthetic_adapter(), CongestionFeed)


def test_timesteps_ascending():
    assert _synthetic_adapter().timesteps() == [0, 60]


def test_levels_at_returns_edge_congestion_by_edge():
    a = _synthetic_adapter()
    out = {ec.edge_id: ec for ec in a.levels_at(60)}
    assert set(out) == {"e1", "e2"}
    assert all(isinstance(v, EdgeCongestion) for v in out.values())
    assert out["e1"].congestion_level == 3       # sr=0.2 observado
    assert out["e2"].congestion_level == 0       # NaN -> flujo libre
    # nivel 0-5 con su marca de tiempo
    assert 0 <= out["e1"].congestion_level <= 5
    assert out["e1"].snapshot_timestamp == out["e2"].snapshot_timestamp


def test_snapshots_yields_all_edge_timestep_pairs():
    assert sum(1 for _ in _synthetic_adapter().snapshots()) == 4
