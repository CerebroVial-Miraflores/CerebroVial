"""Tests del broadcaster de red + wake del replay en vivo (TTH-12 Fase 3, CT-12.6)."""
import numpy as np
import pytest

from src.congestion.infrastructure.broadcaster import NetworkCongestionBroadcaster
from src.congestion.infrastructure.repositories import WazeJamsRepo
from src.congestion.infrastructure.sumo_replay_adapter import SumoReplayAdapter


@pytest.mark.asyncio
async def test_broadcaster_delivers_wake_to_subscribers():
    b = NetworkCongestionBroadcaster()
    q1 = await b.subscribe()
    q2 = await b.subscribe()
    delivered = await b.publish()           # wake de red, sin payload
    assert delivered == 2
    assert (await q1.get())["type"] == "congestion-updated"
    assert (await q2.get())["type"] == "congestion-updated"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    b = NetworkCongestionBroadcaster()
    q = await b.subscribe()
    await b.unsubscribe(q)
    assert await b.publish() == 0


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


def test_live_replay_fires_wake_per_step(congestion_session):
    """CT-12.6: el replay en vivo dispara un wake tras escribir cada paso."""
    repo = WazeJamsRepo(congestion_session)
    adapter = _synthetic_adapter()
    wakes = {"n": 0}
    steps = adapter.live_replay(
        repo, speedup=1e9, max_steps=2, wake=lambda: wakes.__setitem__("n", wakes["n"] + 1),
    )
    assert steps == 2
    assert wakes["n"] == 2                 # un wake por paso
    assert repo.count() == 4               # 2 aristas × 2 pasos escritos
