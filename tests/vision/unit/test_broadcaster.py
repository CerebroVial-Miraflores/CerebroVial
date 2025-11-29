import pytest
import asyncio
from src.vision.infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster

@pytest.fixture
def broadcaster():
    return RealtimeBroadcaster()

@pytest.mark.asyncio
async def test_subscribe_unsubscribe(broadcaster):
    queue = await broadcaster.subscribe("cam1")
    assert isinstance(queue, asyncio.Queue)
    assert "cam1" in broadcaster._subscribers
    assert queue in broadcaster._subscribers["cam1"]
    
    await broadcaster.unsubscribe("cam1", queue)
    assert "cam1" not in broadcaster._subscribers

@pytest.mark.asyncio
async def test_broadcast(broadcaster):
    q1 = await broadcaster.subscribe("cam1")
    q2 = await broadcaster.subscribe("cam1")
    q3 = await broadcaster.subscribe("cam2")
    
    data = {"test": "data"}
    await broadcaster.broadcast("cam1", data)
    
    assert await q1.get() == data
    assert await q2.get() == data
    assert q3.empty()

@pytest.mark.asyncio
async def test_broadcast_slow_consumer(broadcaster):
    # Queue size 1
    q1 = await broadcaster.subscribe("cam1", queue_size=1)
    
    await broadcaster.broadcast("cam1", {"msg": 1})
    await broadcaster.broadcast("cam1", {"msg": 2}) # Should be dropped or fill queue
    
    # First message should be there
    assert await q1.get() == {"msg": 1}
    
    # Second message might be dropped if queue was full. 
    # Implementation uses put_nowait, so if full it raises QueueFull and catches it.
    # So q1 should be empty now if we didn't consume fast enough before second broadcast?
    # Wait, queue size 1. 
    # 1. put msg1 -> OK. q size = 1.
    # 2. put msg2 -> Full -> Exception caught -> Skipped.
    # So q1 should have msg1, and msg2 is lost.
    
    assert q1.empty()

@pytest.mark.asyncio
async def test_latest_state(broadcaster):
    data = {"state": "initial"}
    await broadcaster.broadcast("cam1", data)
    
    # New subscriber should get latest state immediately
    q1 = await broadcaster.subscribe("cam1")
    assert await q1.get() == data
