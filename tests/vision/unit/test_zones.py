import pytest
import numpy as np
from src.vision.infrastructure.zones import ZoneManager
from src.vision.domain import DetectedVehicle

def test_zone_manager_initialization():
    config = {
        "zone1": [[0, 0], [100, 0], [100, 100], [0, 100]]
    }
    manager = ZoneManager(config, resolution=(200, 200))
    assert "zone1" in manager.zones

def test_zone_manager_update():
    config = {
        "zone1": [[0, 0], [100, 0], [100, 100], [0, 100]]
    }
    manager = ZoneManager(config, resolution=(200, 200))
    
    # Vehicle inside zone
    v1 = DetectedVehicle(id="1", type="car", confidence=0.9, bbox=(10, 10, 50, 50), timestamp=0)
    # Vehicle outside zone
    v2 = DetectedVehicle(id="2", type="car", confidence=0.9, bbox=(150, 150, 190, 190), timestamp=0)
    
    statuses = manager.update([v1, v2])
    
    assert len(statuses) == 1
    assert statuses[0].zone_id == "zone1"
    assert statuses[0].count == 1

def test_zone_manager_empty_detections():
    config = {
        "zone1": [[0, 0], [100, 0], [100, 100], [0, 100]]
    }
    manager = ZoneManager(config, resolution=(200, 200))
    statuses = manager.update([])
    assert statuses[0].count == 0
