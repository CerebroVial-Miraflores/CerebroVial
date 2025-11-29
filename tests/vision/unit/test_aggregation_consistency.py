import pytest
from unittest.mock import MagicMock
from src.vision.application.aggregators.sync_aggregator import TrafficDataAggregator
from src.vision.domain.entities import FrameAnalysis, ZoneVehicleCount

def test_aggregation_consistency():
    # Setup
    repo = MagicMock()
    aggregator = TrafficDataAggregator(repo, window_duration=1.0)
    
    # Simulate a vehicle changing class: ID "1" is car then truck then car
    # Frame 1: Car
    z1 = ZoneVehicleCount(
        zone_id="zone1", vehicle_count=1, timestamp=100,
        vehicles=["1"], vehicle_details={"1": "car"},
        camera_id="cam1", street_monitored="street1"
    )
    f1 = FrameAnalysis(frame_id=1, timestamp=100, vehicles=[], total_count=1, zones=[z1])
    
    # Frame 2: Truck (Noise)
    z2 = ZoneVehicleCount(
        zone_id="zone1", vehicle_count=1, timestamp=101,
        vehicles=["1"], vehicle_details={"1": "truck"},
        camera_id="cam1", street_monitored="street1"
    )
    f2 = FrameAnalysis(frame_id=2, timestamp=101, vehicles=[], total_count=1, zones=[z2])
    
    # Frame 3: Car
    z3 = ZoneVehicleCount(
        zone_id="zone1", vehicle_count=1, timestamp=102,
        vehicles=["1"], vehicle_details={"1": "car"},
        camera_id="cam1", street_monitored="street1"
    )
    f3 = FrameAnalysis(frame_id=3, timestamp=102, vehicles=[], total_count=1, zones=[z3])
    
    # Aggregate
    aggregator.aggregate_and_persist(f1)
    aggregator.aggregate_and_persist(f2)
    aggregator.aggregate_and_persist(f3)
    
    # Force flush
    aggregator.flush()
    
    # Verify
    assert repo.save.called
    data = repo.save.call_args[0][0]
    
    # Total vehicles should be 1 (unique ID "1")
    assert data.total_vehicles == 1
    
    # Counts should reflect majority vote (Car: 2, Truck: 1 -> Car)
    assert data.car_count == 1
    assert data.truck_count == 0
    
    # Sum of types should equal total
    sum_types = data.car_count + data.bus_count + data.truck_count + data.motorcycle_count
    assert sum_types == data.total_vehicles

def test_aggregation_multiple_vehicles():
    repo = MagicMock()
    aggregator = TrafficDataAggregator(repo, window_duration=1.0)
    
    # ID 1: Car (stable)
    # ID 2: Truck (stable)
    z1 = ZoneVehicleCount(
        zone_id="zone1", vehicle_count=2, timestamp=100,
        vehicles=["1", "2"], vehicle_details={"1": "car", "2": "truck"},
        camera_id="cam1", street_monitored="street1"
    )
    f1 = FrameAnalysis(frame_id=1, timestamp=100, vehicles=[], total_count=2, zones=[z1])
    
    aggregator.aggregate_and_persist(f1)
    aggregator.flush()
    
    data = repo.save.call_args[0][0]
    assert data.total_vehicles == 2
    assert data.car_count == 1
    assert data.truck_count == 1
    assert data.car_count + data.truck_count == data.total_vehicles
