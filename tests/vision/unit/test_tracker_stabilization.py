import pytest
from unittest.mock import MagicMock
from src.vision.infrastructure.tracking.supervision_tracker import SupervisionTracker
from src.vision.domain.entities import DetectedVehicle

def test_class_stabilization():
    # Setup tracker with classes
    classes = {'car': 1, 'truck': 2}
    tracker = SupervisionTracker(classes)
    
    # Mock ByteTrack to return consistent tracker_id=1 for all updates
    # We need to mock update_with_detections to return a Detections object
    # with tracker_id=[1] and the class_id we passed in.
    
    # Since SupervisionTracker wraps sv.ByteTrack, and sv.ByteTrack logic is complex (IoU matching),
    # it's hard to mock purely without running the real tracker logic.
    # However, if we provide consistent bounding boxes, ByteTrack should assign the same ID.
    
    # Sequence of detections: 3 cars, 1 truck (noise), 2 cars.
    # Expected: Always 'car' after stabilization, or at least 'car' at the end.
    
    bbox = (100, 100, 200, 200)
    
    # Frame 1: Car
    d1 = [DetectedVehicle("0", "car", 0.9, bbox, 1.0)]
    res1 = tracker.track(d1)
    assert res1[0].type == 'car'
    assert tracker.class_history[1] == [1] # ID 1 is car
    
    # Frame 2: Car
    d2 = [DetectedVehicle("0", "car", 0.9, bbox, 1.1)]
    res2 = tracker.track(d2)
    assert res2[0].type == 'car'
    assert tracker.class_history[1] == [1, 1]
    
    # Frame 3: Truck (Noise)
    d3 = [DetectedVehicle("0", "truck", 0.8, bbox, 1.2)]
    res3 = tracker.track(d3)
    # Should still be car because history is [1, 1, 2] -> Majority is 1 (car)
    assert res3[0].type == 'car' 
    assert tracker.class_history[1] == [1, 1, 2]
    
    # Frame 4: Truck (Noise continues?)
    d4 = [DetectedVehicle("0", "truck", 0.8, bbox, 1.3)]
    res4 = tracker.track(d4)
    # History: [1, 1, 2, 2]. Tie? max() behavior depends on implementation, usually first one encountered or lowest value.
    # If tie, it might flip. Let's see.
    
    # Frame 5: Car
    d5 = [DetectedVehicle("0", "car", 0.9, bbox, 1.4)]
    res5 = tracker.track(d5)
    # History: [1, 1, 2, 2, 1]. Majority: 1 (car).
    assert res5[0].type == 'car'
    
def test_history_limit():
    classes = {'car': 1}
    tracker = SupervisionTracker(classes)
    bbox = (100, 100, 200, 200)
    
    # Add 35 detections
    for i in range(35):
        d = [DetectedVehicle("0", "car", 0.9, bbox, float(i))]
        tracker.track(d)
        
    # Check history size limit (30)
    # Note: tracker_id might not be 1 if ByteTrack assigns differently, but usually it starts at 1.
    # We get the tracker_id from the internal dict keys.
    tracker_id = list(tracker.class_history.keys())[0]
    assert len(tracker.class_history[tracker_id]) == 30
