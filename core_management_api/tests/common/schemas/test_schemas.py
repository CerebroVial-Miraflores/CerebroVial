import pytest
from pydantic import ValidationError
from cerebrovial_shared.schemas import (
    Camera, GraphEdge, GraphConnection,
    WazeJam
)

# --- Camera Tests ---
def test_camera_valid():
    cam = Camera(
        camera_id="cam_1", lat=-12.0, lon=-77.0, heading=90, fov=60
    )
    assert cam.camera_id == "cam_1"

def test_camera_invalid_heading():
    with pytest.raises(ValidationError):
        Camera(camera_id="cam_1", lat=0, lon=0, heading=400, fov=60)

def test_camera_invalid_fov():
    with pytest.raises(ValidationError):
        Camera(camera_id="cam_1", lat=0, lon=0, heading=0, fov=-10)

# --- Graph Tests ---
def test_graph_connection():
    conn = GraphConnection(from_edge_id="e1", to_edge_id="e2", turn_direction="Left")
    assert conn.turn_direction == "Left"

def test_graph_edge_valid():
    edge = GraphEdge(
        edge_id="e1", source_node="n1", target_node="n2",
        distance_m=100, lanes=2
    )
    assert edge.lanes == 2

def test_graph_edge_invalid_lanes():
    with pytest.raises(ValidationError):
        GraphEdge(
            edge_id="e1", source_node="n1", target_node="n2",
            distance_m=100, lanes=0
        )

# --- Waze Tests ---
def test_waze_jam_valid():
    jam = WazeJam(
        event_uuid="j1", snapshot_timestamp=123456789,
        waze_line_geometry=[{"x": 0, "y": 0}],
        speed_mps=10.0, delay_seconds=5, congestion_level=1,
        jam_length_m=100, road_type=1
    )
    assert jam.speed_mps == 10.0

def test_waze_jam_invalid_speed():
    with pytest.raises(ValidationError):
        WazeJam(
            event_uuid="j1", snapshot_timestamp=123456789,
            waze_line_geometry=[{"x": 0, "y": 0}],
            speed_mps=-5.0, delay_seconds=5, congestion_level=1,
            jam_length_m=100, road_type=1
        )
