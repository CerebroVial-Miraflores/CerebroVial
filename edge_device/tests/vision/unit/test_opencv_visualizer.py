"""Tests del `OpenCVVisualizer` contra el Protocol `FrameRenderer` (§7.2).

Firma nueva: `render(frame: Frame, analysis: FrameAnalysis) -> np.ndarray`.
"""
import numpy as np
import pytest

from src.vision.domain.entities import (
    DetectedVehicle,
    Frame,
    FrameAnalysis,
    ZoneVehicleCount,
)
from src.vision.domain.protocols import FrameRenderer
from src.vision.domain.value_objects import VehicleId, ZoneId
from src.vision.presentation.visualization import (
    OpenCVVisualizer,
    build_visualizer_from_vision_cfg,
)


@pytest.fixture
def blank_frame():
    return Frame(id=7, timestamp=1.0, image=np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.fixture
def analysis_with_zone():
    return FrameAnalysis(
        frame_id=7,
        timestamp=1.0,
        vehicles=[
            DetectedVehicle(
                id=VehicleId("v1"),
                type="car",
                confidence=0.9,
                bbox=(10, 10, 50, 50),
                timestamp=1.0,
                speed=42.0,
            ),
            DetectedVehicle(
                id=VehicleId("v2"),
                type="bus",
                confidence=0.8,
                bbox=(100, 100, 200, 200),
                timestamp=1.0,
                speed=None,
            ),
        ],
        unique_vehicles=2,
        zones={
            ZoneId("north"): ZoneVehicleCount(
                zone_id=ZoneId("north"),
                count=2,
                vehicle_ids=[VehicleId("v1"), VehicleId("v2")],
                occupancy=0.25,
            ),
        },
    )


def test_satisfies_frame_renderer_protocol():
    """structural compliance con el Protocol del dominio."""
    visualizer: FrameRenderer = OpenCVVisualizer()
    assert hasattr(visualizer, "render") and callable(visualizer.render)


def test_render_returns_ndarray_same_shape_and_dtype(blank_frame, analysis_with_zone):
    visualizer = OpenCVVisualizer(zones_config={"north": [[0, 0], [100, 0], [100, 100], [0, 100]]})
    out = visualizer.render(blank_frame, analysis_with_zone)
    assert isinstance(out, np.ndarray)
    assert out.shape == blank_frame.image.shape
    assert out.dtype == blank_frame.image.dtype


def test_render_does_not_mutate_input_frame(blank_frame, analysis_with_zone):
    """El Protocol exige no mutar la entity — debe trabajar sobre una copia."""
    visualizer = OpenCVVisualizer(zones_config={"north": [[0, 0], [100, 0], [100, 100], [0, 100]]})
    snapshot = blank_frame.image.copy()
    out = visualizer.render(blank_frame, analysis_with_zone)
    assert np.array_equal(blank_frame.image, snapshot)
    assert out is not blank_frame.image


def test_render_draws_something_when_vehicles_present(blank_frame, analysis_with_zone):
    """Sanity: con vehículos+zonas, el output difiere del input puro."""
    visualizer = OpenCVVisualizer(zones_config={"north": [[0, 0], [100, 0], [100, 100], [0, 100]]})
    out = visualizer.render(blank_frame, analysis_with_zone)
    assert not np.array_equal(out, blank_frame.image)


def test_render_with_no_zones_config_handles_zone_data_gracefully(
    blank_frame, analysis_with_zone
):
    """zones_config vacío no rompe aunque el analysis traiga zonas."""
    visualizer = OpenCVVisualizer()
    out = visualizer.render(blank_frame, analysis_with_zone)
    assert isinstance(out, np.ndarray)


def test_render_with_empty_analysis_returns_only_header(blank_frame):
    analysis = FrameAnalysis(
        frame_id=0, timestamp=0.0, vehicles=[], unique_vehicles=0, zones={}
    )
    visualizer = OpenCVVisualizer()
    out = visualizer.render(blank_frame, analysis)
    assert out.shape == blank_frame.image.shape


def test_factory_extracts_legacy_list_zones():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"zones": {"east": [[0, 0], [10, 0], [10, 10]]}})
    visualizer = build_visualizer_from_vision_cfg(cfg)
    assert visualizer.zones_config == {"east": [[0, 0], [10, 0], [10, 10]]}


def test_factory_extracts_new_polygon_zones():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {"zones": {"south": {"polygon": [[0, 0], [10, 0], [10, 10]], "street": "X"}}}
    )
    visualizer = build_visualizer_from_vision_cfg(cfg)
    assert visualizer.zones_config == {"south": [[0, 0], [10, 0], [10, 10]]}


def test_factory_with_no_zones_returns_empty_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"zones": None})
    visualizer = build_visualizer_from_vision_cfg(cfg)
    assert visualizer.zones_config == {}
