import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Text,
    Index,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from geoalchemy2 import Geometry
from .database import Base

# JSONB in PostgreSQL, JSON in SQLite (used by the test suite).
JsonType = JSONB().with_variant(JSON(), "sqlite")

# --- Graph Topology ---

class GraphNodeDB(Base):
    __tablename__ = "graph_nodes"

    node_id = Column(String, primary_key=True, index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    has_camera = Column(Boolean, default=False)
    geom = Column(Geometry('POINT', srid=4326))

class GraphEdgeDB(Base):
    __tablename__ = "graph_edges"

    edge_id = Column(String, primary_key=True, index=True)
    source_node = Column(String, ForeignKey("graph_nodes.node_id"), nullable=False)
    target_node = Column(String, ForeignKey("graph_nodes.node_id"), nullable=False)
    distance_m = Column(Float, nullable=False)
    lanes = Column(Integer, nullable=False)
    geom = Column(Geometry('LINESTRING', srid=4326))

class CameraDB(Base):
    __tablename__ = "cameras"

    camera_id = Column(String, primary_key=True, index=True)
    node_id = Column(String, ForeignKey("graph_nodes.node_id"), nullable=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    heading = Column(Float, nullable=False)
    fov = Column(Float, nullable=False)
    geom = Column(Geometry('POINT', srid=4326))

# --- Waze Data (Hypertables) ---

class WazeJamDB(Base):
    __tablename__ = "waze_jams"

    event_uuid = Column(String, primary_key=True, index=True)
    snapshot_timestamp = Column(DateTime, primary_key=True, index=True) # Part of PK for hypertable
    edge_id = Column(String, ForeignKey("graph_edges.edge_id"), nullable=True)
    speed_mps = Column(Float, nullable=False)
    delay_seconds = Column(Integer, nullable=False)
    congestion_level = Column(Integer, nullable=False)
    jam_length_m = Column(Integer, nullable=False)
    road_type = Column(Integer, nullable=False)
    geom = Column(Geometry('LINESTRING', srid=4326))

class WazeAlertDB(Base):
    __tablename__ = "waze_alerts"

    alert_uuid = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, primary_key=True, index=True) # Part of PK for hypertable
    edge_id = Column(String, ForeignKey("graph_edges.edge_id"), nullable=True)
    alert_type = Column(String, nullable=False)
    alert_subtype = Column(String, nullable=True)
    reliability = Column(Integer, nullable=False)
    confidence = Column(Integer, nullable=False)
    geom = Column(Geometry('POINT', srid=4326))

# --- Vision Data (Hypertables) ---

class VisionTrackDB(Base):
    __tablename__ = "vision_tracks"

    track_uuid = Column(String, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.camera_id"), nullable=False)
    entry_timestamp = Column(DateTime, primary_key=True, index=True) # Part of PK for hypertable
    exit_timestamp = Column(DateTime, nullable=False)
    class_id = Column(Integer, nullable=False)
    avg_speed_px = Column(Float, nullable=False)
    geom = Column(Geometry('LINESTRING', srid=4326)) # Trajectory

class VisionFlowDB(Base):
    __tablename__ = "vision_flows"

    flow_id = Column(String, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.camera_id"), nullable=False)
    timestamp_bin = Column(DateTime, primary_key=True, index=True) # Part of PK for hypertable
    period_seconds = Column(Integer, nullable=False)
    from_edge_id = Column(String, ForeignKey("graph_edges.edge_id"), nullable=True)
    to_edge_id = Column(String, ForeignKey("graph_edges.edge_id"), nullable=True)
    turn_direction = Column(String, nullable=True)
    vehicle_count = Column(Integer, nullable=False)
    avg_speed_mps = Column(Float, nullable=True)


# --- Auth ---

class UserDB(Base):
    __tablename__ = "users"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # operator | manager | admin (canónicos DHU-022)
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )


# --- Adaptive engine persistence (TTH-10, SDD §4.2, data-model.md §2) ---

class MotorDecisionDB(Base):
    """Append-only history of every decision produced by the adaptive engine.

    The write-path resolves the opaque ``intersection_id`` from the request
    payload to a ``graph_nodes.node_id`` (DHU-021 V1) before insertion;
    ``inputs_snapshot`` captures the request payload at the moment of decision.
    ``flow_total`` and ``y_load_factor`` reflect the values actually computed
    by the engine (not recomputed downstream).
    """
    __tablename__ = "motor_decisions"

    decision_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    node_id = Column(
        String,
        ForeignKey("graph_nodes.node_id"),
        nullable=False,
        index=True,
    )
    decided_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    mode = Column(String, nullable=False)  # "webster" | "max_pressure"
    cycle_seconds = Column(Float, nullable=False)
    flow_total = Column(Float, nullable=False)
    y_load_factor = Column(Float, nullable=True)  # NULL when peak hits default_cycle (WebsterInfeasible)
    next_phase = Column(String, nullable=True)
    reasoning = Column(Text, nullable=False)
    phase_timings = Column(JsonType, nullable=False)
    adjustments = Column(JsonType, nullable=False, default=list)
    inputs_snapshot = Column(JsonType, nullable=True)

    __table_args__ = (
        Index(
            "ix_motor_decisions_node_id_decided_at",
            "node_id",
            text("decided_at DESC"),
        ),
    )


class EngineActiveStateDB(Base):
    """Currently active strategy per intersection (one row per node).

    Mutable. The activation event is distinct from the decision event:
    ``motor_decisions.decided_at`` records when the engine computed a
    recommendation; ``engine_active_state.activated_at`` records when an
    operator (or future automation) promoted that recommendation to active.
    """
    __tablename__ = "engine_active_state"

    node_id = Column(
        String,
        ForeignKey("graph_nodes.node_id"),
        primary_key=True,
    )
    active_decision_id = Column(
        String(36),
        ForeignKey("motor_decisions.decision_id"),
        nullable=False,
    )
    activated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    activated_by = Column(String, nullable=True)
