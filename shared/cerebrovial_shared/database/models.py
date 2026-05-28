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
    CheckConstraint,
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

# --- Vision Data (Hypertable) ---

class VisionAggregateDB(Base):
    __tablename__ = "vision_aggregates"

    camera_id = Column(String, primary_key=True)
    zone_id = Column(String, primary_key=True)
    # TIMESTAMPTZ: el dominio define la ventana como tz-aware UTC (tth-08-fase1-diseno §5.4).
    # index=True suple el indice por tiempo y evita el indice por defecto de create_hypertable.
    window_start = Column(DateTime(timezone=True), primary_key=True, index=True) # Part of PK for hypertable
    window_end = Column(DateTime(timezone=True), nullable=False)
    window_duration_seconds = Column(Float, nullable=False)
    unique_vehicles = Column(Integer, nullable=False)
    car_count = Column(Integer, nullable=False, server_default="0")
    bus_count = Column(Integer, nullable=False, server_default="0")
    truck_count = Column(Integer, nullable=False, server_default="0")
    motorcycle_count = Column(Integer, nullable=False, server_default="0")
    mean_speed_kmh = Column(Float, nullable=True)
    flow_vehicles_per_hour = Column(Float, nullable=False)
    mean_occupancy = Column(Float, nullable=False)
    density_vehicles_per_km = Column(Float, nullable=True)
    queue = Column(Integer, nullable=True)  # NULL en MVP1; F41 lo poblara

    __table_args__ = (
        CheckConstraint("unique_vehicles >= 0", name="ck_vision_aggregates_unique_vehicles_nonneg"),
        CheckConstraint("car_count >= 0", name="ck_vision_aggregates_car_count_nonneg"),
        CheckConstraint("bus_count >= 0", name="ck_vision_aggregates_bus_count_nonneg"),
        CheckConstraint("truck_count >= 0", name="ck_vision_aggregates_truck_count_nonneg"),
        CheckConstraint("motorcycle_count >= 0", name="ck_vision_aggregates_motorcycle_count_nonneg"),
        CheckConstraint("flow_vehicles_per_hour >= 0", name="ck_vision_aggregates_flow_nonneg"),
        CheckConstraint("mean_occupancy BETWEEN 0 AND 1", name="ck_vision_aggregates_occupancy_range"),
    )


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
