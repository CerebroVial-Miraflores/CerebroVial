import uuid
from datetime import datetime, timezone

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
    UniqueConstraint,
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

# --- Intersections (PMU) ---

class IntersectionDB(Base):
    """Intersección semaforizada del PMU de Miraflores (entidad de primera clase).

    intersection_id == ``nombre`` del mapeo canónico
    (``documentation/contracts/mapeo_pmu_edges_v2.yaml``); junction_id es el id del
    junction SUMO. tls_id se puebla SOLO cuando está verificado contra una config de
    control real (hoy: larco_benavides; el resto NULL — DEUDA-CTRL-TLS). El puente al
    grafo va por ``intersection_edges`` → ``graph_edges`` (NO hay FK a graph_nodes:
    junction_id es un str opaco). spatial_index=False: la fase NO crea índices GIST.
    """
    __tablename__ = "intersections"

    intersection_id = Column(String, primary_key=True, index=True)
    junction_id = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    los_pmu = Column(String, nullable=True)
    tls_id = Column(String, nullable=True)
    geom = Column(Geometry('POINT', srid=4326, spatial_index=False))


class IntersectionEdgeDB(Base):
    """Puente intersección → arista del grafo, con dirección (incoming/outgoing).

    edge_id es FK real a graph_edges.edge_id (id SUMO crudo, p. ej. ``129466113#3``):
    requiere que el net real esté cargado (``scripts/build_graph_geometry.py``) antes de
    sembrar. PK compuesta (intersection_id, edge_id): incoming/outgoing son disjuntos por
    intersección en el mapeo.
    """
    __tablename__ = "intersection_edges"

    intersection_id = Column(
        String, ForeignKey("intersections.intersection_id"), primary_key=True
    )
    edge_id = Column(String, ForeignKey("graph_edges.edge_id"), primary_key=True)
    direction = Column(String, nullable=False)  # "incoming" | "outgoing"

    __table_args__ = (
        CheckConstraint(
            "direction IN ('incoming', 'outgoing')",
            name="ck_intersection_edges_direction",
        ),
    )


class CameraDB(Base):
    """Cámara como ACCESORIO de una intersección (no de un graph_node).

    intersection_id es FK a intersections (nullable: una cámara puede no estar mapeada).
    stream_url es el HLS de Claro; la asociación cámara-Claro ↔ intersección es NOMINAL
    (DEUDA-CAM-GEO), sin concordancia geográfica real todavía.
    """
    __tablename__ = "cameras"

    camera_id = Column(String, primary_key=True, index=True)
    intersection_id = Column(
        String, ForeignKey("intersections.intersection_id"), nullable=True
    )
    stream_url = Column(String, nullable=True)
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


# --- GRU prediction persistence (TTH-09 Fase 5, CT-09.5) ---

class PredictionDB(Base):
    """Append-only registry of every GRU prediction (CT-09.5).

    One row per (inference x direction x step): a single ``/predictions/predict``
    call yields 4 directions x 30 steps = 120 rows. The model is a multi-output
    CLASSIFIER (D-009, jam level 0-5): we store the discrete ``level`` (argmax of
    ``probs``) and the 6-float ``probs`` vector -- NOT a continuous ratio (obsolete;
    see ``prediction_contract.md`` 6/8, Nota D-005). Source of data for HU-14; the
    predicted-vs-observed join, the real observation and the 24h window are HU-14
    scope, NOT TTH-09. ``intersection_id`` is the opaque request id (no FK to
    graph_nodes, contract 8). Best-effort write-path: a persistence failure must
    not break the response. Exposes no update/delete (append-only invariant).
    """
    __tablename__ = "predictions"

    prediction_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    intersection_id = Column(String, nullable=False)  # opaque request id; no FK (contract 8)
    direction = Column(String, nullable=False)  # "N" | "S" | "E" | "W"
    step = Column(Integer, nullable=False)  # 1..30 (t+1 ... t+30)
    level = Column(Integer, nullable=False)  # 0..5 (argmax of probs, D-009)
    probs = Column(JsonType, nullable=False)  # list[float], 6 softmax
    model_version = Column(String, nullable=False)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index(
            "ix_predictions_intersection_id_generated_at",
            "intersection_id",
            "generated_at",
        ),
    )


# --- TomTom corridors (track feature/tomtom, Fase B) ---

class CorridorDB(Base):
    """Corredor dibujado por el operador: cadena ordenada y dirigida de aristas.

    El SENTIDO del corredor vive en la secuencia de aristas (``corridor_edges.sequence``),
    no en un campo aparte: las aristas de ``graph_edges`` ya son dirigidas (par ``-id``/``id``
    por calzada). Entidad propia (dato nuestro, persistible); NADA de tráfico de TomTom
    vive acá.
    """
    __tablename__ = "corridors"

    corridor_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String, nullable=True)  # operator; sin FK (patrón engine_active_state.activated_by)


class CorridorEdgeDB(Base):
    """Puente corredor → arista (dirigido). Molde: ``intersection_edges``.

    ``tomtom_openlr`` es un IDENTIFICADOR de segmento (OpenLR, estándar abierto), NO un
    Result de tráfico de TomTom. Los valores de tráfico (currentSpeed, nivel, travel-time)
    y la GEOMETRÍA de TomTom JAMÁS se persisten (ToS 11.6.1). NULL = arista sin cobertura
    TomTom → se pinta "sin dato".

    PK compuesta ``(corridor_id, edge_id)``: una arista puede estar en varios corredores,
    una sola vez por corredor. ``UniqueConstraint(corridor_id, sequence)`` garantiza orden
    sin colisiones en la cadena (dos aristas con la misma posición es un estado imposible).
    """
    __tablename__ = "corridor_edges"

    corridor_id = Column(String, ForeignKey("corridors.corridor_id"), primary_key=True)
    edge_id = Column(String, ForeignKey("graph_edges.edge_id"), primary_key=True)
    sequence = Column(Integer, nullable=False)
    tomtom_openlr = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("corridor_id", "sequence", name="uq_corridor_edges_corridor_sequence"),
    )
