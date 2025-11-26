from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, BigInteger
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from .database import Base
from datetime import datetime

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
