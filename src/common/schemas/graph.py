from typing import List, Literal
from pydantic import BaseModel, Field

class GraphConnection(BaseModel):
    """
    Represents a valid movement (turn) from one edge to another.
    """
    from_edge_id: str = Field(..., description="Incoming edge ID")
    to_edge_id: str = Field(..., description="Outgoing edge ID")
    turn_direction: Literal['Left', 'Right', 'Straight', 'U-Turn'] = Field(..., description="Direction of the turn")

class GraphNode(BaseModel):
    """
    Represents a node in the traffic graph (intersection).
    """
    node_id: str = Field(..., description="Unique node identifier")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    has_camera: bool = Field(False, description="Whether this node has a camera")

class GraphEdge(BaseModel):
    """
    Represents an edge in the traffic graph (road segment).
    Corresponds to GRAPH_TOPOLOGY_STATIC table.
    """
    edge_id: str = Field(..., description="Unique edge identifier")
    source_node: str = Field(..., description="Source node ID")
    target_node: str = Field(..., description="Target node ID")
    distance_m: float = Field(..., description="Physical distance in meters")
    lanes: int = Field(..., ge=1, description="Number of lanes")
    waze_segment_ids: List[str] = Field(default_factory=list, description="Mapped Waze segment IDs")
    incoming_connections: List[GraphConnection] = Field(default_factory=list, description="Allowed turns into this edge")
    outgoing_connections: List[GraphConnection] = Field(default_factory=list, description="Allowed turns out of this edge")

class GraphTopology(BaseModel):
    """
    Represents the full graph topology.
    """
    nodes: List[GraphNode]
    edges: List[GraphEdge]
