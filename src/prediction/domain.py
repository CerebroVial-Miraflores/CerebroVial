"""
Domain entities for the Congestion Prediction module.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class TrafficFlowData:
    """
    Aggregated traffic flow data for a specific time window.
    """
    intersection_id: str
    timestamp: float
    vehicle_count: int
    average_speed: float

@dataclass
class CongestionPrediction:
    """
    Prediction result for future congestion levels.
    """
    intersection_id: str
    prediction_timestamp: float
    predicted_congestion_level: float  # 0.0 to 1.0
    confidence_interval: List[float]
