import time
from typing import List, Dict
from collections import defaultdict
from ..domain import FrameAnalysis, TrafficData, TrafficRepository

class TrafficDataAggregator:
    """
    Aggregates frame analysis results over a time window and saves to repository.
    """
    def __init__(self, repository: TrafficRepository, window_duration: float = 60.0):
        self.repository = repository
        self.window_duration = window_duration
        self.buffer: List[FrameAnalysis] = []
        self.last_flush_time = time.time()

    def aggregate_and_persist(self, analysis: FrameAnalysis):
        """
        Add analysis to buffer and flush if window exceeded.
        """
        self.buffer.append(analysis)
        
        current_time = time.time()
        if current_time - self.last_flush_time >= self.window_duration:
            self.flush()

    def flush(self):
        """
        Aggregate buffered data and save.
        """
        if not self.buffer:
            self.last_flush_time = time.time()
            return

        # Group by zone
        zone_stats = defaultdict(list)
        
        for analysis in self.buffer:
            if not analysis.zones:
                continue
            for zone in analysis.zones:
                zone_stats[zone.zone_id].append(zone)

        # Calculate aggregates for each zone
        timestamp = time.time()
        duration = timestamp - self.last_flush_time
        
        for zone_id, statuses in zone_stats.items():
            if not statuses:
                continue
                
            # Avg Density
            counts = [s.vehicle_count for s in statuses]
            avg_density = sum(counts) / len(counts)
            
            # Avg Speed
            # Weighted average by vehicle count in each frame to be more accurate
            total_speed_sum = 0.0
            total_vehicles_with_speed = 0
            
            for s in statuses:
                if s.avg_speed > 0 and s.vehicle_count > 0:
                    total_speed_sum += s.avg_speed * s.vehicle_count
                    total_vehicles_with_speed += s.vehicle_count
            
            avg_speed = total_speed_sum / total_vehicles_with_speed if total_vehicles_with_speed > 0 else 0.0
            
            # Vehicle Types
            vehicle_types = defaultdict(int)
            for s in statuses:
                for v_type, count in s.vehicle_types.items():
                    vehicle_types[v_type] += count
            
            # Normalize types to average per frame (density-like) or total seen?
            # TrafficData definition says "Count of unique vehicles". 
            # Since we don't track unique IDs across frames in aggregator easily without a set,
            # and ZoneVehicleCount.vehicles gives us IDs, we can use that if we want unique counts.
            # But simpler for now: Average distribution per frame (density breakdown)
            # OR: Total observations (accumulated).
            # Let's go with Average Density per Type for consistency with avg_density.
            
            avg_vehicle_types = {k: v / len(statuses) for k, v in vehicle_types.items()}
            
            data = TrafficData(
                timestamp=timestamp,
                zone_id=zone_id,
                duration_seconds=duration,
                avg_density=avg_density,
                avg_speed=avg_speed,
                vehicle_types=avg_vehicle_types
            )
            
            self.repository.save(data)
            print(f"[Aggregator] Saved stats for {zone_id}: Density={avg_density:.1f}")

        self.buffer = []
        self.last_flush_time = timestamp
