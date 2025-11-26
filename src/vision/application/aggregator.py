import time
from typing import List, Dict
from collections import defaultdict
from ..domain import FrameAnalysis, TrafficData, TrafficRepository

class TrafficAggregator:
    """
    Aggregates frame analysis results over a time window and saves to repository.
    """
    def __init__(self, repository: TrafficRepository, window_duration: float = 60.0):
        self.repository = repository
        self.window_duration = window_duration
        self.buffer: List[FrameAnalysis] = []
        self.last_flush_time = time.time()

    def process(self, analysis: FrameAnalysis):
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
            counts = [s.count for s in statuses]
            avg_density = sum(counts) / len(counts)
            
            # Avg Speed & Types (Need to look at vehicles in zone)
            # This is tricky because ZoneStatus only has count.
            # Ideally ZoneStatus should have more info, or we look at vehicles in buffer
            # For now, let's approximate speed from ALL vehicles in frame if we can't filter by zone easily here
            # OR we update ZoneStatus to include speed stats.
            
            # Let's do a simpler approach for now:
            # We will just aggregate what we have. Speed might need to be added to ZoneStatus later.
            # For now, avg_speed will be 0 if not tracked per zone.
            
            # TODO: Improve ZoneStatus to include speed/types per zone.
            # For now, we will just use the global frame analysis for types if needed, 
            # but that's inaccurate for specific zones.
            
            # Let's leave speed/types empty/approx for now to get the pipeline working.
            avg_speed = 0.0
            vehicle_types = {} 
            
            data = TrafficData(
                timestamp=timestamp,
                zone_id=zone_id,
                duration_seconds=duration,
                avg_density=avg_density,
                avg_speed=avg_speed,
                vehicle_types=vehicle_types
            )
            
            self.repository.save(data)
            print(f"[Aggregator] Saved stats for {zone_id}: Density={avg_density:.1f}")

        self.buffer = []
        self.last_flush_time = timestamp
