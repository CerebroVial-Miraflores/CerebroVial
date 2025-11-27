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
            
            # Avg Occupancy
            occupancies = [s.occupancy for s in statuses]
            avg_occupancy = sum(occupancies) / len(occupancies) if occupancies else 0.0
            
            # Avg Speed
            # Weighted average by vehicle count in each frame to be more accurate
            total_speed_sum = 0.0
            total_vehicles_with_speed = 0
            
            for s in statuses:
                if s.avg_speed > 0 and s.vehicle_count > 0:
                    total_speed_sum += s.avg_speed * s.vehicle_count
                    total_vehicles_with_speed += s.vehicle_count
            
            avg_speed = total_speed_sum / total_vehicles_with_speed if total_vehicles_with_speed > 0 else 0.0
            
            # Flow Rate (Unique vehicles seen in the window)
            # We collect all unique vehicle IDs seen in this zone across all frames in the buffer
            unique_vehicles = set()
            for s in statuses:
                if s.vehicles:
                    unique_vehicles.update(s.vehicles)
            flow_rate_per_min = len(unique_vehicles)
            
            # Vehicle Types Breakdown (Average per frame to represent density/composition)
            # OR Total unique vehicles per type?
            # CameraTrafficData says "Number of cars detected". 
            # If flow_rate_per_min is unique vehicles, then type counts should probably sum up to flow_rate_per_min (roughly).
            # Let's count unique vehicles per type.
            
            unique_vehicles_by_type = defaultdict(set)
            for s in statuses:
                if s.vehicles and s.vehicle_types:
                    # This is tricky because s.vehicle_types is a summary dict, not linked to IDs directly in the struct
                    # But s.vehicles has IDs. We need to know the type of each ID.
                    # ZoneVehicleCount doesn't map ID to type directly in the current struct (it has list of IDs and dict of types).
                    # We can approximate by accumulating the max count seen or averaging.
                    # BUT, for flow rate we used unique IDs.
                    # Let's assume for now we just want the average count per frame (Density of cars, Density of buses...)
                    # This is more robust for "how many cars are there right now".
                    pass

            # Actually, let's use the accumulated counts from the frames and average them (Average Density per Type)
            # This matches "car_count" as "Average number of cars present".
            
            type_counts = defaultdict(list)
            for s in statuses:
                for v_type, count in s.vehicle_types.items():
                    type_counts[v_type].append(count)
            
            avg_type_counts = {k: sum(v) / len(statuses) for k, v in type_counts.items()}
            
            # Metadata (take from first status)
            camera_id = statuses[0].camera_id
            street_monitored = statuses[0].street_monitored
            
            data = TrafficData(
                timestamp=timestamp,
                zone_id=zone_id,
                camera_id=camera_id,
                street_monitored=street_monitored,
                duration_seconds=duration,
                avg_density=avg_density,
                total_vehicles=int(avg_density), # Mapping avg_density to total_vehicles (rounded)
                avg_speed=avg_speed,
                avg_occupancy=avg_occupancy,
                flow_rate_per_min=flow_rate_per_min,
                car_count=int(avg_type_counts.get('car', 0)),
                bus_count=int(avg_type_counts.get('bus', 0)),
                truck_count=int(avg_type_counts.get('truck', 0)),
                motorcycle_count=int(avg_type_counts.get('motorcycle', 0)),
                vehicle_types=avg_type_counts
            )
            
            self.repository.save(data)
            print(f"[Aggregator] Saved stats for {zone_id}: Density={avg_density:.1f}")

        self.buffer = []
        self.last_flush_time = timestamp
