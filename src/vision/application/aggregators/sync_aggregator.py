import time
from typing import List, Dict, Optional
from collections import defaultdict
from ...domain.entities import FrameAnalysis, TrafficData, ZoneVehicleCount
from ...domain.repositories import TrafficRepository

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
            
            # Vehicle Types Breakdown (Total unique vehicles per type)
            # We use the vehicle_details map (ID -> Type) to count unique vehicles by type
            unique_vehicles_by_type = defaultdict(set)
            
            for s in statuses:
                if s.vehicle_details:
                    for v_id, v_type in s.vehicle_details.items():
                        unique_vehicles_by_type[v_type].add(v_id)
            
            # Counts of unique vehicles
            car_count = len(unique_vehicles_by_type.get('car', set()))
            bus_count = len(unique_vehicles_by_type.get('bus', set()))
            truck_count = len(unique_vehicles_by_type.get('truck', set()))
            motorcycle_count = len(unique_vehicles_by_type.get('motorcycle', set()))
            
            # Total unique vehicles (Flow Rate)
            # Note: A vehicle might be detected as different types in different frames (flickering),
            # but usually the ID persists. We should count unique IDs overall for flow_rate.
            # flow_rate_per_min is already calculated above as len(unique_vehicles)
            
            # For consistency, total_vehicles in CameraTrafficData usually refers to the sum of counts or flow.
            # Let's map total_vehicles to flow_rate_per_min (Total Unique)
            total_vehicles = flow_rate_per_min
            
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
                total_vehicles=total_vehicles,
                avg_speed=avg_speed,
                avg_occupancy=avg_occupancy,
                flow_rate_per_min=flow_rate_per_min,
                car_count=car_count,
                bus_count=bus_count,
                truck_count=truck_count,
                motorcycle_count=motorcycle_count,
                vehicle_types={k: len(v) for k, v in unique_vehicles_by_type.items()}
            )
            
            self.repository.save(data)
            print(f"[Aggregator] Saved stats for {zone_id}: Density={avg_density:.1f}")

        self.buffer = []
        self.last_flush_time = timestamp
