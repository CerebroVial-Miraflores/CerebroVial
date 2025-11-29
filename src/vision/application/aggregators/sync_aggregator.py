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
            
            # Breakdown by type with conflict resolution (Majority Vote)
            vehicle_type_observations = defaultdict(list)
            for s in statuses:
                if s.vehicle_details:
                    for v_id, v_type in s.vehicle_details.items():
                        vehicle_type_observations[v_id].append(v_type)
            
            # Resolve type for each vehicle ID
            resolved_vehicle_types = {}
            for v_id, types in vehicle_type_observations.items():
                if not types:
                    continue
                # Pick most frequent type
                resolved_type = max(set(types), key=types.count)
                resolved_vehicle_types[v_id] = resolved_type
            
            # Count by resolved type
            counts_by_type = defaultdict(int)
            for v_type in resolved_vehicle_types.values():
                counts_by_type[v_type] += 1
            
            car_count = counts_by_type['car']
            bus_count = counts_by_type['bus']
            truck_count = counts_by_type['truck']
            motorcycle_count = counts_by_type['motorcycle']
            
            # Total vehicles is the number of unique IDs
            total_vehicles = len(resolved_vehicle_types)
            
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
                vehicle_types=dict(counts_by_type)
            )
            
            self.repository.save(data)
            print(f"[Aggregator] Saved stats for {zone_id}: Density={avg_density:.1f}")

        self.buffer = []
        self.last_flush_time = timestamp
