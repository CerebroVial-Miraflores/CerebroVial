"""
Asynchronous aggregator that does not block the main pipeline.
"""
import time
import threading
import queue
from typing import List
from collections import defaultdict
from ...domain.entities import FrameAnalysis, TrafficData
from ...domain.repositories import TrafficRepository

class AsyncTrafficDataAggregator:
    """
    Aggregator that uses a separate thread for flushing to DB/CSV.
    The main pipeline never waits for persistence I/O.
    """
    
    def __init__(
        self, 
        repository: TrafficRepository, 
        window_duration: float = 60.0,
        flush_queue_size: int = 100
    ):
        self.repository = repository
        self.window_duration = window_duration
        
        # Thread-safe buffer
        self.buffer: List[FrameAnalysis] = []
        self.buffer_lock = threading.Lock()
        self.last_flush_time = time.time()
        
        # Queue for data ready to persist
        self.flush_queue = queue.Queue(maxsize=flush_queue_size)
        
        # Worker thread for I/O
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._flush_worker,
            name="PersistenceWorker",
            daemon=True
        )
        self._worker_thread.start()

    def aggregate_and_persist(self, analysis: FrameAnalysis):
        """
        Adds analysis to buffer. Non-blocking.
        """
        with self.buffer_lock:
            self.buffer.append(analysis)
            current_time = time.time()
            
            # If time passed, prepare flush
            if current_time - self.last_flush_time >= self.window_duration:
                self._schedule_flush()

    def _schedule_flush(self):
        """
        Moves data from buffer to flush queue (non-blocking).
        Called with lock acquired.
        """
        if not self.buffer:
            self.last_flush_time = time.time()
            return
        
        # Calculate aggregates BEFORE releasing lock
        timestamp = time.time()
        duration = timestamp - self.last_flush_time
        aggregated_data = self._compute_aggregates(self.buffer, timestamp, duration)
        
        # Clear buffer
        self.buffer = []
        self.last_flush_time = timestamp
        
        # Send to worker (non-blocking)
        try:
            self.flush_queue.put_nowait(aggregated_data)
        except queue.Full:
            print("[WARNING] Flush queue full - data dropped")

    def _compute_aggregates(
        self, 
        buffer: List[FrameAnalysis], 
        timestamp: float, 
        duration: float
    ) -> List[TrafficData]:
        """
        Computes aggregates without I/O. Pure CPU-bound.
        """
        zone_stats = defaultdict(list)
        
        for analysis in buffer:
            if not analysis.zones:
                continue
            for zone in analysis.zones:
                zone_stats[zone.zone_id].append(zone)
        
        results = []
        for zone_id, statuses in zone_stats.items():
            if not statuses:
                continue
            
            # Calculations (same as before)
            counts = [s.vehicle_count for s in statuses]
            avg_density = sum(counts) / len(counts)
            
            occupancies = [s.occupancy for s in statuses]
            avg_occupancy = sum(occupancies) / len(occupancies) if occupancies else 0.0
            
            # Weighted speed
            total_speed_sum = 0.0
            total_vehicles_with_speed = 0
            for s in statuses:
                if s.avg_speed > 0 and s.vehicle_count > 0:
                    total_speed_sum += s.avg_speed * s.vehicle_count
                    total_vehicles_with_speed += s.vehicle_count
            avg_speed = total_speed_sum / total_vehicles_with_speed if total_vehicles_with_speed > 0 else 0.0
            
            # Unique vehicles
            unique_vehicles = set()
            for s in statuses:
                if s.vehicles:
                    unique_vehicles.update(s.vehicles)
            flow_rate_per_min = len(unique_vehicles)
            
            # Breakdown by type
            unique_vehicles_by_type = defaultdict(set)
            for s in statuses:
                if s.vehicle_details:
                    for v_id, v_type in s.vehicle_details.items():
                        unique_vehicles_by_type[v_type].add(v_id)
            
            car_count = len(unique_vehicles_by_type.get('car', set()))
            bus_count = len(unique_vehicles_by_type.get('bus', set()))
            truck_count = len(unique_vehicles_by_type.get('truck', set()))
            motorcycle_count = len(unique_vehicles_by_type.get('motorcycle', set()))
            total_vehicles = flow_rate_per_min
            
            # Metadata
            metadata = statuses[0]
            
            data = TrafficData(
                timestamp=timestamp,
                zone_id=zone_id,
                camera_id=metadata.camera_id,
                street_monitored=metadata.street_monitored,
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
            results.append(data)
        
        return results

    def _flush_worker(self):
        """
        Worker thread that handles persistence I/O.
        """
        while not self._stop_event.is_set():
            try:
                # Block until data available or timeout
                data_batch = self.flush_queue.get(timeout=1.0)
                
                # Persist (I/O)
                for data in data_batch:
                    try:
                        self.repository.save(data)
                        print(f"[Aggregator] Saved {data.zone_id}: Density={data.avg_density:.1f}")
                    except Exception as e:
                        print(f"[ERROR] Failed to save data: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Flush worker error: {e}")

    def force_flush(self):
        """
        Forces immediate flush. Useful when closing application.
        """
        with self.buffer_lock:
            self._schedule_flush()

    def stop(self):
        """Stops the worker thread."""
        self.force_flush()
        self._stop_event.set()
        self._worker_thread.join(timeout=3.0)
