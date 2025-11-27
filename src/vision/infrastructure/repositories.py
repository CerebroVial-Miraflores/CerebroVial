import os
import csv
import time
from datetime import datetime
from ..domain import TrafficRepository, TrafficData

class CSVTrafficRepository(TrafficRepository):
    """
    Saves traffic data to CSV files.
    Rotates files daily.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._ensure_file_exists()

    def _get_filename(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.output_dir, f"traffic_log_{today}.csv")

    def _ensure_file_exists(self):
        filename = self._get_filename()
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "camera_id", "street_monitored", "car_count", "bus_count", 
                    "truck_count", "motorcycle_count", "total_vehicles", "occupancy_rate", 
                    "flow_rate_per_min", "avg_speed", "avg_density", "zone_id", "duration_seconds"
                ])

    def save(self, data: TrafficData):
        filename = self._get_filename()
        # Simple check if day changed to create new header if needed (though append works fine)
        if not os.path.exists(filename):
            self._ensure_file_exists()
            
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            # dt_str = datetime.fromtimestamp(data.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            # We follow CameraTrafficData schema which just has timestamp float, 
            # but usually CSVs are better with readable time. 
            # However, to be EXACTLY compatible with the schema/generator, let's stick to the requested fields.
            # The generator has: timestamp, camera_id, street_monitored, counts..., total, occupancy, flow.
            # I added avg_speed, avg_density, zone_id, duration as extras at the end.
            
            writer.writerow([
                f"{data.timestamp:.2f}",
                data.camera_id,
                data.street_monitored,
                data.car_count,
                data.bus_count,
                data.truck_count,
                data.motorcycle_count,
                data.total_vehicles,
                f"{data.avg_occupancy:.4f}",
                data.flow_rate_per_min,
                f"{data.avg_speed:.2f}" if data.avg_speed else "0.00",
                f"{data.avg_density:.2f}",
                data.zone_id,
                f"{data.duration_seconds:.2f}"
            ])
