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
                    "timestamp", "datetime", "zone_id", "duration_seconds", 
                    "avg_density", "avg_speed", "vehicle_types"
                ])

    def save(self, data: TrafficData):
        filename = self._get_filename()
        # Simple check if day changed to create new header if needed (though append works fine)
        if not os.path.exists(filename):
            self._ensure_file_exists()
            
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            dt_str = datetime.fromtimestamp(data.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                data.timestamp,
                dt_str,
                data.zone_id,
                data.duration_seconds,
                f"{data.avg_density:.2f}",
                f"{data.avg_speed:.2f}" if data.avg_speed else "",
                str(data.vehicle_types)
            ])
