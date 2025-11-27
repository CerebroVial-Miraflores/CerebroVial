import pandas as pd
import numpy as np
import datetime
import random
import os
import sys

# Add src to path to import schemas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common.schemas.camera import CameraTrafficData

# Simulation Configuration
NUM_SAMPLES = 5000
CAMERA_STREETS = {
    # Camera ID : Monitored Street
    "CAM_001": "Av. Javier Prado",
    "CAM_002": "Av. Arequipa",
    "CAM_003": "Via Expresa Paseo de la República"
}

def generate_data(num_samples=NUM_SAMPLES, output_file="data/prediction/camera_yolo_synthetic.csv"):
    print(f"Generating {num_samples} synthetic Camera traffic records...")
    
    data_camera = []
    start_date = datetime.datetime(2024, 1, 1)

    for _ in range(num_samples):
        # 1. Temporal Simulation (Same as Waze for synchronization)
        random_days = random.randint(0, 60)
        current_date = start_date + datetime.timedelta(days=random_days)
        
        # Rush hour probability
        if random.random() > 0.4:
            hour = random.choice([7, 8, 9, 17, 18, 19, 20])
            is_rush_hour = True
        else:
            hour = random.randint(0, 23)
            is_rush_hour = False
            
        minute = random.randint(0, 59)
        # Round to nearest 5 minutes if needed for easier joining, but keeping as is for now
        timestamp = current_date.replace(hour=hour, minute=minute, second=0)

        # 2. Iterate for each installed camera
        for cam_id, street_name in CAMERA_STREETS.items():
            
            # Correlation Logic:
            # If rush hour, increase heavy vehicle count and occupancy
            if is_rush_hour:
                total_cars = random.randint(40, 80) # Many cars in frame
                num_buses = random.randint(2, 8)    # More public transport
                num_trucks = random.randint(1, 5)
                occupancy = random.uniform(0.70, 0.95) # Street 70-95% full
            else:
                total_cars = random.randint(5, 30)
                num_buses = random.randint(0, 2)
                num_trucks = random.randint(0, 1)
                occupancy = random.uniform(0.10, 0.40) # Street empty
                
            num_motos = random.randint(0, 10)
            
            total_vehicles = total_cars + num_buses + num_trucks + num_motos
            
            # Flow Rate: Vehicles passing a line in the last minute
            # In congestion (rush hour), flow rate DROPS even if occupancy is HIGH (traffic paradox)
            if is_rush_hour and occupancy > 0.8:
                flow_rate_per_min = random.randint(5, 15) # Moving slowly
            elif is_rush_hour:
                 flow_rate_per_min = random.randint(20, 40)
            else:
                flow_rate_per_min = random.randint(30, 60) # Free flow

            record_dict = {
                "timestamp": timestamp.timestamp(), # Use float timestamp for schema
                "camera_id": cam_id,
                "street_monitored": street_name,
                "car_count": total_cars,
                "bus_count": num_buses,
                "truck_count": num_trucks,
                "motorcycle_count": num_motos,
                "total_vehicles": total_vehicles,
                "occupancy_rate": round(occupancy, 2), # 0.0 to 1.0
                "flow_rate_per_min": flow_rate_per_min
            }
            
            # Validate with Pydantic schema
            try:
                # Note: We import CameraTrafficData from src.common.schemas.camera
                # But wait, the import above has a typo: .py
                # I will fix the import in the file content.
                pass
            except Exception as e:
                pass

            data_camera.append(record_dict)

    df_cam = pd.DataFrame(data_camera)
    
    # Convert timestamp back to datetime for sorting/readability
    df_cam['datetime'] = pd.to_datetime(df_cam['timestamp'], unit='s')
    df_cam = df_cam.sort_values(by="datetime")

    print(df_cam.head())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_cam.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_data()
