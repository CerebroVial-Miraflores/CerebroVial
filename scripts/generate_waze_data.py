import pandas as pd
import numpy as np
import datetime
import random
import os
import sys

# Add src to path to import schemas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common.schemas.waze import WazeTrafficData

# Simulation Configuration
NUM_SAMPLES = 5000  # Number of rows to generate
CITY = "Lima"
STREETS = [
    "Av. Javier Prado", "Via Expresa Paseo de la República", 
    "Av. Arequipa", "Av. La Marina", "Panamericana Sur", 
    "Av. Brasil", "Av. Faucett", "Circuito de Playas"
]

# Define road types (Typical Waze IDs)
# 1: Streets, 2: Primary Street, 3: Freeways, etc.
ROAD_TYPES = {
    "Av. Javier Prado": 3,
    "Via Expresa Paseo de la República": 3,
    "Panamericana Sur": 3,
    "Av. Arequipa": 2,
    "Av. La Marina": 2,
    "Av. Brasil": 2,
    "Av. Faucett": 2,
    "Circuito de Playas": 4
}

def generate_data(num_samples=NUM_SAMPLES, output_file="data/prediction/waze_traffic_synthetic.csv"):
    print(f"Generating {num_samples} synthetic Waze traffic records...")
    
    data = []
    start_date = datetime.datetime(2024, 1, 1)

    for _ in range(num_samples):
        # 1. Simulate Date and Time
        # Add bias to have more data during rush hours (7-9am, 5-8pm)
        random_days = random.randint(0, 60)
        current_date = start_date + datetime.timedelta(days=random_days)
        
        # Rush hour probability
        if random.random() > 0.4:
            hour = random.choice([7, 8, 9, 17, 18, 19, 20]) # Rush hour
            is_rush_hour = True
        else:
            hour = random.randint(0, 23) # Off-peak hour
            is_rush_hour = False
            
        minute = random.randint(0, 59)
        timestamp = current_date.replace(hour=hour, minute=minute)
        
        # 2. Select Street
        street = random.choice(STREETS)
        road_type = ROAD_TYPES[street]
        
        # 3. Simulate Congestion (Waze Jam Level 0-5)
        # If rush hour, level tends to be higher (3, 4, 5)
        if is_rush_hour:
            level = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
            speed_kmh = random.uniform(5, 20) # Slow
            delay = random.randint(300, 1200) # High delay (seconds)
        else:
            level = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
            speed_kmh = random.uniform(30, 60) # Fast
            delay = random.randint(0, 300) # Low delay
            
        # Correct logic: If level is 5 (blocked), speed is very low
        if level == 5:
            speed_kmh = random.uniform(0, 5)
            
        # 4. Segment length (meters)
        length = random.randint(500, 3000)
        
        # 5. Base coordinates (Simple simulation around Lima)
        base_lat = -12.0464
        base_lon = -77.0428
        lat = base_lat + random.uniform(-0.05, 0.05)
        lon = base_lon + random.uniform(-0.05, 0.05)

        # 6. Build record
        record_dict = {
            "uuid": f"waze_{random.randint(100000, 999999)}",
            "timestamp": timestamp.timestamp(), # Use float timestamp for schema
            "city": CITY,
            "street": street,
            "road_type": road_type,
            "location_lat": round(lat, 6),
            "location_lon": round(lon, 6),
            "length_meters": length,
            "speed_kmh": round(speed_kmh, 1),
            "delay_seconds": delay,
            "level": level 
        }
        
        # Validate with Pydantic schema
        try:
            waze_data = WazeTrafficData(**record_dict)
            data.append(waze_data.model_dump())
        except Exception as e:
            print(f"Error validating record: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert timestamp back to datetime for readability if needed, but schema uses float
    # Let's keep it as is or add a readable column
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Sort by date
    df = df.sort_values(by="datetime")

    # Show first rows
    print(df.head())

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_data()
