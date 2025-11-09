import pandas as pd
import numpy as np

def generate_parking_data(n_rows=3000, seed=42):
    np.random.seed(seed)

    # Indian-specific parking areas
    areas = ["Airport", "Metro Station", "Shopping Mall", "Cricket Stadium", "Tech Park", "Railway Station", "City Center"]

    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="H")

    # Define event probabilities for each area (Stadium & City Center have higher chances)
    event_prob = {
        "Airport": 0.05,
        "Metro Station": 0.03,
        "Shopping Mall": 0.08,
        "Cricket Stadium": 0.20,
        "Tech Park": 0.10,
        "Railway Station": 0.05,
        "City Center": 0.12
    }

    # Randomly pick areas
    area_col = np.random.choice(areas, n_rows)

    # Assign nearby events based on area-specific probability
    nearby_events = np.array([
        np.random.choice([0, 1], p=[1 - event_prob[a], event_prob[a]]) for a in area_col
    ])

    data = pd.DataFrame({
        "timestamp": np.random.choice(timestamps, n_rows),
        "area_name": area_col,
        "traffic_level": np.random.randint(1, 11, n_rows),  # 1-10
        "nearby_events": nearby_events,
        "day_of_week": np.random.choice(range(7), n_rows),
        "total_slots": np.random.choice([50, 75, 100, 150, 200], n_rows, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    })

    # Simulate how many slots are occupied
    base = (data["traffic_level"] * 0.08 * data["total_slots"]).astype(int)
    event_increase = data["nearby_events"] * (0.2 * data["total_slots"]).astype(int)
    noise = np.random.randint(-5, 6, n_rows)
    occupied = (base + event_increase + noise).clip(0, data["total_slots"])
    data["occupied_slots"] = occupied
    data["available_slots"] = (data["total_slots"] - data["occupied_slots"]).clip(0)

    # Clean up timestamp
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.floor("H")
    data = data.sort_values("timestamp").reset_index(drop=True)

    return data

if __name__ == "__main__":
    df = generate_parking_data(n_rows=3000)
    df.to_csv("parking_data.csv", index=False)
    print("Saved updated parking_data.csv (rows = {})".format(len(df)))
    print("Sample rows:")
    print(df.head())
