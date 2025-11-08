import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def train_and_save_model(csv_path="parking_data.csv", model_path="parking_model.pkl", meta_path="model_meta.json"):
    df = pd.read_csv(csv_path)
    print("Loaded dataset:", df.shape)

    # One-hot encode area names for model training
    X = pd.get_dummies(df[["area_name", "traffic_level", "nearby_events", "day_of_week", "total_slots"]], drop_first=True)
    y = df["available_slots"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model trained successfully — MAE: {mae:.2f}")

    # Save model and metadata
    joblib.dump((model, X.columns.tolist()), model_path)
    print(f"Model saved to {model_path}")

    meta = {
        "mae": float(mae),
        "n_samples": int(len(df)),
        "features": X.columns.tolist(),
        "note": "Trained on Indian parking data with event influence"
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    train_and_save_model()