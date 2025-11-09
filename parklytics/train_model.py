import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

def _safe_read_events(events_csv: str):
    try:
        ev = pd.read_csv(events_csv)
        return ev
    except Exception:
        return pd.DataFrame(columns=["event_id","area_name","event_name","date","start_time","end_time","impact_level","notes"])

def _safe_read_bookings(bookings_csv: str):
    try:
        bk = pd.read_csv(bookings_csv)
        return bk
    except Exception:
        return pd.DataFrame(columns=["booking_id","created_at","user_name","area_name","date","start_time","duration_mins",
                                     "slot_type","quoted_price","paid_amount","payment_status","qr_payload","total_slots","predicted_available"])

def _time_str_to_hour(tstr: str) -> int:
    tstr = str(tstr).strip()
    # supports "HH:MM" or "HH:MM AM/PM"
    try:
        if "AM" in tstr or "PM" in tstr:
            from datetime import datetime
            tm = datetime.strptime(tstr, "%I:%M %p").time()
        else:
            from datetime import datetime
            tm = datetime.strptime(tstr, "%H:%M").time()
        return tm.hour
    except Exception:
        return 0

def _enrich_with_events_and_bookings(df: pd.DataFrame, events_csv: str, bookings_csv: str) -> pd.DataFrame:
    """
    Adjusts traffic proxy using simple signals from events and bookings:
    - If any event overlaps the (area, hour) bucket, bump traffic.
    - Compute recent booking density per (area, hour) and bump traffic further.
    The model still predicts available_slots; we overwrite 'nearby_events' when event present,
    and adjust 'traffic_level' to a bounded value 1..10.
    """
    if df.empty:
        return df.copy()

    base = df.copy()
    base["hour"] = base["timestamp"].dt.hour

    # Start with existing values
    traffic = base["traffic_level"].astype(int).clip(1, 10)
    events_df = _safe_read_events(events_csv)
    bookings_df = _safe_read_bookings(bookings_csv)

    # Event signal: mark nearby_events = 1 if any event exists same area & date & hour overlap
    if not events_df.empty:
        # create per (area,date,hour) set where events exist
        ev = events_df.copy()
        # naive mapping of start_time/end_time to hours covered
        def hours_covered(row):
            try:
                s = row["start_time"]
                e = row["end_time"]
                sh = _time_str_to_hour(s)
                eh = _time_str_to_hour(e)
                if eh <= sh:
                    eh = sh  # avoid negative
                return list(range(sh, eh+1))
            except Exception:
                return []
        ev["hours"] = ev.apply(hours_covered, axis=1)
        marks = set()
        for _, r in ev.iterrows():
            d = str(r["date"])
            a = r["area_name"]
            imp = int(r.get("impact_level", 0))
            hrs = r["hours"]
            for h in hrs:
                marks.add((a, d, h, imp))

        # apply to base
        ev_indicator = []
        ev_bump = np.zeros(len(base), dtype=int)
        for i, r in base.iterrows():
            key_hits = [m for m in marks if m[0] == r["area_name"] and m[1] == str(r["timestamp"].date()) and m[2] == int(r["hour"])]
            if key_hits:
                ev_indicator.append(1)
                # bump traffic proportional to max impact
                max_imp = max(k[3] for k in key_hits)
                ev_bump[i] = [0,1,2,3][max_imp] if max_imp <= 3 else 3
            else:
                ev_indicator.append(0)
        base["nearby_events"] = ev_indicator
        traffic = (traffic + ev_bump).clip(1, 10)

    # Booking density signal: last 14 days approx — here we only have historical; simple density by (area, hour)
    if not bookings_df.empty:
        b = bookings_df.copy()
        b["hour"] = b["start_time"].apply(lambda s: _time_str_to_hour(s))
        dens = b.groupby(["area_name","hour"]).size().reset_index(name="density")
        dens_dict = {(row["area_name"], int(row["hour"])): int(row["density"]) for _, row in dens.iterrows()}
        dens_bump = np.zeros(len(base), dtype=int)
        for i, r in base.iterrows():
            dkey = (r["area_name"], int(r["hour"]))
            dval = dens_dict.get(dkey, 0)
            # scale to 0..3
            if dval >= 15:
                dens_bump[i] = 3
            elif dval >= 8:
                dens_bump[i] = 2
            elif dval >= 3:
                dens_bump[i] = 1
        traffic = (traffic + dens_bump).clip(1, 10)

    base["traffic_level"] = traffic.astype(int)
    base = base.drop(columns=["hour"], errors="ignore")
    return base

def train_and_save_model(csv_path="parking_data.csv",
                         model_path="parking_model.pkl",
                         meta_path="model_meta.json",
                         enriched=False,
                         events_csv="events.csv",
                         bookings_csv="bookings.csv"):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print("Loaded dataset:", df.shape)

    if enriched:
        print("Enriching training data with events & bookings signals...")
        df = _enrich_with_events_and_bookings(df, events_csv=events_csv, bookings_csv=bookings_csv)

    # One-hot encode area names for model training
    X = pd.get_dummies(df[["area_name", "traffic_level", "nearby_events", "day_of_week", "total_slots"]], drop_first=True)
    y = df["available_slots"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
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
        "note": "Trained on Indian parking data with event+booking influence" if enriched else "Trained on Indian parking data"
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    # Default: enriched training for admin refresh
    train_and_save_model(enriched=True)
