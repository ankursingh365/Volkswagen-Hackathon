import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import json
import qrcode
from io import BytesIO
from datetime import datetime, timedelta, date, time
from data_generator import generate_parking_data
import warnings
warnings.filterwarnings("ignore")

import os
import uuid

st.set_page_config(page_title="Parklytics Prototype", layout="wide")

# ------------------------- CONSTANTS -------------------------
DATA_CSV = "parking_data.csv"
BOOKINGS_CSV = "bookings.csv"
EVENTS_CSV = "events.csv"
PRICING_JSON = "pricing_policy.json"
MODEL_PKL = "parking_model.pkl"
MODEL_META = "model_meta.json"
AREA_CAP_JSON = "area_capacity.json"


# Indian areas (keep same style as before)
indian_areas = ["Airport", "Metro Station", "Shopping Mall", "Cricket Stadium",
                "Tech Park", "Railway Station", "City Center"]

DURATION_OPTIONS = ["30 mins", "1 hour", "2 hours", "3 hours", "4 hours"]
DURATION_TO_MINS = {"30 mins": 30, "1 hour": 60, "2 hours": 120, "3 hours": 180, "4 hours": 240}

SLOT_TYPES = ["near_exit", "far_exit"]
SLOT_TYPE_MULT_DEFAULT = {"near_exit": 1.10, "far_exit": 0.95}
EVENT_MULTIPLIERS_DEFAULT = [1.00, 1.10, 1.25, 1.40]  # impact 0..3

# ------------------------- HELPERS (LOAD/SAVE) -------------------------
@st.cache_data
def load_or_create_data():
    try:
        df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    except FileNotFoundError:
        df = generate_parking_data(n_rows=3000)
        df.to_csv(DATA_CSV, index=False)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('H')
    # Normalize area names to our set (if any drift)
    if not set(df['area_name']).issubset(set(indian_areas)):
        df['area_name'] = np.random.choice(indian_areas, len(df))
    return df

@st.cache_data
def load_model_and_meta():
    try:
        model, cols = joblib.load(MODEL_PKL)
        with open(MODEL_META, "r") as f:
            meta = json.load(f)
    except Exception:
        model, cols = None, []
        meta = {"mae": None}
    return model, cols, meta

@st.cache_data
def load_bookings():
    if os.path.exists(BOOKINGS_CSV):
        df = pd.read_csv(BOOKINGS_CSV)
        # parse times safely if present
        if "created_at" in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df
    cols = ["booking_id","created_at","user_name","area_name","date","start_time","duration_mins",
            "slot_type","quoted_price","paid_amount","payment_status","qr_payload",
            "total_slots","predicted_available"]
    return pd.DataFrame(columns=cols)

@st.cache_data
def load_events():
    if os.path.exists(EVENTS_CSV):
        df = pd.read_csv(EVENTS_CSV)
        return df
    cols = ["event_id","area_name","event_name","date","start_time","end_time","impact_level","notes"]
    return pd.DataFrame(columns=cols)

@st.cache_data
def load_pricing_policy():
    if os.path.exists(PRICING_JSON):
        with open(PRICING_JSON, "r") as f:
            return json.load(f)
    # default policy
    policy = {
        "min_price_per_30m": 20.0,
        "price_per_head": 0.0,
        "price_per_hour": 0.0,
        "slot_type_multipliers": SLOT_TYPE_MULT_DEFAULT,
        "event_multiplier_per_impact": EVENT_MULTIPLIERS_DEFAULT,
        "traffic_multiplier_scale": 0.6  # contributes up to +0.6x
    }
    # write default for persistence
    with open(PRICING_JSON, "w") as f:
        json.dump(policy, f, indent=2)
    return policy

def save_df_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    # invalidate caches for readers of that file
    if path == BOOKINGS_CSV:
        load_bookings.clear()
    if path == EVENTS_CSV:
        load_events.clear()
    if path == DATA_CSV:
        load_or_create_data.clear()

def save_policy(policy: dict):
    with open(PRICING_JSON, "w") as f:
        json.dump(policy, f, indent=2)
    load_pricing_policy.clear()

@st.cache_data
def load_area_capacity(df_for_defaults: pd.DataFrame):
    import os, json
    if os.path.exists(AREA_CAP_JSON):
        with open(AREA_CAP_JSON, "r") as f:
            cap = json.load(f)
            return {k: int(v) for k, v in cap.items()}

    # infer defaults from df (mode of total_slots per area)
    caps = {}
    for a in indian_areas:
        sub = df_for_defaults[df_for_defaults["area_name"] == a]
        if not sub.empty:
            caps[a] = int(sub["total_slots"].mode()[0])
        else:
            caps[a] = 100
    with open(AREA_CAP_JSON, "w") as f:
        json.dump(caps, f, indent=2)
    return caps

def save_area_capacity(capacity_map: dict):
    with open(AREA_CAP_JSON, "w") as f:
        json.dump({k: int(v) for k, v in capacity_map.items()}, f, indent=2)
    load_area_capacity.clear()


def clear_all_caches():
    load_or_create_data.clear()
    load_model_and_meta.clear()
    load_events.clear()
    load_bookings.clear()
    load_pricing_policy.clear()

# ------------------------- CORE DOMAIN HELPERS -------------------------
def day_of_week_from_date(d: date) -> int:
    # Monday=0..Sunday=6 (aligns to your earlier format)
    return d.weekday()

def time_str_to_minutes(tstr: str) -> int:
    # supports "HH:MM" (24h) or "HH:MM AM/PM"
    tstr = tstr.strip()
    try:
        if "AM" in tstr or "PM" in tstr:
            tm = datetime.strptime(tstr, "%I:%M %p").time()
        else:
            tm = datetime.strptime(tstr, "%H:%M").time()
        return tm.hour * 60 + tm.minute
    except Exception:
        return 0

def minutes_to_time_str(m: int) -> str:
    h = (m // 60) % 24
    mi = m % 60
    return f"{h:02d}:{mi:02d}"

def overlap(a_start, a_dur, b_start, b_dur) -> bool:
    # all in minutes from midnight
    return not (a_start + a_dur <= b_start or b_start + b_dur <= a_start)

def find_events_for_slot(events_df: pd.DataFrame, area: str, d: date, start_minutes: int, dur_minutes: int):
    if events_df.empty:
        return []
    df = events_df.copy()
    df = df[(df["area_name"] == area) & (df["date"] == str(d))]
    hits = []
    for _, r in df.iterrows():
        ev_start = time_str_to_minutes(str(r["start_time"]))
        ev_end = time_str_to_minutes(str(r["end_time"]))
        ev_dur = max(0, ev_end - ev_start)
        if overlap(start_minutes, dur_minutes, ev_start, ev_dur):
            hits.append(r)
    return hits

def estimate_traffic_level(base_df: pd.DataFrame, bookings_df: pd.DataFrame, events_df: pd.DataFrame,
                           area: str, d: date, start_minutes: int, total_slots: int) -> int:
    # base hour from selected start time
    hour = (start_minutes // 60) % 24

    # Base from historical availability: lower availability => higher traffic
    # Compute avg availability by area & hour
    temp = base_df.copy()
    temp["hour"] = temp["timestamp"].dt.hour
    area_hour = temp[(temp["area_name"] == area) & (temp["hour"] == hour)]
    if area_hour.empty:
        base_avail = temp[temp["area_name"] == area]["available_slots"].mean()
    else:
        base_avail = area_hour["available_slots"].mean()
    if np.isnan(base_avail):
        base_avail = total_slots * 0.6
    demand = max(0.0, 1.0 - (base_avail / max(1, total_slots)))
    base_traffic = 1 + int(round(demand * 9))  # map to 1..10

    # Event impact
    hits = find_events_for_slot(events_df, area, d, start_minutes, 30)
    max_impact = 0
    for r in hits:
        try:
            imp = int(r["impact_level"])
            max_impact = max(max_impact, imp)
        except Exception:
            pass
    event_bump = [0, 1, 2, 3][max_impact] if max_impact <= 3 else 3

    # Recent bookings density (last 14 days same area/hour)
    recent_boost = 0
    if not bookings_df.empty:
        b = bookings_df.copy()
        # filter by area & start hour
        b["hour"] = b["start_time"].apply(lambda s: time_str_to_minutes(str(s)) // 60 if pd.notna(s) else None)
        b = b[(b["area_name"] == area) & (b["hour"] == hour)]
        # consider "recent" by created_at (if present)
        if "created_at" in b.columns:
            try:
                cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=14)
                b_recent = b[b["created_at"] >= cutoff]
                density = len(b_recent)
            except Exception:
                density = len(b)
        else:
            density = len(b)
        # scale 0..3 (simple heuristic)
        if density >= 15:
            recent_boost = 3
        elif density >= 8:
            recent_boost = 2
        elif density >= 3:
            recent_boost = 1

    traffic_level = min(10, max(1, base_traffic + event_bump + recent_boost))
    return int(traffic_level)

def quote_price(policy: dict, predicted_avail: int, total_slots: int,
                traffic_level: int, impact_level: int, slot_type: str, duration_label: str):
    base = float(policy.get("min_price_per_30m", 20.0))
    traffic_scale = float(policy.get("traffic_multiplier_scale", 0.6))
    event_mults = policy.get("event_multiplier_per_impact", EVENT_MULTIPLIERS_DEFAULT)
    slot_mults = policy.get("slot_type_multipliers", SLOT_TYPE_MULT_DEFAULT)

    traffic_norm = (traffic_level - 1) / 9.0
    demand = max(0.0, 1.0 - (predicted_avail / max(1.0, float(total_slots))))
    event_mult = event_mults[min(max(0, int(impact_level)), len(event_mults)-1)]
    slot_mult = slot_mults.get(slot_type, 1.0)

    duration_factor = {"30 mins": 1.0, "1 hour": 1.8, "2 hours": 3.2, "3 hours": 4.5, "4 hours": 5.5}[duration_label]

    price = base * (1 + traffic_scale * traffic_norm + 0.4 * demand) * event_mult * slot_mult * duration_factor
    price = max(price, base)
    return round(float(price), 2)

def predict_availability(model, model_cols, area: str, day_of_week: int,
                         traffic_level: int, nearby_events: int, total_slots: int):
    # Build row aligned to model_cols
    input_dict = {
        "traffic_level": traffic_level,
        "nearby_events": nearby_events,
        "day_of_week": day_of_week,
        "total_slots": total_slots
    }
    row = pd.DataFrame(columns=model_cols)
    row.loc[0] = 0
    for k in ["traffic_level", "nearby_events", "day_of_week", "total_slots"]:
        if k in row.columns:
            row.at[0, k] = input_dict[k]
    col_area = f"area_name_{area}"
    if col_area in row.columns:
        row.at[0, col_area] = 1
    pred = model.predict(row.fillna(0))[0]
    return int(max(0, round(pred)))

def add_booking_row(bookings_df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df = bookings_df.copy()
    df.loc[len(df)] = row
    return df

def ensure_session():
    if "current_booking_id" not in st.session_state:
        st.session_state.current_booking_id = None

# ------------------------- LOAD ALL DATA -------------------------
df = load_or_create_data()
model, model_cols, meta = load_model_and_meta()
bookings_df = load_bookings()
events_df = load_events()
policy = load_pricing_policy()
area_capacity = load_area_capacity(df)
ensure_session()

# ------------------------- HEADER -------------------------
st.title("Parklytics — Predictive Parking Marketplace")
role = st.sidebar.radio("Role", ["User", "Admin"])

# Top metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total records (simulated)", len(df))
with col2:
    st.metric("Avg available slots", round(df['available_slots'].mean(), 1))
with col3:
    mae_text = f"{meta.get('mae'):.2f}" if meta.get('mae') is not None else "N/A (train model)"
    st.metric("Model MAE (lower is better)", mae_text)

# --------------------- COMMON DASHBOARD (LEFT) --------------------------
left, right = st.columns([2, 1])
with left:
    st.subheader("Dashboard: Availability & Trends")

    # # 1) Time-series
    # ts = df.set_index('timestamp').groupby('area_name')['available_slots'].resample('D').mean().reset_index()
    # if not ts.empty:
    #     fig_ts = px.line(ts, x='timestamp', y='available_slots', color='area_name',
    #                      title="Daily Average Available Slots by Area")
    #     st.plotly_chart(fig_ts, use_container_width=True)
        # 1) Time-series (filterable by area)
    st.subheader("Available Slots vs Timestamp")

    # Dropdown to choose a single area or all
    ts_area = st.selectbox(
        "Select Parking Area",
        ["All Areas"] + indian_areas,
        index=0,
        key="ts_area_filter"
    )

    # Daily mean available slots per area
    ts_daily = (
        df[["timestamp", "area_name", "available_slots"]]
        .set_index("timestamp")
        .groupby("area_name")["available_slots"]
        .resample("D").mean()
        .reset_index()
    )

    if not ts_daily.empty:
        if ts_area == "All Areas":
            fig_ts = px.line(
                ts_daily,
                x="timestamp",
                y="available_slots",
                color="area_name",
                title="Daily Average Available Slots by Area (All)"
            )
        else:
            ts_one = ts_daily[ts_daily["area_name"] == ts_area].copy()
            fig_ts = px.line(
                ts_one,
                x="timestamp",
                y="available_slots",
                title=f"Daily Average Available Slots — {ts_area}"
            )

        st.plotly_chart(fig_ts, use_container_width=True)


    # 2) Area comparison bar
    area_avail = df.groupby("area_name")["available_slots"].mean().reset_index().sort_values("available_slots", ascending=False)
    fig_area = px.bar(area_avail, x='area_name', y='available_slots',
                      title="Average Available Slots by Area")
    st.plotly_chart(fig_area, use_container_width=True)

    # 3) Heatmap
    st.subheader("Heatmap: Avg Availability by Hour & Area")
    df['hour'] = df['timestamp'].dt.hour
    heat = df.groupby(['area_name', 'hour'])['available_slots'].mean().reset_index()
    heat_pivot = heat.pivot(index='area_name', columns='hour', values='available_slots').fillna(0)
    fig_heat = px.imshow(
        heat_pivot,
        labels=dict(x="Hour of Day", y="Area", color="Avg Available Slots"),
        x=list(heat_pivot.columns),
        y=list(heat_pivot.index),
        aspect="auto",
        title="Heatmap: Avg Available Slots (Area vs Hour)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 4) Event Impact Analysis
    st.subheader("Event Impact on Parking Availability")
    event_impact = df.groupby(["area_name", "nearby_events"])["available_slots"].mean().reset_index()
    event_impact["event_status"] = event_impact["nearby_events"].map({0: "No Event", 1: "Event"})
    fig_event = px.bar(
        event_impact, x="area_name", y="available_slots",
        color="event_status", barmode="group", text_auto=".2f",
        title="Average Available Slots: Event vs No Event"
    )
    st.plotly_chart(fig_event, use_container_width=True)
    st.caption("Shows how nearby events reduce parking availability —> especially at Stadiums or City Centers.")

# --------------------- USER DASHBOARD (RIGHT) --------------------------
if role == "User":
    with right:
        st.subheader("Book Parking & Get Price")

        # Inputs
        user_name = st.text_input("Driver name", value="Demo User")
        sel_date = st.date_input("Select date", value=datetime.now().date(), min_value=datetime.now().date())
        # times in 15-min steps for the selected date; for prototype allow next 1..8 hours
        now = datetime.now()
        if sel_date == now.date():
            base_start = now + timedelta(minutes=(15 - now.minute % 15))
        else:
            base_start = datetime.combine(sel_date, time(8, 0))  # start at 8:00 for future day
        time_choices = [(base_start + timedelta(minutes=15 * i)).strftime("%H:%M") for i in range(1, 17)]
        start_time_label = st.selectbox("Start time", time_choices)

        day_idx = day_of_week_from_date(sel_date)
        day_label = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_idx]
        st.caption(f"Day of week: **{day_label}**")

        area = st.selectbox("Select Area", indian_areas)
        slot_type = st.selectbox("Slot type", SLOT_TYPES)
        #total_slots = st.selectbox("Total slots at selected area", sorted(df["total_slots"].unique()))
        # Automatically infer total_slots based on selected area
        area_slots = int(area_capacity.get(area, 100))
        st.caption(f"Total slots for {area}: **{area_slots}**")
        total_slots = area_slots

        duration = st.selectbox("Duration", DURATION_OPTIONS)

        # Event + Traffic + Prediction
        start_minutes = time_str_to_minutes(start_time_label)
        dur_minutes = DURATION_TO_MINS[duration]

        hits = find_events_for_slot(events_df, area, sel_date, start_minutes, dur_minutes)
        impact_level = 0
        if hits:
            impact_level = int(max(min(int(x["impact_level"]), 3) for _, x in pd.DataFrame(hits).iterrows()))
        nearby_events = 1 if impact_level > 0 else 0

        traffic_level = estimate_traffic_level(df, bookings_df, events_df, area, sel_date, start_minutes, total_slots)

        # Prediction (needs model)
        if model is None:
            st.warning("Model not available. Please run Admin → Retrain model.")
            predicted_avail = int(round(df[df["area_name"] == area]["available_slots"].mean() if (df["area_name"] == area).any() else total_slots * 0.6))
        else:
            predicted_avail = predict_availability(model, model_cols, area, day_idx, traffic_level, nearby_events, total_slots)

        st.success(f"Predicted available slots at start: **{predicted_avail}** / {total_slots}")

        # Pricing
        price_quote = quote_price(policy, predicted_avail, total_slots, traffic_level, impact_level, slot_type, duration)
        st.write(f"**Suggested Price:** ₹{price_quote}")

        # QR & Booking
        if st.button("Book & Generate QR"):
            booking_id = str(uuid.uuid4())
            payload = {
                "booking_id": booking_id,
                "name": user_name,
                "area": area,
                "slot_type": slot_type,
                "date": str(sel_date),
                "start_time": start_time_label,
                "duration_mins": dur_minutes,
                "price": price_quote
            }
            qr_payload = json.dumps(payload)
            qr = qrcode.QRCode(box_size=6, border=2)
            qr.add_data(qr_payload)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.image(buf, caption="Booking QR (mock). Scan to view booking details.")

            # Record booking with PENDING payment
            new_row = {
                "booking_id": booking_id,
                "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "user_name": user_name,
                "area_name": area,
                "date": str(sel_date),
                "start_time": start_time_label,
                "duration_mins": dur_minutes,
                "slot_type": slot_type,
                "quoted_price": price_quote,
                "paid_amount": 0.0,
                "payment_status": "PENDING",
                "qr_payload": qr_payload,
                "total_slots": int(total_slots),
                "predicted_available": int(predicted_avail)
            }
            updated = add_booking_row(bookings_df, new_row)
            save_df_to_csv(updated, BOOKINGS_CSV)
            st.session_state.current_booking_id = booking_id
            st.success(f"Booking created with ID: {booking_id}. Payment pending.")

        if st.session_state.current_booking_id:
            st.info(f"Current Booking ID: {st.session_state.current_booking_id}")
            if st.button("Mark Paid (Dummy)"):
                bdf = load_bookings()
                if not bdf.empty:
                    idx = bdf.index[bdf["booking_id"] == st.session_state.current_booking_id].tolist()
                    if idx:
                        i = idx[0]
                        bdf.at[i, "payment_status"] = "SUCCESS"
                        bdf.at[i, "paid_amount"] = bdf.at[i, "quoted_price"]
                        save_df_to_csv(bdf, BOOKINGS_CSV)
                        st.success("Payment marked as SUCCESS.")
                    else:
                        st.warning("Booking not found to update.")
                else:
                    st.warning("No bookings found.")
                # refresh
                bookings_df = load_bookings()

# --------------------- ADMIN DASHBOARD (RIGHT) --------------------------
if role == "Admin":
    with right:
        st.subheader("Admin Panel")

        # --- Pricing Policy ---
        st.markdown("### Pricing Policy")
        with st.form("pricing_form"):
            min_price = st.number_input("Minimum price per 30 minutes (₹)", min_value=0.0, value=float(policy.get("min_price_per_30m", 20.0)), step=1.0)
            price_per_head = st.number_input("Price per head (optional)", min_value=0.0, value=float(policy.get("price_per_head", 0.0)), step=1.0)
            price_per_hour = st.number_input("Price per hour (optional)", min_value=0.0, value=float(policy.get("price_per_hour", 0.0)), step=1.0)
            traffic_scale = st.slider("Traffic multiplier max scale (+0..+0.9)", 0.0, 0.9, float(policy.get("traffic_multiplier_scale", 0.6)), 0.05)
            near_mult = st.number_input("Near-exit multiplier", min_value=0.5, max_value=2.0, value=float(policy.get("slot_type_multipliers", SLOT_TYPE_MULT_DEFAULT).get("near_exit", 1.10)), step=0.05)
            far_mult = st.number_input("Far-exit multiplier", min_value=0.5, max_value=2.0, value=float(policy.get("slot_type_multipliers", SLOT_TYPE_MULT_DEFAULT).get("far_exit", 0.95)), step=0.05)
            ev0 = st.number_input("Event multiplier (impact 0)", min_value=0.5, max_value=3.0, value=float(policy.get("event_multiplier_per_impact", EVENT_MULTIPLIERS_DEFAULT)[0]), step=0.05)
            ev1 = st.number_input("Event multiplier (impact 1)", min_value=0.5, max_value=3.0, value=float(policy.get("event_multiplier_per_impact", EVENT_MULTIPLIERS_DEFAULT)[1]), step=0.05)
            ev2 = st.number_input("Event multiplier (impact 2)", min_value=0.5, max_value=3.0, value=float(policy.get("event_multiplier_per_impact", EVENT_MULTIPLIERS_DEFAULT)[2]), step=0.05)
            ev3 = st.number_input("Event multiplier (impact 3)", min_value=0.5, max_value=3.0, value=float(policy.get("event_multiplier_per_impact", EVENT_MULTIPLIERS_DEFAULT)[3]), step=0.05)
            save_btn = st.form_submit_button("Save Pricing Policy")
            if save_btn:
                new_policy = {
                    "min_price_per_30m": float(min_price),
                    "price_per_head": float(price_per_head),
                    "price_per_hour": float(price_per_hour),
                    "slot_type_multipliers": {"near_exit": float(near_mult), "far_exit": float(far_mult)},
                    "event_multiplier_per_impact": [float(ev0), float(ev1), float(ev2), float(ev3)],
                    "traffic_multiplier_scale": float(traffic_scale),
                }
                save_policy(new_policy)
                policy = load_pricing_policy()
                st.success("Pricing policy saved.")

        st.markdown("---")

        st.markdown("---")
        st.markdown("### Area Capacity (per location)")

        # Show inputs in two columns for compact layout
        colL, colR = st.columns(2)

        with st.form("area_capacity_form"):
            new_caps = {}
            for i, area_name in enumerate(indian_areas):
                default_val = int(area_capacity.get(area_name, 100))
                ctx = colL if i % 2 == 0 else colR
                with ctx:
                    new_caps[area_name] = st.number_input(
                        f"{area_name} slots",
                        min_value=10, max_value=2000,
                        value=default_val, step=5, key=f"cap_{area_name}"
                    )
            save_caps = st.form_submit_button("Save Capacities")
            if save_caps:
                save_area_capacity(new_caps)
                area_capacity = load_area_capacity(df)
                st.success("Area capacities saved.")


        # --- Event Management ---
        st.markdown("### Event Management")
        if not events_df.empty:
            st.dataframe(events_df.sort_values(["date","start_time"]))
        with st.form("add_event"):
            area_e = st.selectbox("Area", indian_areas, key="ev_area")
            ev_name = st.text_input("Event name", value="Special Event")
            ev_date = st.date_input("Event date", value=datetime.now().date())
            ev_start = st.text_input("Start time (HH:MM)", value="17:00")
            ev_end = st.text_input("End time (HH:MM)", value="20:00")
            ev_imp = st.selectbox("Impact level (0..3)", [0,1,2,3], index=1)
            ev_notes = st.text_input("Notes", value="")
            add_ev = st.form_submit_button("Add / Update Event")
            if add_ev:
                new_event = {
                    "event_id": str(uuid.uuid4()),
                    "area_name": area_e,
                    "event_name": ev_name,
                    "date": str(ev_date),
                    "start_time": ev_start,
                    "end_time": ev_end,
                    "impact_level": int(ev_imp),
                    "notes": ev_notes
                }
                e_df = load_events()
                e_df = pd.concat([e_df, pd.DataFrame([new_event])], ignore_index=True)
                save_df_to_csv(e_df, EVENTS_CSV)
                events_df = load_events()
                st.success("Event saved.")

        # Delete event by id
        if not events_df.empty:
            st.markdown("#### Delete Event")
            del_id = st.selectbox("Select event_id to delete", [""] + events_df["event_id"].tolist())
            if del_id and st.button("Delete selected event"):
                edf = load_events()
                edf = edf[edf["event_id"] != del_id]
                save_df_to_csv(edf, EVENTS_CSV)
                events_df = load_events()
                st.success("Event deleted.")

        st.markdown("---")

        # --- Transactions / Ledger ---
        st.markdown("### Transactions / Bookings")
        bdf = load_bookings()
        if not bdf.empty:
            st.dataframe(bdf.sort_values("created_at", ascending=False))
            # Simple aggregates
            paid = bdf[bdf["payment_status"] == "SUCCESS"]
            total_rev = float(paid["paid_amount"].sum()) if not paid.empty else 0.0
            avg_ticket = float(paid["paid_amount"].mean()) if not paid.empty else 0.0
            success_rate = (len(paid) / len(bdf)) * 100.0 if len(bdf) else 0.0
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Revenue (₹)", f"{total_rev:.2f}")
            with c2:
                st.metric("Avg Ticket (₹)", f"{avg_ticket:.2f}")
            with c3:
                st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.info("No bookings yet.")

        st.markdown("---")

        # --- Retrain Model ---
        st.markdown("### Model Refresh (learn from events & bookings)")
        st.caption("Recomputes traffic signals and retrains model using events and recent booking density.")
        if st.button("Retrain model now"):
            # We call train_model.py’s function directly to keep logic centralized
            try:
                from train_model import train_and_save_model
                train_and_save_model(csv_path=DATA_CSV, model_path=MODEL_PKL, meta_path=MODEL_META, enriched=True,
                                     events_csv=EVENTS_CSV, bookings_csv=BOOKINGS_CSV)
                clear_all_caches()
                # reload
                model, model_cols, meta = load_model_and_meta()
                st.success(f"Model retrained. New MAE: {meta.get('mae'):.2f}" if meta.get("mae") is not None else "Model retrained.")
            except Exception as e:
                st.error(f"Retrain failed: {e}")

# --------------------- FOOTER NOTES --------------------------
st.markdown("----")
st.write("**Notes:** Prototype demonstrates Admin/User roles, bookings with dummy QR payments, events, pricing policy, model retraining with event & booking signals, and predictive availability.")
