import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import json
import qrcode
from io import BytesIO
from datetime import datetime, timedelta
from data_generator import generate_parking_data
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Parklytics Prototype", layout="wide")

@st.cache_data
def load_or_create_data():
    try:
        df = pd.read_csv("parking_data.csv", parse_dates=["timestamp"])
    except FileNotFoundError:
        df = generate_parking_data(n_rows=3000)
        df.to_csv("parking_data.csv", index=False)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('H')
    return df

@st.cache_data
def load_model_and_meta():
    try:
        model, cols = joblib.load("parking_model.pkl")
        with open("model_meta.json", "r") as f:
            meta = json.load(f)
    except Exception:
        model, cols = None, []
        meta = {"mae": None}
    return model, cols, meta

df = load_or_create_data()
model, model_cols, meta = load_model_and_meta()

# Updated Indian location names
indian_areas = ["Airport", "Metro Station", "Shopping Mall", "Cricket Stadium",
                "Tech Park", "Railway Station", "City Center"]

if not set(df['area_name']).issubset(set(indian_areas)):
    df['area_name'] = np.random.choice(indian_areas, len(df))

# Layout
st.title("Parklytics — Prototype: Predictive Parking Marketplace")
st.markdown("**Team:** Parklytics  •  **Region:** India 🇮🇳  •  **Features:** predictive ML, heatmap, dynamic pricing, event analysis, QR booking & pre-booking")

# Top metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total records (simulated)", len(df))
with col2:
    st.metric("Avg available slots", round(df['available_slots'].mean(), 1))
with col3:
    mae_text = f"{meta.get('mae'):.2f}" if meta.get('mae') is not None else "N/A (train model)"
    st.metric("Model MAE (lower is better)", mae_text)

# Split layout
left, right = st.columns([2, 1])

# --------------------- LEFT: DASHBOARD --------------------------
with left:
    st.subheader("Dashboard: Availability & Trends")

    # 1) Time-series
    ts = df.set_index('timestamp').groupby('area_name')['available_slots'].resample('D').mean().reset_index()
    if not ts.empty:
        fig_ts = px.line(ts, x='timestamp', y='available_slots', color='area_name',
                         title="Daily Average Available Slots by Area")
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

# --------------------- RIGHT: PREDICTION & BOOKING --------------------------
with right:
    st.subheader("Predict Parking Availability")

    # Inputs
    traffic = st.slider("Traffic level (1 - 10)", 1, 10, 5)
    event = st.selectbox("Nearby event?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    day = st.selectbox("Day of week", list(range(7)),
                       format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    area = st.selectbox("Select Area", indian_areas)
    total_slots = st.selectbox("Total slots at selected area", sorted(df["total_slots"].unique()))
    duration = st.selectbox("Required parking duration", ["30 mins", "1 hour", "2 hours", "3 hours", "4 hours"])

    # Pre-booking time (new feature)
    st.markdown("**Select Pre-booking Time (Up to 4 hours in advance)**")
    now = datetime.now()
    future_times = [(now + timedelta(hours=i)).strftime("%I:%M %p") for i in range(1, 5)]
    prebook_time = st.selectbox("Choose desired start time:", future_times)

    # Dynamic pricing (duration-aware)
    st.markdown("**Dynamic Pricing Simulation**")
    base_price = 20  # per 30 mins
    cur_hour = pd.Timestamp.now().hour
    avg_av_by_area_hour = df[(df['area_name'] == area) & (df['timestamp'].dt.hour == cur_hour)]['available_slots'].mean()
    if np.isnan(avg_av_by_area_hour):
        avg_av_by_area_hour = df[df['area_name'] == area]['available_slots'].mean()

    demand_factor = (10 - (avg_av_by_area_hour / total_slots) * 10) / 10
    traffic_norm = (traffic - 1) / 9
    event_factor = 0.25 if event == 1 else 0.0
    duration_factors = {"30 mins": 1.0, "1 hour": 1.8, "2 hours": 3.2, "3 hours": 4.5, "4 hours": 5.5}
    duration_factor = duration_factors[duration]
    multiplier = 1 + 0.6 * traffic_norm + event_factor + 0.4 * demand_factor
    dynamic_price = round(base_price * multiplier * duration_factor, 2)

    st.write(f"**Base price (per 30 mins): ₹{base_price}**")
    st.write(f"**Selected duration:** {duration}")
    st.write(f"**Selected start time:** {prebook_time}")
    st.success(f"Suggested Parking Price: ₹{dynamic_price}")
    st.caption("Price varies with traffic, availability, events, and duration. Pre-book up to 4 hours in advance.")

    # Prediction
    if st.button("Predict Availability"):
        if model is None:
            st.warning("Model not available. Please run `train_model.py` first.")
        else:
            input_dict = {
                "traffic_level": traffic,
                "nearby_events": event,
                "day_of_week": day,
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
            pred_int = int(max(0, round(pred)))
            st.success(f"Predicted available slots: **{pred_int}** / {total_slots}")
            st.write("Prediction generated using RandomForest model trained on simulated Indian parking data.")

    st.markdown("---")
    st.subheader("Mock Booking & QR Confirmation")
    name = st.text_input("Driver name", value="Demo User")
    if st.button("Book Slot (Generate QR)"):
        booking = {
            "name": name,
            "area": area,
            "duration": duration,
            "prebook_time": prebook_time,
            "price": dynamic_price
        }
        qr_payload = json.dumps(booking)
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(qr_payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.image(buf, caption="Booking QR (mock). Scan to view booking details.")
        st.success(f"Booking confirmed for {name} at {area}. Start: {prebook_time}. Total: ₹{dynamic_price}")
        st.info("In production, this QR would be verified at entry for digital check-in.")

st.markdown("----")
st.write("**Notes:** Prototype demonstrates predictive analytics, duration-aware pricing, event-impact insights, and a 4-hour pre-booking feature.")
