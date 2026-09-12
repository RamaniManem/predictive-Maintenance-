import streamlit as st
import pandas as pd
import plotly.express as px
import joblib


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/predicted_sensor_data.csv")


@st.cache_resource
def load_model():
    model = joblib.load("models/anomaly_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


data = load_data()
model, scaler = load_model()


# -----------------------------
# TITLE
# -----------------------------
st.title("⚙️ Predictive Maintenance System")
st.subheader("Machine Health Monitoring & Anomaly Detection")

st.write(
    "This dashboard monitors machine temperature and vibration "
    "and uses Machine Learning to detect abnormal operating conditions."
)


# -----------------------------
# MACHINE STATUS
# -----------------------------
normal_count = (data["ml_status"] == "Normal").sum()
anomaly_count = (data["ml_status"] == "Anomaly").sum()

total_readings = len(data)
anomaly_percentage = (anomaly_count / total_readings) * 100


# -----------------------------
# ALERT
# -----------------------------
if anomaly_count > 0:
    st.error(
        f"🚨 ALERT: {anomaly_count} anomalous readings detected!"
    )
else:
    st.success("✅ Machine operating normally.")


# -----------------------------
# SUMMARY METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Readings",
        total_readings
    )

with col2:
    st.metric(
        "Normal Readings",
        normal_count
    )

with col3:
    st.metric(
        "Anomalies",
        anomaly_count
    )

with col4:
    st.metric(
        "Anomaly Rate",
        f"{anomaly_percentage:.1f}%"
    )


# -----------------------------
# CURRENT MACHINE CONDITION
# -----------------------------
st.divider()

st.header("📊 Machine Sensor Data")


latest_data = data.iloc[0]

temperature = latest_data["temperature"]
vibration = latest_data["vibration"]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{temperature:.2f} °C"
    )

with col2:
    st.metric(
        "📳 Vibration",
        f"{vibration:.2f}"
    )


# -----------------------------
# TEMPERATURE GRAPH
# -----------------------------
st.divider()

st.header("🌡️ Temperature Monitoring")

temperature_chart = px.line(
    data,
    x="timestamp",
    y="temperature",
    title="Temperature Over Time",
    labels={
        "timestamp": "Time",
        "temperature": "Temperature (°C)"
    }
)

st.plotly_chart(
    temperature_chart,
    use_container_width=True
)


# -----------------------------
# VIBRATION GRAPH
# -----------------------------
st.header("📳 Vibration Monitoring")

vibration_chart = px.line(
    data,
    x="timestamp",
    y="vibration",
    title="Machine Vibration Over Time",
    labels={
        "timestamp": "Time",
        "vibration": "Vibration"
    }
)

st.plotly_chart(
    vibration_chart,
    use_container_width=True
)


# -----------------------------
# NORMAL VS ANOMALY GRAPH
# -----------------------------
st.header("🚨 Anomaly Detection")

status_chart = px.scatter(
    data,
    x="temperature",
    y="vibration",
    color="ml_status",
    title="Normal vs Anomalous Machine Conditions",
    labels={
        "temperature": "Temperature (°C)",
        "vibration": "Vibration",
        "ml_status": "Machine Status"
    }
)

st.plotly_chart(
    status_chart,
    use_container_width=True
)


# -----------------------------
# RECENT ANOMALIES
# -----------------------------
st.divider()

st.header("🔍 Detected Anomalies")

anomalies = data[data["ml_status"] == "Anomaly"]

if len(anomalies) > 0:
    st.dataframe(
        anomalies[
            [
                "timestamp",
                "temperature",
                "vibration",
                "ml_status"
            ]
        ].head(20),
        use_container_width=True
    )
else:
    st.success("No anomalies detected.")


# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Predictive Maintenance System | "
    "Machine Learning based anomaly detection"
)