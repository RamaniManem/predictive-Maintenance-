import os
import numpy as np
import pandas as pd

# Make the random data reproducible
np.random.seed(42)

# Number of sensor readings
normal_samples = 950
anomaly_samples = 50

# -----------------------------
# NORMAL MACHINE DATA
# -----------------------------
normal_temperature = np.random.normal(
    loc=55,
    scale=5,
    size=normal_samples
)

normal_vibration = np.random.normal(
    loc=2.5,
    scale=0.4,
    size=normal_samples
)

# -----------------------------
# ANOMALOUS MACHINE DATA
# -----------------------------
anomaly_temperature = np.random.normal(
    loc=85,
    scale=8,
    size=anomaly_samples
)

anomaly_vibration = np.random.normal(
    loc=6.5,
    scale=1.0,
    size=anomaly_samples
)

# Combine normal and abnormal readings
temperature = np.concatenate([
    normal_temperature,
    anomaly_temperature
])

vibration = np.concatenate([
    normal_vibration,
    anomaly_vibration
])

# Create labels
status = (
    ["Normal"] * normal_samples +
    ["Anomaly"] * anomaly_samples
)

# Create timestamps
timestamps = pd.date_range(
    start="2026-01-01",
    periods=normal_samples + anomaly_samples,
    freq="h"
)

# Create DataFrame
data = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "status": status
})

# Shuffle the data
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save CSV file
file_path = "data/sensor_data.csv"
data.to_csv(file_path, index=False)

print("Sensor data generated successfully!")
print(f"File saved to: {file_path}")
print(f"Total readings: {len(data)}")
print("\nSample data:")
print(data.head())