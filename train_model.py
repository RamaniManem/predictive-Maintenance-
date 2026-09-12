import os
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Load sensor data
data = pd.read_csv("data/sensor_data.csv")

# Select sensor features
features = data[["temperature", "vibration"]]


# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# Create Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Train the model
model.fit(scaled_features)


# Predict anomalies
predictions = model.predict(scaled_features)

# Isolation Forest:
# 1  = Normal
# -1 = Anomaly
data["prediction"] = predictions

data["ml_status"] = data["prediction"].map({
    1: "Normal",
    -1: "Anomaly"
})


# Create models folder
os.makedirs("models", exist_ok=True)


# Save model and scaler
joblib.dump(model, "models/anomaly_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")


# Save results
data.to_csv("data/predicted_sensor_data.csv", index=False)


print("Machine learning model trained successfully!")
print("Model saved to: models/anomaly_model.pkl")
print("Scaler saved to: models/scaler.pkl")

print("\nPrediction summary:")
print(data["ml_status"].value_counts())

print("\nSample predictions:")
print(
    data[
        ["temperature", "vibration", "status", "ml_status"]
    ].head(10)
)