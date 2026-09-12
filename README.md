# 🚀 Predictive Maintenance System

## 📌 Project Overview

The Predictive Maintenance System is a Machine Learning based system designed to monitor machine health using **temperature** and **vibration** sensor data.

The system analyzes sensor readings and detects unusual operating conditions (anomalies) that may indicate a possible machine failure.

This helps in identifying potential problems early and supports **predictive maintenance** instead of waiting for equipment failure.

---

## 🎯 Objectives

- Monitor machine temperature and vibration.
- Generate and analyze sensor data.
- Detect abnormal machine conditions using Machine Learning.
- Identify potential equipment failures early.
- Display machine health information through an interactive dashboard.
- Provide alerts when anomalies are detected.

---

## 🧠 Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Plotly**
- **Machine Learning – Isolation Forest**

---

## ⚙️ System Workflow

```text
Sensor Data
     ↓
Data Generation
     ↓
Data Preprocessing
     ↓
Machine Learning Model
     ↓
Anomaly Detection
     ↓
Prediction Results
     ↓
Streamlit Dashboard
     ↓
Machine Health Alert
📊 Features
🌡️ Temperature Monitoring

Monitors machine temperature readings and displays temperature trends.

📳 Vibration Monitoring

Analyzes vibration levels to identify unusual machine behavior.

🚨 Anomaly Detection

The Isolation Forest Machine Learning algorithm identifies abnormal sensor readings.

📈 Interactive Dashboard

A Streamlit dashboard provides:

Total sensor readings
Normal readings
Anomalies
Anomaly percentage
Temperature graph
Vibration graph
Normal vs Anomaly visualization
Detected anomaly table
📁 Project Structure
Predictive maintainace system/
│
├── data/
│   ├── sensor_data.csv
│   └── predicted_sensor_data.csv
│
├── models/
│   ├── anomaly_model.pkl
│   └── scaler.pkl
│
├── generate_data.py
├── train_model.py
├── dashboard.py
├── README.md
└── .gitignore
🚀 How to Run
1. Activate the virtual environment
.\.venv311\Scripts\Activate.ps1
2. Generate sensor data
python generate_data.py
3. Train the Machine Learning model
python train_model.py
4. Start the dashboard
streamlit run dashboard.py

Then open:

http://localhost:8501
🤖 Machine Learning

The project uses the Isolation Forest algorithm for anomaly detection.

The trained model analyzes:

Temperature
Vibration

and classifies machine readings as:

Normal

or

Anomaly
📈 Current Model Results

The generated dataset contains:

1000 total readings
950 normal readings
50 anomalous readings

The trained model successfully detects abnormal machine conditions from the sensor data.

💡 Future Enhancements
Real-time IoT sensor integration.
ESP32-based sensor collection.
Live machine monitoring.
Email/SMS alerts.
Failure prediction before an anomaly occurs.
Cloud-based monitoring.
Multiple machine support.
Database integration.
Advanced Machine Learning models.