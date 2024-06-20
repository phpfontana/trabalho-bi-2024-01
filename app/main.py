import pandas as pd
import numpy as np
import streamlit as st
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# Paths to data and model
MODEL_PATH = "./models/classifiers/knn_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


# Load the sample data
def load_data():
    try:
        data = pd.read_csv("./data/production/FD001.csv", encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(
            "./data/production/FD001.csv", encoding="ISO-8859-1"
        )
    return data


# Normalize the data
def normalize_data(data):
    scaler = StandardScaler()
    feature_columns = data.columns.difference(
        ["unit_number", "time_in_cycles", "RUL", "maintanance_urgency"]
    )
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data


# Streamlit app
st.title("Ingestion Pipeline Simulation with KNN Classification")

# Load the data and model
data = load_data()
model = load_model()

# Normalize the data
data = normalize_data(data)


# Select unit
unit = st.selectbox("Select Unit", data["unit_number"].unique())

# Filter data for the selected unit
unit_data = data[data["unit_number"] == unit].copy()

# Custom color palette
urgency_palette = sns.color_palette(["green", "orange", "red"])

# Display the preview of the selected unit
st.subheader("Preview of Selected Unit")
st.table(unit_data.head())

# Fixed sensor columns to display
fixed_sensor_columns = [
    "sensor_measurement_2_rm",
    "sensor_measurement_3_rm",
    "sensor_measurement_4_rm",
    "sensor_measurement_7_rm",
    "sensor_measurement_8_rm",
    "sensor_measurement_9_rm",
    "sensor_measurement_11_rm",
    "sensor_measurement_12_rm",
    "sensor_measurement_13_rm",
]

# Ensure the fixed sensor columns exist in the dataset
valid_sensor_columns = [col for col in fixed_sensor_columns if col in unit_data.columns]

# Streaming simulation
st.subheader("Streaming Data Simulation")
plot_area = st.empty()


# Simulation function
def simulate_streaming(data, sensor_columns):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    max_cycles = len(data)
    for cycle in range(max_cycles):
        for i, sensor in enumerate(sensor_columns):
            ax = axes[i]
            ax.clear()
            sns.lineplot(
                x=data["time_in_cycles"][: cycle + 1],
                y=data[sensor][: cycle + 1],
                hue=data["maintanance_urgency"][: cycle + 1],
                palette=urgency_palette,
                ax=ax,
            )
            ax.set_title(f"Sensor Data: {sensor}")
            ax.set_xlabel("Cycles")
            ax.set_ylabel("Sensor Readings")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            ["Long", "Medium", "Urgent"],
            title="Maintenance Urgency",
            loc="upper right",
        )
        plot_area.pyplot(fig)
        


# Run simulation
if st.button("Start Simulation"):
    simulate_streaming(unit_data, valid_sensor_columns)