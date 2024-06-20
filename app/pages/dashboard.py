import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from typing import List
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


def load_csv_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f'Error loading data: {e}')
    
    return data

def filter_data(data, column: str, value):
    return data[data[column] == value]

def get_col_max(data, column: str):
    return data[column].max()

def load_ml_model(model_path: str):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f'Error loading model: {e}')
    
    return model

def split_features_labels(data, target: str, features: List[str]) -> DataFrame:
    X = data[features]
    y = data[target]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def predict_row(row, model, scaler, features: List[str]):
    # Drop the target column
    X_row = row[features].values.reshape(1, -1)  # Reshape for scaler

    # Scale the row
    X_row_scaled = scaler.transform(X_row)

    # Predict the category
    y_pred = model.predict(X_row_scaled).item()

    row['y_pred'] = y_pred
    return row


def main():

    # Paths to model and data
    MODEL_PATH = './models/classifiers/rf_model.pkl'
    DATA_PATH = './data/production/FD001.csv'
    SCALER_PATH = './models/scalers/scaler.pkl'

    # Load data and model
    data = load_csv_data(DATA_PATH)
    model = load_ml_model(MODEL_PATH)
    
    # Filter subset of data
    subset = filter_data(data, 'unit_number', 2)
    
    # Load pre-fitted scaler
    scaler = joblib.load(SCALER_PATH)

    # Define features
    features = subset.columns[3:-1]

    # Get the max cycle
    max_cycle = get_col_max(data, 'time_in_cycles')

    st.set_page_config(
        page_title='Realtime Remaining Useful Life Prediction Dashboard', page_icon='📈', layout='wide'
        )

    col001, col002 = st.columns(2)

    with col001:
        st.title('Realtime Remaining Useful Life Prediction 📈')

    # laceholdrr for the current label
    label_placeholder = col002.empty()

    # Button to start the inference
    start_button = st.button("Start Process")

    if start_button:

        # Initialize list for predictions
        predictions = []
            
        col1, col2, col3 = st.columns(3)

        # Create placeholders for each plot
        with col1:
            placeholder1 = st.empty()
            placeholder2 = st.empty()
            placeholder3 = st.empty()

        with col2:    
            placeholder4 = st.empty()
            placeholder5 = st.empty()
            placeholder6 = st.empty()

        with col3:
            placeholder7 = st.empty()
            placeholder8 = st.empty()
            placeholder9 = st.empty()

        color_map = {'urgent': 'red', 'medium': 'orange', 'long': 'blue'}

        # Stream each row and plot in real-time
        for idx, row in subset.iterrows():
            predicted_row = predict_row(row, model, scaler, features)
            predictions.append(predicted_row)
            
            # Create DataFrame from current predictions
            subset_with_predictions = pd.DataFrame(predictions)
            
            fig_1 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_2_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 02",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_2 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_3_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 03",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_3 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_4_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 04",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_4 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_7_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 07",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_5 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_8_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 08",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_6 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_9_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 09",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_7 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_11_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 11",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_8 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_12_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 12",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            fig_9 = px.line(subset_with_predictions, x="time_in_cycles", y="sensor_measurement_13_rm", color='y_pred', color_discrete_map=color_map).update_layout(showlegend=False, title={'text': "Sensor 13",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

            # Update plots in placeholders
            with placeholder1:
                st.plotly_chart(fig_1)
            with placeholder2:
                st.plotly_chart(fig_2)
            with placeholder3:
                st.plotly_chart(fig_3)
            with placeholder4:
                st.plotly_chart(fig_4)
            with placeholder5:
                st.plotly_chart(fig_5)
            with placeholder6:
                st.plotly_chart(fig_6)
            with placeholder7:
                st.plotly_chart(fig_7)
            with placeholder8:
                st.plotly_chart(fig_8)
            with placeholder9:
                st.plotly_chart(fig_9)


            current_label = (subset_with_predictions['y_pred'].tail(1).item())
            label_color = color_map[current_label]
            label_placeholder.markdown(f"## <span style='color:{label_color}'>{current_label.upper()}</span>", unsafe_allow_html=True)

            # Sleep to simulate real-time streaming
            time.sleep(0.1)  

if __name__ == '__main__':
    main()