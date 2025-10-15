import streamlit as st
import pandas as pd
import requests
import time
import os
import plotly.express as px

st.set_page_config(page_title="Cairo Energy Dashboard", layout="wide")

API_URL = "http://localhost:8000"

st.title(" Cairo Smart Energy Dashboard ")
st.markdown("Visualize, train, and monitor energy prediction models interactively")

st.sidebar.header("Model Controls")


if st.sidebar.button("### Train Model"):
    with st.spinner("Training model..."):
        try:
            res = requests.post(f"{API_URL}/train")
            time.sleep(2)
            res.raise_for_status()
            st.sidebar.success("Model retrained successfully!")
            st.sidebar.json(res.json())
        except Exception as e:
            st.sidebar.error("Training failed!")
            st.sidebar.text(str(e))


if st.sidebar.button("### Predict Energy"):
    with st.spinner("Generating predictions..."):
        try:
            res = requests.get(f"{API_URL}/predict")
            time.sleep(2)
            res.raise_for_status()
            st.sidebar.success("Predictions completed!")
            st.sidebar.json(res.json())
        except Exception as e:
            st.sidebar.error("Prediction failed!")
            st.sidebar.text(str(e))


if st.sidebar.button("### Show Metrics"):
    try:
        res = requests.get(f"{API_URL}/metrics")
        res.raise_for_status()
        metrics = res.json()
        st.sidebar.success("Metrics loaded!")
        st.sidebar.metric(label="Mean Absolute Error (MAE)", value=round(metrics.get("MAE", 0), 4))
        st.sidebar.metric(label="R² Score", value=round(metrics.get("R2", 0), 4))
    except Exception as e:
        st.sidebar.error("Could not load metrics!")
        st.sidebar.text(str(e))


st.subheader("Prediction Data Visualization")

try:
    df = pd.read_csv("data/processed/predicted_energy.csv")
    
    with st.expander("Preview Data"):
        st.dataframe(df.head())

    st.markdown("### Chart Configuration")
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns.tolist()
    
    if numeric_columns:
        selected_column = st.selectbox("Choose column to visualize:", numeric_columns)
        chart_type = st.radio(
            "Select chart type:",
            ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"],
            horizontal=True
        )

        st.markdown(f"### {chart_type} for {selected_column}")

        if chart_type == "Line Chart":
            st.line_chart(df[selected_column])
        elif chart_type == "Bar Chart":
            st.bar_chart(df[selected_column])
        elif chart_type == "Area Chart":
            st.area_chart(df[selected_column])
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, y=selected_column)
            st.plotly_chart(fig)
    else:
        st.info("No numeric columns available for visualization.")

    st.markdown("### Statistical Summary")
    st.write(df.describe())

except FileNotFoundError:
    st.warning("No prediction file found. Generate predictions first.")


st.markdown("### Model Performance Over Time")
metrics_path = "data/metrics/training_log.csv"

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    
    if "timestamp" not in metrics_df.columns:
        metrics_df["timestamp"] = pd.date_range(start="2025-01-01", periods=len(metrics_df), freq="D")

    selected_metrics = st.multiselect(
        "Choose metrics to plot:",
        options=[col for col in metrics_df.columns if col != "timestamp"],
        default=[col for col in ["MAE", "R2"] if col in metrics_df.columns]
    )

    if selected_metrics:
        st.line_chart(metrics_df.set_index("timestamp")[selected_metrics])
        st.caption("Metrics are logged every time the model is retrained.")
    else:
        st.info("Please select at least one metric to visualize.")
else:
    st.warning("No training log found yet. Train the model first to see performance trends.")

st.markdown("---")
st.caption("Cairo Smart Energy System Prediction © 2025 — powered by Streamlit + FastAPI")
