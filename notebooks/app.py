import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly

# -------------------------------------------------------------
# Streamlit App: Sales Forecasting Dashboard
# -------------------------------------------------------------

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📈 Sales Forecasting Dashboard")
st.markdown("Upload your sales data and forecast future trends using Facebook Prophet.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your sales data (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)

    # Basic validation
    if 'date' not in df.columns or 'sales' not in df.columns:
        st.error("❌ CSV must contain 'date' and 'sales' columns.")
    else:
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        # Prophet expects 'ds' and 'y' column names
        data = df.rename(columns={'date': 'ds', 'sales': 'y'})

        # Model training
        model = Prophet()
        model.fit(data)

        # Future dataframe
        future = model.make_future_dataframe(periods=30)  # forecast for next 30 days
        forecast = model.predict(future)

        # Show forecast results
        st.subheader("🔮 Forecasted Data")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Plot forecast
        st.subheader("📉 Forecast Visualization")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        # Plot forecast components
        st.subheader("📊 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

else:
    st.write("⚠️ Please upload a CSV file with 'date' and 'sales' columns.")
