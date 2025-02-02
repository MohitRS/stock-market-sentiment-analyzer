import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 🎨 Must be first Streamlit command!
st.set_page_config(page_title="Stock Market Sentiment Predictor", layout="centered")

# Load Model and Scaler
model_path = "models/best_model.pkl"
scaler_path = "models/scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success(f"✅ Loaded Model: {type(model).__name__}")
except Exception as e:
    st.error(f"❌ Model or Scaler Not Found! Error: {e}")
    st.stop()

st.title("📈 Stock Market Sentiment-Based Prediction")
st.write("### Enter the Market and Sentiment Conditions Below:")

# User Input
sentiment = st.slider("📊 Market Sentiment Score", -1.0, 1.0, 0.0)
close_lag1 = st.number_input("💹 Previous Day Close Price", value=600.00, format="%.2f")
daily_return = st.number_input("📉 Daily Return (%)", value=0.00, format="%.2f")

# **Fix: Only use the features the scaler was trained on**
X_input = np.array([[sentiment, close_lag1, daily_return]])  # 3 Features

# Predict Button
if st.button("🔮 Predict Stock Price"):
    try:
        # Transform Input
        X_scaled = scaler.transform(X_input)

        # Make Prediction
        prediction = model.predict(X_scaled)[0]

        # Display Prediction
        st.success(f"📈 **Predicted Close Price: ${prediction:.2f}**")

    except Exception as e:
        st.error(f"⚠️ Prediction Failed! Error: {e}")
