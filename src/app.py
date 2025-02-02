import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 🎨 Custom Styling
st.set_page_config(page_title="Stock Market Prediction", layout="centered")

st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            width: 100%;
        }
        .stSlider {
            color: #FF4B4B;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 📌 Title & Introduction
st.title("📈 Stock Market Sentiment Analysis & Prediction")
st.write(
    "This tool predicts the **ETF closing price** based on sentiment score, previous day close price, and daily return.\n\n"
    "🔹 **Sentiment Score (-1 to +1):** Represents positive/negative market sentiment.\n"
    "🔹 **Previous Day Close Price:** The closing price of the ETF on the previous day.\n"
    "🔹 **Daily Return (%):** The percentage change in ETF price compared to the previous day."
)

# ✅ Load the Model & Scaler (with Error Handling)
model_path = "models/best_model.pkl"
scaler_path = "models/scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.error("❌ Model files not found. Please train the model first.")
    st.stop()  # Stop execution if models are missing

# 📌 User Inputs
sentiment = st.slider("📊 Sentiment Score", -1.0, 1.0, 0.0)
close_lag1 = st.number_input("💹 Previous Day Close Price", value=600.00, format="%.2f")
daily_return = st.number_input("📉 Daily Return (%)", value=0.00, format="%.2f")

# Compute Interaction Feature
interaction = sentiment * close_lag1

# 🔮 Prediction Button
if st.button("🔮 Predict"):
    try:
        # Prepare input
        X_input = np.array([[sentiment, close_lag1, daily_return, interaction]])
        X_scaled = scaler.transform(X_input)

        # Make Prediction
        prediction = model.predict(X_scaled)[0]

        # Display Result
        st.success(f"📈 **Predicted Close Price: {prediction:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")

# 📢 Footer
st.write("\n---\n")
st.write(
    "📌 **How it Works:**\n"
    "1. The model takes the sentiment score, last close price, and daily return as input.\n"
    "2. It normalizes the inputs using the **trained scaler**.\n"
    "3. The trained Ridge Regression model makes a prediction.\n"
    "4. The predicted closing price is displayed!"
)

st.write("💡 **Next Steps:** We will integrate real market data and improve the model performance!")
