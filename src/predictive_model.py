import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load merged data
data_path = "data/merged_data.csv"
if not os.path.exists(data_path):
    print(f"❌ ERROR: '{data_path}' not found! Please run data processing first.")
    exit()

data = pd.read_csv(data_path)

# Ensure Date column is in datetime format
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

# Feature Engineering
data["Close_lag1"] = data["Close"].shift(1)  # Previous day's Close
data["Daily_Return"] = data["Close"].pct_change()  # % Price change
data["Sentiment_Close_Interaction"] = data["rolling_sentiment"] * data["Close"]

# Drop NaN values (important for lagged features)
data.dropna(inplace=True)

# Ensure sufficient data is available
if len(data) < 10:
    print(f"❌ ERROR: Not enough data points ({len(data)} rows). Consider increasing dataset size.")
    exit()

# Define features (X) and target (y)
X = data[["rolling_sentiment", "Close_lag1", "Daily_Return", "Sentiment_Close_Interaction"]]
y = data["Close"]

# Train-test split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for deployment
scaler_path = "models/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to '{scaler_path}'.")

# 🏆 **Try Two Models**
ridge = Ridge(alpha=1.0)  # Best for small data, prevents overfitting
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)  # Non-linear, good for small datasets

# Train both models
ridge.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

# Make predictions
ridge_pred = ridge.predict(X_test_scaled)
rf_pred = rf.predict(X_test_scaled)

# Evaluate models
ridge_mae = mean_absolute_error(y_test, ridge_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Choose the best model based on MAE
best_model = ridge if ridge_mae < rf_mae else rf
best_pred = ridge_pred if ridge_mae < rf_mae else rf_pred
best_name = "Ridge Regression" if best_model == ridge else "Random Forest"

# Print Evaluation
print("\n🔍 Model Performance:")
print(f"📌 Ridge MAE: {ridge_mae:.2f}, R²: {ridge_r2:.2f}")
print(f"📌 Random Forest MAE: {rf_mae:.2f}, R²: {rf_r2:.2f}")

# Save the best model
best_model_path = "models/best_model.pkl"
joblib.dump(best_model, best_model_path)
print(f"✅ Best model ({best_name}) saved successfully at '{best_model_path}'!")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Close Price", color="blue")
plt.plot(best_pred, label="Predicted Close Price", color="red", linestyle="dashed")
plt.title(f"Final ETF Price Prediction Based on Sentiment & Market Data ({best_name})")
plt.legend()
plt.show()
