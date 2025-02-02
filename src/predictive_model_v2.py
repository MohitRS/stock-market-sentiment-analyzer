import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, TweedieRegressor
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

# Convert price-related columns to float
for col in ["Close", "High", "Low", "Open", "Volume"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Feature Engineering (Remove potentially overfitting features)
data["Close_lag1"] = data["Close"].shift(1)  # Previous day's Close
data["Daily_Return"] = data["Close"].pct_change(fill_method=None)  # % Price change

# Drop NaN values (important for lagged features)
data.dropna(inplace=True)

# Ensure sufficient data is available
if len(data) < 10:
    print(f"❌ ERROR: Not enough data points ({len(data)} rows). Consider increasing dataset size.")
    exit()

# Define features (X) and target (y) - **Remove potentially overfitting features**
X = data[["Sentiment_Score", "Close_lag1", "Daily_Return"]]
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

# **🧠 Try Multiple Models with Tuning**
models = {
    "Ridge": Ridge(alpha=20),  # Increased alpha for better generalization
    "Lasso": Lasso(alpha=1.0),  # Stronger regularization
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.4),  # More L2 regularization
    "BayesianRidge": BayesianRidge(alpha_1=1e-2, alpha_2=1e-2),  # Stronger regularization
    "HuberRegressor": HuberRegressor(alpha=0.001),  # Handles outliers well
    "TweedieRegressor": TweedieRegressor(power=1, alpha=0.1)  # Prevents overfitting
}

# Train & Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    # Cross-validation to check for overfitting
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    results[name] = (mae, r2, cv_mae, model, pred)

# Find the best model (smallest CV MAE)
best_name = min(results, key=lambda k: results[k][2])  # Use cross-validation MAE
best_mae, best_r2, best_cv_mae, best_model, best_pred = results[best_name]

# Print Evaluation
print("\n🔍 Model Performance:")
for name, (mae, r2, cv_mae, _, _) in results.items():
    print(f"📌 {name} -> MAE: {mae:.2f}, R²: {r2:.2f}, CV MAE: {cv_mae:.2f}")

# Save the best model
best_model_path = "models/best_model.pkl"
joblib.dump(best_model, best_model_path)
print(f"✅ Best model: {best_name} (Saved to '{best_model_path}')")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Close Price", color="blue")
plt.plot(best_pred, label="Predicted Close Price", color="red", linestyle="dashed")
plt.title(f"ETF Price Prediction Based on Sentiment & Market Data ({best_name})")
plt.legend()
plt.show()

# Residual Analysis: Plot residuals to check for bias in predictions
residuals = y_test - best_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color='red', linestyle='dashed')
plt.title(f"Residual Distribution of {best_name}")
plt.xlabel("Prediction Error")
plt.show()
