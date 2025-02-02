import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
etf_data = pd.read_csv("data/SPY_historical.csv")
sentiment_data = pd.read_csv("data/twitter_sentiment_analyzed.csv")

# Convert Date columns to YYYY-MM-DD format
etf_data["Date"] = pd.to_datetime(etf_data["Date"]).dt.date
sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"]).dt.date  # 🔹 FIXED: Using 'Date' instead of 'timestamp'

# 🔹 Ensure sentiment values are numeric
sentiment_data["rolling_sentiment"] = pd.to_numeric(sentiment_data["rolling_sentiment"], errors="coerce")

# 🔹 Merge datasets on 'Date'
merged_data = pd.merge(etf_data, sentiment_data, on="Date", how="inner")

# Debugging Prints
print("ETF Data Sample After Processing:\n", etf_data.head())
print("Sentiment Data Sample After Processing:\n", sentiment_data.head())
print("Merged Data Sample:\n", merged_data.head())
print(f"Number of rows in merged data: {len(merged_data)}")

# Check for NaN values in merged data
if merged_data["rolling_sentiment"].isna().sum() > 0:
    print("❌ Warning: Some rolling sentiment values are still NaN!")

# Check if we have enough data for correlation
if len(merged_data) <= 1:
    print("❌ Not enough data points for correlation!")
else:
    # Calculate correlation
    correlation = merged_data[["Close", "rolling_sentiment"]].corr()
    print("Correlation between sentiment & ETF Close Price:")
    print(correlation)

    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Between ETF Price & Sentiment Score")
    plt.show()
