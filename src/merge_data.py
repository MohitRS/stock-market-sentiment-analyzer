import pandas as pd

# Load datasets
prices = pd.read_csv("data/yfinance_data.csv")
sentiment = pd.read_csv("data/processed_sentiment.csv")

# Convert Date columns to datetime format
prices["Date"] = pd.to_datetime(prices["Date"])
sentiment["Date"] = pd.to_datetime(sentiment["Date"])

# Merge on Date
merged = pd.merge(prices, sentiment, on="Date", how="left")

# Fill missing sentiment values with 0 (neutral)
merged["Sentiment_Score"].fillna(0, inplace=True)

# Save merged data
merged.to_csv("data/merged_data.csv", index=False)
print("✅ Merged dataset saved as data/merged_data.csv")
