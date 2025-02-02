import pandas as pd

# Load datasets
etf_data = pd.read_csv("data/SPY_historical.csv")
sentiment_data = pd.read_csv("data/twitter_sentiment_analyzed.csv")

# Convert Date columns to datetime format
etf_data["Date"] = pd.to_datetime(etf_data["Date"])
sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"])

# Sort both datasets by Date
etf_data = etf_data.sort_values("Date")
sentiment_data = sentiment_data.sort_values("Date")

# Merge on Date (inner join)
merged_data = pd.merge(etf_data, sentiment_data, on="Date", how="inner")

# Save merged dataset
merged_data.to_csv("data/merged_data.csv", index=False)

print(f"✅ Merged dataset created: {len(merged_data)} rows saved to 'data/merged_data.csv'.")
