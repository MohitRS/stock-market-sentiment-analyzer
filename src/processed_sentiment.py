import pandas as pd

# Load Kaggle dataset
df = pd.read_csv("data/all-data.csv", encoding="ISO-8859-1", header=None)
df.columns = ["Sentiment", "News"]

# Map sentiment to numerical scores
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
df["Sentiment_Score"] = df["Sentiment"].str.lower().map(sentiment_map)

# Assign random dates (Kaggle dataset has no timestamps)
import numpy as np
dates = pd.date_range(start="2019-01-01", periods=len(df), freq="D")
df["Date"] = np.random.choice(dates, size=len(df))  # Assign random dates

# Aggregate by date
daily_sentiment = df.groupby("Date")["Sentiment_Score"].mean().reset_index()

# Save processed data
daily_sentiment.to_csv("data/processed_sentiment.csv", index=False)
print("✅ Processed sentiment data saved as data/processed_sentiment.csv")
