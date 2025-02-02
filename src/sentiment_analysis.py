import pandas as pd
from transformers import pipeline

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Load raw Twitter data
df = pd.read_csv("data/twitter_sentiment_raw.csv")

# Apply sentiment analysis
df["sentiment"] = df["text"].apply(lambda x: sentiment_model(x)[0]["label"])

# Convert sentiment to numerical values
sentiment_mapping = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
df["sentiment_score"] = df["sentiment"].map(sentiment_mapping)

# Save analyzed sentiment data
df.to_csv("data/twitter_sentiment_analyzed.csv", index=False)

print("✅ Sentiment analysis completed and saved!")
