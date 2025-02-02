import pandas as pd
import numpy as np

# Generate synthetic dates from January 1-31, 2025
dates = pd.date_range(start="2025-01-01", end="2025-01-31").date

# Generate random sentiment scores between -1 (negative) and 1 (positive)
sentiment_scores = np.random.uniform(-1, 1, size=len(dates))

# Create a DataFrame
df = pd.DataFrame({"Date": dates, "rolling_sentiment": sentiment_scores})

# Save as CSV
df.to_csv("data/twitter_sentiment_analyzed.csv", index=False)

print("✅ Synthetic sentiment data created for January 2025.")
