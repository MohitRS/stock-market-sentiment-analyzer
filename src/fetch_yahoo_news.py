import requests
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize Sentiment Analyzer
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# MarketWatch RSS Feeds
MARKETWATCH_RSS = "https://feeds.marketwatch.com/marketwatch/topstories/"

def fetch_marketwatch_news():
    response = requests.get(MARKETWATCH_RSS)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch MarketWatch news. Status: {response.status_code}")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Parse XML
    root = ET.fromstring(response.text)
    
    # Extract news titles
    news_data = []
    for item in root.findall(".//item"):
        title = item.find("title").text
        date = pd.to_datetime("today").date()  # RSS does not provide structured dates
        sentiment_score = sia.polarity_scores(title)["compound"]
        
        news_data.append({"Date": date, "Title": title, "Sentiment": sentiment_score})
    
    return pd.DataFrame(news_data)

# Fetch data
print("📥 Fetching MarketWatch news...")
news_df = fetch_marketwatch_news()

# Handle empty data
if news_df.empty:
    print("⚠️ No news data available. Exiting.")
else:
    # Aggregate sentiment per day
    daily_sentiment = news_df.groupby("Date")["Sentiment"].mean().reset_index()
    
    # Save data
    daily_sentiment.to_csv("data/historical_sentiment.csv", index=False)
    print("✅ Sentiment data saved to data/historical_sentiment.csv")
