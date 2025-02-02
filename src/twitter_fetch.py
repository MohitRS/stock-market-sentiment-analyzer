import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

# Load Twitter API credentials from .env file
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Authenticate using the Bearer Token
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_tweets(query="Stock Market", max_results=100, start_date="2025-01-01", end_date="2025-01-31"):
    """
    Fetch tweets about stocks for January 2025 using the Twitter API.
    
    :param query: Search query (e.g., 'Stock Market', '$SPY', 'ETF trends')
    :param max_results: Number of tweets to fetch per request
    :param start_date: Start date for fetching tweets (ISO format)
    :param end_date: End date for fetching tweets (ISO format)
    :return: Pandas DataFrame containing tweets
    """
    try:
        tweets = []
        for tweet in tweepy.Paginator(client.search_recent_tweets, 
                                      query=query, 
                                      tweet_fields=["created_at", "text", "public_metrics"], 
                                      max_results=max_results, 
                                      start_time=f"{start_date}T00:00:00Z", 
                                      end_time=f"{end_date}T23:59:59Z").flatten(limit=1000):
            tweets.append({
                "timestamp": tweet.created_at,
                "text": tweet.text,
                "retweets": tweet.public_metrics["retweet_count"],
                "likes": tweet.public_metrics["like_count"]
            })

        if tweets:
            df = pd.DataFrame(tweets)
            df.to_csv("data/twitter_sentiment_raw.csv", index=False)
            print(f"✅ Successfully fetched {len(df)} tweets from {start_date} to {end_date}")
            return df
        else:
            print("No tweets found for the given period.")
            return None
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return None

if __name__ == "__main__":
    df = fetch_tweets(query="Stock Market", max_results=100)
