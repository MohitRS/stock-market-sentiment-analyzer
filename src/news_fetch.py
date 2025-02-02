import requests
import json
import pandas as pd

API_KEY = "4dd595c5d15d490a9b14cc195264c717"

def fetch_financial_news(query="Stock Market", page_size=20):
    """
    Fetch latest financial news articles using NewsAPI.

    :param query: Search query (e.g., 'Stock Market', 'ETF trends')
    :param page_size: Number of articles to fetch
    :return: Pandas DataFrame with news articles
    """
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}&pageSize={page_size}"

    try:
        response = requests.get(url)
        data = response.json()
        
        if "articles" in data:
            articles = data["articles"]
            df = pd.DataFrame(articles)[["source", "title", "description", "url", "publishedAt"]]
            df["source"] = df["source"].apply(lambda x: x["name"])  # Extract source name
            return df
        else:
            print(f"Error fetching news: {data}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    df = fetch_financial_news()
    
    if df is not None:
        print(df.head())
        df.to_csv("data/financial_news.csv", index=False)
