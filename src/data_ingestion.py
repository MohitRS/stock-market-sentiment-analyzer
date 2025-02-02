import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_etf_data(ticker="SPY", start="2025-01-01", end=None):
    """
    Fetch historical ETF stock price data from Yahoo Finance.

    :param ticker: ETF ticker symbol (e.g., SPY, QQQ, VOO)
    :param start: Start date (YYYY-MM-DD)
    :param end: End date (YYYY-MM-DD), default is today.
    :return: Pandas DataFrame with stock price data
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")  # Get latest date

    try:
        etf = yf.Ticker(ticker)
        df = etf.history(period="1d", start=start, end=end)

        # Keep only useful columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.reset_index(inplace=True)
        
        # Convert to YYYY-MM-DD format
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        
        return df
    except Exception as e:
        print(f"Error fetching ETF data: {e}")
        return None


if __name__ == "__main__":
    ticker = "SPY"  # Change to any ETF you want
    df = fetch_etf_data(ticker)
    
    if df is not None:
        print(df.head())  # Display first few rows
        df.to_csv(f"data/SPY_historical.csv", index=False)
