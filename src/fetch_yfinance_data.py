import yfinance as yf
import pandas as pd
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Define the ETF ticker and timeframe
TICKER = "SPY"  # S&P 500 ETF
START_DATE = "2019-01-01"  # 5 years ago
END_DATE = "2024-12-31"  # Today

# Fetch data from Yahoo Finance
print(f"📥 Downloading data for {TICKER} from {START_DATE} to {END_DATE}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Reset index to keep 'Date' as a column
df.reset_index(inplace=True)

# Save to CSV
csv_path = "data/yfinance_data.csv"
df.to_csv(csv_path, index=False)

# Print success message and preview
print(f"✅ Data saved to {csv_path}")
print(df.head())

