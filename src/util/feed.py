import time
import requests
import pandas as pd


def get_yahoo_finance_data(symbol, lookback=3600, interval="1m"):
    """
    reverse engineer http rest get request
    interval options: 1m, 15m, 60m or 1h
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Current time and 1 day ago in Unix timestamp
    end = int(time.time())
    start = end - lookback  # default lookback 1 hr

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start,
        "period2": end,
        "interval": interval,
        "includePrePost": "true",
        "events": "div,splits",
    }

    response = requests.get(url, headers=headers, params=params)
    return response.json()


def parse_yahoo_data(response_json):
    # Extract the arrays from the nested structure
    data = response_json["chart"]["result"][0]

    timestamps = data["timestamp"]
    indicators = data["indicators"]["quote"][0]

    # Create DataFrame with all the OHLCV data
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "Open": pd.to_numeric(
                indicators["open"], errors="coerce"
            ),  # Convert to float, None becomes NaN
            "High": pd.to_numeric(indicators["high"], errors="coerce"),
            "Low": pd.to_numeric(indicators["low"], errors="coerce"),
            "Close": pd.to_numeric(indicators["close"], errors="coerce"),
            "Volume": pd.to_numeric(indicators["volume"], errors="coerce"),
        }
    )

    # Convert timestamp to datetime
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    df = df[df["Volume"] > 0]

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Set timestamp as index
    df = df.drop(["timestamp"], axis=1)

    return df
