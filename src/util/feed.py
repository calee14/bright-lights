import requests
import pandas as pd
from time import time


def get_yahoo_finance_data(symbol, lookback=3600, interval="1m", offset=0):
    """
    Reverse engineer http rest get request
    interval options: 1m, 15m, 60m or 1h

    This function groups every 3 candles together, starting from the back.
    For example, if you have 100 candles, it will group:
    - Candles 98-100 into 1 candle
    - Candles 95-97 into 1 candle
    - Candles 92-94 into 1 candle
    - etc.
    """
    if interval[-1] == "m" and int(interval[:-1]) <= 5:
        interval_size = int(interval[:-1])
        interval = "1m"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # Current time and 1 day ago in Unix timestamp
    end = int(time()) - offset
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
    data = response.json()

    if interval_size:
        # Group every 3 candles starting from the back
        data = group_candles_from_back(data, group_size=interval_size)

    return data


def group_candles_from_back(yahoo_data, group_size=3):
    """
    Groups candles together starting from the most recent candle (the back).

    For OHLC data:
    - Open: Use the open of the FIRST candle in the group
    - High: Use the maximum high across all candles in the group
    - Low: Use the minimum low across all candles in the group
    - Close: Use the close of the LAST candle in the group
    - Volume: Sum the volume across all candles in the group
    - Timestamp: Use the timestamp of the LAST candle in the group

    Args:
        yahoo_data: The JSON response from Yahoo Finance
        group_size: Number of candles to group together (default: 3)

    Returns:
        Modified yahoo_data with grouped candles
    """
    try:
        # Extract the chart data
        chart = yahoo_data["chart"]["result"][0]

        # Get the raw data arrays
        timestamps = chart["timestamp"]
        quote = chart["indicators"]["quote"][0]

        opens = quote["open"]
        highs = quote["high"]
        lows = quote["low"]
        closes = quote["close"]
        volumes = quote["volume"]

        # Get the total number of candles
        num_candles = len(timestamps)

        if num_candles < group_size:
            # Not enough candles to group, return as-is
            return yahoo_data

        # Calculate how many candles to skip at the beginning
        # to ensure we start grouping from the back
        remainder = num_candles % group_size

        # Initialize lists for grouped data
        grouped_timestamps = []
        grouped_opens = []
        grouped_highs = []
        grouped_lows = []
        grouped_closes = []
        grouped_volumes = []

        # Skip the remainder candles at the beginning to keep it simple
        # This ensures we only have complete groups of group_size
        start_idx = remainder

        # Now process complete groups of group_size
        for i in range(start_idx, num_candles, group_size):
            end_idx = min(i + group_size, num_candles)

            # Extract the group
            group_timestamps = timestamps[i:end_idx]
            group_opens = opens[i:end_idx]
            group_highs = highs[i:end_idx]
            group_lows = lows[i:end_idx]
            group_closes = closes[i:end_idx]
            group_volumes = volumes[i:end_idx]

            # Filter out None values for aggregation
            valid_highs = [h for h in group_highs if h is not None]
            valid_lows = [lo for lo in group_lows if lo is not None]
            valid_opens = [o for o in group_opens if o is not None]
            valid_closes = [c for c in group_closes if c is not None]
            valid_volumes = [v for v in group_volumes if v is not None]

            # Skip if we don't have valid data
            if not (valid_opens and valid_closes and valid_highs and valid_lows):
                continue

            # Group the candles
            grouped_timestamps.append(
                group_timestamps[-1]
            )  # Use last timestamp in group
            grouped_opens.append(valid_opens[0])  # First open
            grouped_highs.append(max(valid_highs))  # Max high
            grouped_lows.append(min(valid_lows))  # Min low
            grouped_closes.append(valid_closes[-1])  # Last close
            grouped_volumes.append(sum(valid_volumes) if valid_volumes else 0)

        # Update the chart data with grouped values
        chart["timestamp"] = grouped_timestamps
        chart["indicators"]["quote"][0]["open"] = grouped_opens
        chart["indicators"]["quote"][0]["high"] = grouped_highs
        chart["indicators"]["quote"][0]["low"] = grouped_lows
        chart["indicators"]["quote"][0]["close"] = grouped_closes
        chart["indicators"]["quote"][0]["volume"] = grouped_volumes

        return yahoo_data

    except (KeyError, IndexError, TypeError) as e:
        print(f"Error grouping candles: {e}")
        # Return original data if there's an error
        return yahoo_data


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
