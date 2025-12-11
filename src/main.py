# src/main.py
import threading
from time import sleep, time
from typing import Any, List, Dict, Optional
from src.util.bot import (
    message_queue,
    messenger,
    start_bot,
    stop_bot,
    is_bot_ready,
)
from src.util.feed import get_yahoo_finance_data, parse_yahoo_data
from src.util.plot import plot_recent_candlesticks

from concurrent.futures import Future, ThreadPoolExecutor, thread
from rich.markdown import Markdown
from rich.console import Console
from datetime import datetime
from threading import Event
import asyncio
import json
import re
import argparse

console = Console()


def std_alert(data, std=2.0):
    """
    Makes a alert whenever the current
    price (last row in df) reaches
    the std threshold.

    Args:
        data: Pandas dataframe
        std: Float
    Returns:
        dict containing signal
    """

    window = min(20, len(data) - 1)
    mean = data["Close"].iloc[:-1].tail(window).mean()
    std_dev = data["Close"].iloc[:-1].tail(window).std()

    current_price = data["Close"].iloc[-1]

    if std_dev == 0:
        return None

    deviation = (current_price - mean) / std_dev

    # Output alert
    if abs(deviation) >= std:
        avg_volume = data["Volume"].iloc[:-1].tail(window).mean()
        current_volume = data["Volume"].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        alert = {
            "type": "MEAN_REVERSION",
            "direction": "ABOVE" if deviation > 0 else "BELOW",
            "current_price": current_price,
            "mean": mean,
            "std_dev": std_dev,
            "deviation": deviation,
            "threshold": std,
            "volume_ratio": volume_ratio,
            "timestamp": data.index[-1]
            if hasattr(data.index[-1], "strftime")
            else datetime.now(),
        }

        return alert
    return None


def calculate_roc(data, period=5):
    """
    Calculate current rate of change.

    Args:
        data: Pandas dataframe
        period: lookback period for ROC

    Returns:
        float: rate of change as decimal (0.01 = 1%)
    """
    if len(data) < period:
        return 0

    current_price = data["Close"].iloc[-1]
    past_price = data["Close"].iloc[-period]

    if past_price == 0:
        return 0

    return (current_price - past_price) / past_price


def trend_alert(data, threshold=0.7, n_candles=20, decay_rate=0.98, roc_weight=0.3):
    """
    Makes an alert whenever the ratio
    of bullish vs. bearish scores
    break the threshold.

    Args:
        data: Pandas dataframe
        threshold: Float
        n_candles: Integer
        decay_rate: Float
    Returns:
        dict containing signal
    """

    recent_data = data.tail(n_candles).copy()

    avg_volume = data["Volume"].mean()
    if avg_volume == 0:
        avg_volume = 1

    # Calculate score for upward
    # and downward momentum based
    # on price movement and volume
    green_score = 0
    red_score = 0

    roc_period = min(5, n_candles // 3)

    for i in range(len(recent_data)):
        row = recent_data.iloc[i]

        candles_back = len(recent_data) - 1 - i
        time_weight = decay_rate**candles_back

        price_move = abs(
            (row["Close"] - row["Open"]) / row["Open"] if row["Open"] != 0 else 0
        )

        volume_factor = row["Volume"] / avg_volume

        # Calculate rate of change
        # and add it to score
        current_idx = recent_data.index[i]
        data_position = data.index.get_loc(current_idx)

        if data_position >= roc_period:
            prev_close = data.iloc[data_position - roc_period]["Close"]
            current_close = row["Close"]

            roc = (
                abs((current_close - prev_close) / prev_close) if prev_close != 0 else 0
            )

            roc_normalized = min(roc * 20, 1.0)

        else:
            roc_normalized = 0

        base_score = (1 - roc_weight) * price_move + roc_weight * roc_normalized

        candle_score = base_score * volume_factor * time_weight

        # Accumulate scores by direction
        if row["Close"] >= row["Open"]:
            green_score += candle_score
        else:
            red_score += candle_score

    total_score = green_score + red_score
    if total_score == 0:
        return None

    green_ratio = green_score / total_score
    red_ratio = red_score / total_score

    recent_roc = calculate_roc(data.tail(roc_period * 2), roc_period)
    trend_momentum = "ACCELERATING" if recent_roc > 0.01 else "STEADY"

    # Make alerts
    if green_ratio >= threshold:
        alert = {
            "type": "TREND",
            "direction": "UPTREND",
            "green_score": green_score * 100,
            "red_score": red_score * 100,
            "ratio": green_ratio,
            "threshold": threshold,
            "strength": "STRONG" if green_ratio >= 0.8 else "MODERATE",
            "momentum": trend_momentum,
            "roc": recent_roc * 100,  # As percentage
            "n_candles": n_candles,
            "timestamp": data.index[-1]
            if hasattr(data.index[-1], "strftime")
            else datetime.now(),
        }
        return alert

    elif red_ratio >= threshold:
        alert = {
            "type": "TREND",
            "direction": "DOWNTREND",
            "green_score": green_score * 100,
            "red_score": red_score * 100,
            "ratio": red_ratio,
            "threshold": threshold,
            "strength": "STRONG" if red_ratio >= 0.88 else "MODERATE",
            "momentum": trend_momentum,
            "roc": recent_roc * 100,  # As percentage
            "n_candles": n_candles,
            "timestamp": data.index[-1]
            if hasattr(data.index[-1], "strftime")
            else datetime.now(),
        }
        return alert


def volume_anomaly_alert(data, threshold=2.1, n_candles=10):
    """
    Makes an alert whenever the
    current volume is greater than
    the avg volume by a factor of
    the threshold.

    Args:
        data: Pandas dataframe
        threshold: Float
        n_candles: Integer
    Returns:
        dict containing signal
    """

    if len(data) < n_candles + 1:
        return None

    current_candle = data.iloc[-1]
    current_volume = current_candle["Volume"]

    previous_candles = data.iloc[-(n_candles + 1) : -1]
    avg_volume = previous_candles["Volume"].mean()

    if avg_volume == 0:
        return None

    volume_ratio = current_volume / avg_volume

    if volume_ratio >= threshold:
        recent_volumes = data["Volume"].tail(5).tolist()
        is_accelerating = (
            len(recent_volumes) >= 2 and recent_volumes[-1] > recent_volumes[-2]
        )

        alert = {
            "type": "VOLUME_ANOMALY",
            "current_volume": int(current_volume),
            "avg_volume": int(avg_volume),
            "volume_ratio": round(volume_ratio, 2),
            "threshold": threshold,
            "is_accelerating": is_accelerating,
            "strength": "EXTREME" if volume_ratio >= threshold * 1.5 else "HIGH",
            "n_candles": n_candles,
            "timestamp": data.index[-1]
            if hasattr(data.index[-1], "strftime")
            else datetime.now(),
        }
        return alert

    return None


def check_alerts(symbol="QQQ", lookback=3600, interval="1m", offset=0):
    """
    Args:
        symbol: String
        lookback: Integer
        Interval: String
        Offset: Integer
    Returns:
        Void

    """

    try:
        data = parse_yahoo_data(
            get_yahoo_finance_data("QQQ", lookback=3600, interval="1m", offset=0)
        )

        plot_recent_candlesticks(data, filename="charts/tminus60.jpeg")

        # Check for alerts
        std_signal = std_alert(data, std=1.9)
        trend_signal = trend_alert(
            data, threshold=0.65, n_candles=13, decay_rate=0.9, roc_weight=0.5
        )
        volume_signal = volume_anomaly_alert(data, threshold=2.1, n_candles=10)

        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "std_signal": std_signal,
            "trend_signal": trend_signal,
            "volume_signal": volume_signal,
            "data": data,  # Include data in case you need it
        }

    except Exception as e:
        console.print(f"[red]Error checking alerts: {e}[/red]")
        return None


def display_alerts(alerts: Optional[Dict[Any, Any]]):
    """
    Dislay alerts in terminal
    and on Discord
    """
    if alerts is None:
        return

    std_signal = alerts.get("std_signal")
    trend_signal = alerts.get("trend_signal")
    volume_signal = alerts.get("volume_signal")
    if std_signal:
        console.print("\n[bold red]🔔 MEAN REVERSION ALERT![/bold red]")
        console.print(
            f"[dim]Time: {std_signal['timestamp'].strftime('%H:%M:%S')}[/dim]"
        )
        console.print(f"Direction: {std_signal['direction']}")
        console.print(f"Current Price: ${std_signal['current_price']:.2f}")
        console.print(f"Mean: ${std_signal['mean']:.2f}")
        console.print(f"Deviation: {std_signal['deviation']:.2f} standard deviations")
        console.print(f"Volume Ratio: {std_signal['volume_ratio']:.2f}x average")

        # Send to Discord
        discord_msg = (
            f"🔔 **MEAN REVERSION ALERT!**\n"
            f"Time: {std_signal['timestamp'].strftime('%H:%M:%S')}\n"
            f"Direction: {std_signal['direction']}\n"
            f"Current Price: ${std_signal['current_price']:.2f}\n"
            f"Mean: ${std_signal['mean']:.2f}\n"
            f"Deviation: {std_signal['deviation']:.2f} standard deviations\n"
            f"Volume Ratio: {std_signal['volume_ratio']:.2f}x average"
        )
        message_queue.put(discord_msg)

    if trend_signal:
        console.print("\n[bold green]🔔 TREND ALERT![/bold green]")
        console.print(
            f"[dim]Time: {trend_signal['timestamp'].strftime('%H:%M:%S')}[/dim]"
        )
        console.print(f"Direction: {trend_signal['direction']}")
        console.print(f"Strength: {trend_signal['strength']}")
        console.print(
            f"Momentum: {trend_signal['momentum']} ({trend_signal['roc']:+.2f}%)"
        )
        console.print(f"Ratio: {trend_signal['ratio']:.2%}")
        console.print(f"Green Score: {trend_signal['green_score']:.2f}")
        console.print(f"Red Score: {trend_signal['red_score']:.2f}")

        # Send to Discord
        discord_msg = (
            f"🔔 **TREND ALERT!**\n"
            f"Time: {trend_signal['timestamp'].strftime('%H:%M:%S')}\n"
            f"Direction: {trend_signal['direction']}\n"
            f"Strength: {trend_signal['strength']}\n"
            f"Momentum: {trend_signal['momentum']} ({trend_signal['roc']:+.2f}%)\n"
            f"Ratio: {trend_signal['ratio']:.2%}\n"
            f"Green Score: {trend_signal['green_score']:.2f}\n"
            f"Red Score: {trend_signal['red_score']:.2f}"
        )
        message_queue.put(discord_msg)

    if volume_signal:
        console.print("\n[bold yellow]🔔 VOLUME SPIKE ALERT![/bold yellow]")
        console.print(
            f"[dim]Time: {volume_signal['timestamp'].strftime('%H:%M:%S')}[/dim]"
        )
        console.print(f"Current Volume: {volume_signal['current_volume']:,}")
        console.print(f"Average Volume: {volume_signal['avg_volume']:,}")
        console.print(
            f"Ratio: {volume_signal['volume_ratio']}x (threshold: {volume_signal['threshold']}x)"
        )
        console.print(f"Strength: {volume_signal['strength']}")
        if volume_signal["is_accelerating"]:
            console.print("[red]⚠️  Volume is ACCELERATING![/red]")

        # Send to Discord
        accelerating_text = ""
        if volume_signal["is_accelerating"]:
            console.print("[red]⚠️  Volume is ACCELERATING![/red]")
            accelerating_text = "\n⚠️ Volume is ACCELERATING!"

        discord_msg = (
            f"🔔 **VOLUME SPIKE ALERT!**\n"
            f"Time: {volume_signal['timestamp'].strftime('%H:%M:%S')}\n"
            f"Current Volume: {volume_signal['current_volume']:,}\n"
            f"Average Volume: {volume_signal['avg_volume']:,}\n"
            f"Ratio: {volume_signal['volume_ratio']}x (threshold: {volume_signal['threshold']}x)\n"
            f"Strength: {volume_signal['strength']}"
            f"{accelerating_text}"
        )
        message_queue.put(discord_msg)


def alert_monitor_loop(symbol="QQQ", interval_seconds=1, stop_event=None):
    """
    Continuously monitor for alerts in a loop.

    Args:
        symbol: Stock ticker to monitor
        interval_seconds: How often to check (in seconds)
        stop_event: threading.Event to signal when to stop
    """
    console.print(f"[green]Starting alert monitor for {symbol}...[/green]")

    while True:
        # Check if main loop has
        # terminated
        if stop_event and stop_event.is_set():
            console.print("[yellow]Alert monitor stopped.[/yellow]")
            break

        alerts = check_alerts(symbol=symbol)

        if alerts:
            display_alerts(alerts)

        sleep(interval_seconds)


def start_alert_monitor_thread(symbol="QQQ", interval_seconds=60):
    """
    Start the alert monitor in a background thread.

    Args:
        symbol: Stock ticker to monitor
        interval_seconds: How often to check

    Returns:
        tuple: (thread, stop_event) - use stop_event.set() to stop the thread
    """
    stop_event = Event()

    monitor_thread = threading.Thread(
        target=alert_monitor_loop,
        args=(symbol, interval_seconds, stop_event),
        daemon=True,  # Thread will close when main program exits
    )

    monitor_thread.start()

    return monitor_thread, stop_event


async def main():
    console.print("Bells coming online...")

    bot_task = asyncio.create_task(start_bot())
    messenger_task = asyncio.create_task(messenger())
    while not is_bot_ready():
        await asyncio.sleep(0.2)

    monitor_thread, stop_event = start_alert_monitor_thread(
        symbol="QQQ", interval_seconds=60
    )

    console.print("[green]Alert monitor running in background thread.[/green]")
    console.print("[dim]Press Ctrl+C to stop...[/dim]")

    try:
        while True:
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]⚠️  Shutting down gracefully...[/yellow]")
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            console.print("[dim]✓ Bot task cancelled[/dim]")

        messenger_task.cancel()
        try:
            await messenger_task
        except asyncio.CancelledError:
            console.print("[dim]✓ Messenger cancelled[/dim]")

        stop_event.set()
        monitor_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish

        await stop_bot()

        console.print("[green]✓ Clean shutdown complete.[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
