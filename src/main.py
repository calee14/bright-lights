# src/main.py
import json
from pathlib import Path
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
from src.util.dashboard_client import send_to_dashboard
import asyncio


console = Console()
main_ticker = "QQQ"  # "MNQ=F"

active_reversion_alert = None
reversion_alert_timer = None


def reversion_alert(data, std=2.0, lookback=8):
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

    window = min(lookback, len(data) - 1)
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


def trend_alert(
    data,
    threshold=0.7,
    n_candles=7,
    decay_rate=0.98,
    roc_weight=0.3,
    volume_weight=0.2,
):
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

    candle_sizes = []
    for i in range(len(recent_data)):
        row = recent_data.iloc[i]
        candle_size = abs(row["Close"] - row["Open"])
        candle_sizes.append(candle_size)

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

        # Calculate volume multiplier
        volume_multiplier = 1.0
        if i >= 2:
            # Compare current volume to recent average
            prev_avg_volume = recent_data["Volume"].iloc[max(0, i - 3) : i].mean()
            current_volume = row["Volume"]

            if prev_avg_volume > 0:
                # Calculate volume change ratio
                volume_change = (current_volume - prev_avg_volume) / prev_avg_volume

                # Reward increasing volume, punish decreasing volume
                # Neutral zone: -10% to +10% change (no effect)
                if abs(volume_change) > 0.1:
                    if volume_change > 0:
                        # Volume increasing - reward
                        volume_multiplier = 1.0 + (
                            min(volume_change, 1.0) * volume_weight
                        )
                    else:
                        # Volume decreasing - punish
                        volume_multiplier = 1.0 + (
                            max(volume_change, -0.5) * volume_weight
                        )

        # Calculate rate of change
        # of price movement
        # and add it to score
        current_idx = recent_data.index[i]
        data_position = data.index.get_loc(current_idx)

        if data_position >= roc_period:
            prev_close = data.iloc[data_position - roc_period]["Close"]
            current_close = row["Close"]

            roc_raw = (
                abs((current_close - prev_close) / prev_close) if prev_close != 0 else 0
            )

            roc_normalized = max(min(roc_raw * 100, 1.0), -1.0)

            is_green_candle = row["Close"] >= row["Open"]

            if is_green_candle:
                # Green candle: boost for ROC > 0
                # and penalize for ROC < 0
                roc_multiplier = 1.0 + (roc_normalized * roc_weight)
            else:
                # Red candle: boost for ROC < 0
                # and penalize for ROC > 0
                if roc_normalized < 0:
                    roc_multiplier = 1.0 + (abs(roc_normalized) * roc_weight)
                else:
                    roc_multiplier = 1.0 - (roc_normalized * roc_weight)

        else:
            roc_multiplier = 1.0

        candle_score = price_move * time_weight * roc_multiplier * volume_multiplier

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
            "strength": "STRONG" if green_ratio >= 0.9 else "MODERATE",
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


def price_range_test_alert(data, range_pct=0.5, min_tests=3, lookback=20):
    """
    Detects when price is repeatedly testing a specific range,
    indicating potential support/resistance and imminent breakout.

    Args:
        data: Pandas dataframe
        range_pct: Percentage range to consider as same level (e.g., 0.5 = 0.5%)
        min_tests: Minimum number of tests required to trigger alert
        lookback: Number of candles to look back
    Returns:
        dict containing signal or None
    """

    recent_data = data.tail(lookback)
    current_price = data["Close"].iloc[-1]

    test_points = []

    for i in range(len(recent_data)):
        row = recent_data.iloc[i]
        test_points.append(
            {
                "price": row["High"],
                "type": "HIGH",
                "candle_idx": i,
                "volume": row["Volume"],
            }
        )
        test_points.append(
            {
                "price": row["Low"],
                "type": "LOW",
                "candle_idx": i,
                "volume": row["Volume"],
            }
        )

    # Find clusters of tests within range_pct
    def find_clusters(points, tolerance_pct):
        clusters = []
        used = set()

        for i, point in enumerate(points):
            if i in used:
                continue

            cluster = [point]
            used.add(i)
            base_price = point["price"]

            for j, other in enumerate(points):
                if j in used or j == i:
                    continue

                price_diff_pct = abs(other["price"] - base_price) / base_price * 100

                if price_diff_pct <= tolerance_pct:
                    cluster.append(other)
                    used.add(j)

            if len(cluster) >= min_tests:
                avg_price = sum(p["price"] for p in cluster) / len(cluster)
                avg_volume = sum(p["volume"] for p in cluster) / len(cluster)

                # Determine if resistance or support
                level_type = (
                    "RESISTANCE"
                    if all(p["type"] == "HIGH" for p in cluster[:3])
                    else "SUPPORT"
                    if all(p["type"] == "LOW" for p in cluster[:3])
                    else "MIXED"
                )

                clusters.append(
                    {
                        "avg_price": avg_price,
                        "num_tests": len(cluster),
                        "type": level_type,
                        "avg_volume": avg_volume,
                        "cluster": cluster,
                    }
                )

        return clusters

    clusters = find_clusters(test_points, range_pct)

    if not clusters:
        return None

    # Find the most relevant cluster
    # (closest to current price with most tests)
    clusters_with_score = []
    for cluster in clusters:
        distance_pct = abs(cluster["avg_price"] - current_price) / current_price * 100
        # Score: prioritize proximity and number of tests
        score = cluster["num_tests"] * 10 - distance_pct
        clusters_with_score.append((score, cluster))

    clusters_with_score.sort(reverse=True, key=lambda x: x[0])
    best_cluster = clusters_with_score[0][1]

    # Only alert if current price
    # is near the tested level
    distance_pct = abs(best_cluster["avg_price"] - current_price) / current_price * 100

    if distance_pct > range_pct * 2:
        return None

    # Check if tests are increasing in
    # frequency (acceleration)
    recent_tests = [
        p for p in best_cluster["cluster"] if p["candle_idx"] >= lookback - 5
    ]
    is_accelerating = len(recent_tests) >= min_tests - 1

    # Check volume on recent tests
    recent_volume = data["Volume"].tail(3).mean()
    avg_volume = data["Volume"].tail(lookback).mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

    # Determine urgency
    if best_cluster["num_tests"] >= 5 and is_accelerating:
        urgency = "CRITICAL"
    elif best_cluster["num_tests"] >= 4 or is_accelerating:
        urgency = "HIGH"
    else:
        urgency = "MODERATE"

    # Check if price is above or below the level
    position = "ABOVE" if current_price > best_cluster["avg_price"] else "BELOW"

    alert = {
        "type": "PRICE_RANGE_TEST",
        "level_type": best_cluster["type"],
        "level_price": best_cluster["avg_price"],
        "current_price": current_price,
        "num_tests": best_cluster["num_tests"],
        "distance_pct": distance_pct,
        "position": position,
        "urgency": urgency,
        "is_accelerating": is_accelerating,
        "volume_ratio": volume_ratio,
        "range_pct": range_pct,
        "lookback": lookback,
        "timestamp": data.index[-1]
        if hasattr(data.index[-1], "strftime")
        else datetime.now(),
    }

    return alert


def volume_anomaly_alert(data, threshold=2.1, n_candles=5):
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
    if len(data) < n_candles * 2:
        return None

    recent_candles = data.tail(n_candles)
    recent_avg_volume = recent_candles["Volume"].mean()

    historical_candles = data.iloc[-(n_candles * 2) : -n_candles]
    historical_avg_volume = historical_candles["Volume"].mean()

    if historical_avg_volume == 0:
        return None

    volume_ratio = recent_avg_volume / historical_avg_volume

    if volume_ratio >= threshold:
        volume_direction = "INCREASING"
    elif volume_ratio <= (1 / threshold):
        volume_direction = "DECREASING"
    else:
        return None

    recent_price_change = (
        (recent_candles["Close"].iloc[-1] - recent_candles["Close"].iloc[0])
        / recent_candles["Close"].iloc[0]
    ) * 100

    recent_price_start = recent_candles["Close"].iloc[0]
    recent_price_end = recent_candles["Close"].iloc[-1]
    recent_price_change_pct = (
        (recent_price_end - recent_price_start) / recent_price_start
    ) * 100

    # Determine price bias based on ROC
    if recent_price_change_pct > 0.1:
        price_bias = "BULLISH"
    elif recent_price_change_pct < -0.1:
        price_bias = "BEARISH"
    else:
        price_bias = "NEUTRAL"

    if volume_direction == "INCREASING" and price_bias == "BULLISH":
        interpretation = "STRONG_UPTREND"
        confidence = "HIGH"
    elif volume_direction == "INCREASING" and price_bias == "BEARISH":
        interpretation = "STRONG_DOWNTREND"
        confidence = "HIGH"
    elif volume_direction == "DECREASING" and price_bias == "BULLISH":
        interpretation = "UPTREND_WEAKENING"
        confidence = "MEDIUM"
    elif volume_direction == "DECREASING" and price_bias == "BEARISH":
        interpretation = "DOWNTREND_WEAKENING"
        confidence = "MEDIUM"
    else:
        interpretation = "UNCLEAR"
        confidence = "LOW"

    # Check if volume is accelerating within recent period
    first_half_volume = recent_candles["Volume"].iloc[: n_candles // 2].mean()
    second_half_volume = recent_candles["Volume"].iloc[n_candles // 2 :].mean()
    is_accelerating = second_half_volume > first_half_volume * 1.1  # 10% increase

    alert = {
        "type": "VOLUME_TREND",
        "volume_direction": volume_direction,
        "price_bias": price_bias,
        "interpretation": interpretation,
        "confidence": confidence,
        "recent_avg_volume": int(recent_avg_volume),
        "historical_avg_volume": int(historical_avg_volume),
        "volume_ratio": round(volume_ratio, 2),
        "threshold": threshold,
        "price_change_pct": round(recent_price_change, 2),
        "is_accelerating": is_accelerating,
        "recent_period": n_candles,
        "comparison_period": n_candles,
        "timestamp": data.index[-1]
        if hasattr(data.index[-1], "strftime")
        else datetime.now(),
    }

    return alert


def load_params_from_json(filename, default_params=None):
    """Load parameters from JSON file, return defaults if file doesn't exist"""
    filepath = Path("backtest_results") / filename

    if filepath.exists():
        with open(filepath, "r") as f:
            params = json.load(f)
        return params
    else:
        console.print(
            f"[yellow]No saved parameters found at {filepath}, using defaults[/yellow]"
        )
        return default_params if default_params else {}


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
            get_yahoo_finance_data(
                symbol, lookback=lookback, interval=interval, offset=offset
            )
        )

        plot_recent_candlesticks(data, filename="charts/tminus60.jpeg")

        # Load optimized parameters from JSON
        reversion_params = load_params_from_json(
            "reversion_alert_params.json", default_params={"std": 3.3, "lookback": 7}
        )
        trend_params = load_params_from_json(
            "trend_alert_params.json",
            default_params={
                "threshold": 0.9,
                "n_candles": 7,
                "decay_rate": 0.9,
                "roc_weight": 0.01,
                "volume_weight": 0.15,
            },
        )
        range_params = load_params_from_json(
            "price_range_test_alert_params.json",
            default_params={"range_pct": 0.06, "min_tests": 9, "lookback": 9},
        )
        volume_params = load_params_from_json(
            "volume_anomaly_alert_params.json",
            default_params={"threshold": 2.4, "n_candles": 8},
        )

        # Check for alerts
        reversion_signal = reversion_alert(
            data,
            std=reversion_params.get("std", 3.3),
            lookback=reversion_params.get("lookback", 7),
        )
        trend_signal = trend_alert(
            data,
            threshold=trend_params.get("threshold", 0.9),
            n_candles=trend_params.get("n_candles", 7),
            decay_rate=trend_params.get("decay_rate", 0.9),
            roc_weight=trend_params.get("roc_weight", 0.01),
            volume_weight=trend_params.get("volume_weight", 0.15),
        )
        range_test_signal = price_range_test_alert(
            data,
            range_pct=range_params.get("range_pct", 0.06),
            min_tests=range_params.get("min_tests", 9),
            lookback=range_params.get("lookback", 9),
        )
        volume_anomaly_signal = volume_anomaly_alert(
            data,
            threshold=volume_params.get("threshold", 2.4),
            n_candles=volume_params.get("n_candles", 8),
        )

        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "reversion_signal": reversion_signal,
            "trend_signal": trend_signal,
            "range_test_signal": range_test_signal,
            "volume_signal": volume_anomaly_signal,
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
    global active_reversion_alert, reversion_alert_timer

    if alerts is None:
        return

    reversion_signal = alerts.get("reversion_signal")
    trend_signal = alerts.get("trend_signal")
    range_test_signal = alerts.get("range_test_signal")
    volume_signal = alerts.get("volume_signal")

    if reversion_signal:
        if (
            active_reversion_alert is None
            or active_reversion_alert["direction"] != reversion_signal["direction"]
        ):
            if reversion_alert_timer is not None:
                reversion_alert_timer.cancel()

            def delayed_reversion_alert():
                global active_reversion_alert, reversion_alert_timer

                console.print("\n[bold red]🔔 MEAN REVERSION ALERT![/bold red]")
                console.print(
                    f"[dim]Time: {reversion_signal['timestamp'].strftime('%H:%M:%S')}[/dim]"
                )
                console.print(f"Direction: {reversion_signal['direction']}")
                console.print(
                    f"Current Price: ${reversion_signal['current_price']:.2f}"
                )
                console.print(f"Mean: ${reversion_signal['mean']:.2f}")
                console.print(
                    f"Deviation: {reversion_signal['deviation']:.2f} standard deviations"
                )
                console.print(
                    f"Volume Ratio: {reversion_signal['volume_ratio']:.2f}x average"
                )

                # Send to Discord
                discord_msg = (
                    f"🔔 **MEAN REVERSION ALERT!**\n"
                    f"Time: {reversion_signal['timestamp'].strftime('%H:%M:%S')}\n"
                    f"Direction: {reversion_signal['direction']}\n"
                    f"Current Price: ${reversion_signal['current_price']:.2f}\n"
                    f"Mean: ${reversion_signal['mean']:.2f}\n"
                    f"Deviation: {reversion_signal['deviation']:.2f} standard deviations\n"
                    f"Volume Ratio: {reversion_signal['volume_ratio']:.2f}x average"
                )
                message_queue.put(discord_msg)

                send_to_dashboard("reversion", reversion_signal)

                # Clear the timer reference after it fires
                reversion_alert_timer = None

            # Start a timer thread to execute after 5 minutes (300 seconds)
            reversion_alert_timer = threading.Timer(180.0, delayed_reversion_alert)
            reversion_alert_timer.daemon = True
            reversion_alert_timer.start()

            active_reversion_alert = reversion_signal
    else:
        active_reversion_alert = None

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

        send_to_dashboard("trend", trend_signal)

    if (
        range_test_signal
        and range_test_signal["level_type"] != "MIXED"
        and (
            (
                range_test_signal["level_type"] == "RESISTANCE"
                and range_test_signal["position"] == "BELOW"
            )
            or (
                range_test_signal["level_type"] == "SUPPORT"
                and range_test_signal["position"] == "ABOVE"
            )
        )
    ):
        # Color coding based on urgency
        urgency_colors = {
            "CRITICAL": "bold red",
            "HIGH": "bold yellow",
            "MODERATE": "bold blue",
        }
        color = urgency_colors.get(range_test_signal["urgency"], "bold blue")

        console.print(f"\n[{color}]🎯 PRICE RANGE TEST ALERT![/{color}]")
        console.print(
            f"[dim]Time: {range_test_signal['timestamp'].strftime('%H:%M:%S')}[/dim]"
        )
        console.print(f"Level Type: {range_test_signal['level_type']}")
        console.print(f"Level Price: ${range_test_signal['level_price']:.2f}")
        console.print(
            f"Current Price: ${range_test_signal['current_price']:.2f} ({range_test_signal['position']})"
        )
        console.print(f"Number of Tests: {range_test_signal['num_tests']}")
        console.print(f"Distance from Level: {range_test_signal['distance_pct']:.2f}%")
        console.print(f"Urgency: [{color}]{range_test_signal['urgency']}[/{color}]")

        if range_test_signal["is_accelerating"]:
            console.print(
                "[bold red]⚡ Tests are ACCELERATING - Move Imminent![/bold red]"
            )

        console.print(f"Volume Ratio: {range_test_signal['volume_ratio']:.2f}x average")

        # Neutral interpretation - no directional bias
        if range_test_signal["level_type"] == "RESISTANCE":
            action_msg = (
                "💡 Key resistance being tested - watch for breakout OR rejection"
            )
            console.print(f"[yellow]{action_msg}[/yellow]")
            action_text = f"\n{action_msg}"
        elif range_test_signal["level_type"] == "SUPPORT":
            action_msg = "💡 Key support being tested - watch for bounce OR breakdown"
            console.print(f"[yellow]{action_msg}[/yellow]")
            action_text = f"\n{action_msg}"

        # Send to Discord
        accelerating_text = ""
        if range_test_signal["is_accelerating"]:
            accelerating_text = "\n⚡ Tests are ACCELERATING - Move Imminent!"

        urgency_emoji = (
            "🚨"
            if range_test_signal["urgency"] == "CRITICAL"
            else "⚠️"
            if range_test_signal["urgency"] == "HIGH"
            else "ℹ️"
        )

        discord_msg = (
            f"🎯 **PRICE RANGE TEST ALERT!** {urgency_emoji}\n"
            f"Time: {range_test_signal['timestamp'].strftime('%H:%M:%S')}\n"
            f"**{range_test_signal['level_type']}** at ${range_test_signal['level_price']:.2f}\n"
            f"Current: ${range_test_signal['current_price']:.2f} ({range_test_signal['position']} level)\n"
            f"Tests: {range_test_signal['num_tests']} | Distance: {range_test_signal['distance_pct']:.2f}%\n"
            f"**Urgency: {range_test_signal['urgency']}**\n"
            f"Volume: {range_test_signal['volume_ratio']:.2f}x average"
            f"{accelerating_text}"
            f"{action_text}"
        )
        message_queue.put(discord_msg)

        send_to_dashboard("range", range_test_signal)

    if volume_signal and volume_signal["interpretation"] != "UNCLEAR":
        console.print("\n[bold yellow]🔔 VOLUME TREND ALERT![/bold yellow]")
        console.print(
            f"[dim]Time: {volume_signal['timestamp'].strftime('%H:%M:%S')}[/dim]"
        )
        console.print(f"Volume Direction: {volume_signal['volume_direction']}")
        console.print(f"Price Bias: {volume_signal['price_bias']}")
        console.print(f"Interpretation: [bold]{volume_signal['interpretation']}[/bold]")
        console.print(f"Confidence: {volume_signal['confidence']}")
        console.print(
            f"Volume Ratio: {volume_signal['volume_ratio']}x (Recent vs Historical)"
        )
        console.print(f"Price Change: {volume_signal['price_change_pct']:+.2f}%")

        if volume_signal["is_accelerating"]:
            console.print("[bold yellow]⚡ Volume is ACCELERATING![/bold yellow]")

        # Actionable interpretation
        action_text = ""
        if volume_signal["interpretation"] == "STRONG_UPTREND":
            action_msg = "💡 Consider LONG positions or hold longs"
            console.print(f"[green]{action_msg}[/green]")
            action_text = f"\n{action_msg}"
        elif volume_signal["interpretation"] == "STRONG_DOWNTREND":
            action_msg = "💡 Consider SHORT positions or exit longs"
            console.print(f"[red]{action_msg}[/red]")
            action_text = f"\n{action_msg}"
        elif volume_signal["interpretation"] == "UPTREND_WEAKENING":
            action_msg = "💡 Consider taking profits on longs"
            console.print(f"[yellow]{action_msg}[/yellow]")
            action_text = f"\n{action_msg}"
        elif volume_signal["interpretation"] == "DOWNTREND_WEAKENING":
            action_msg = "💡 Consider reducing shorts"
            console.print(f"[yellow]{action_msg}[/yellow]")
            action_text = f"\n{action_msg}"

        # Send to Discord
        accelerating_text = ""
        if volume_signal["is_accelerating"]:
            accelerating_text = "\n⚡ Volume is ACCELERATING!"

        discord_msg = (
            f"🔔 **VOLUME TREND ALERT!**\n"
            f"Time: {volume_signal['timestamp'].strftime('%H:%M:%S')}\n"
            f"Direction: {volume_signal['volume_direction']} | Price: {volume_signal['price_bias']}\n"
            f"**{volume_signal['interpretation']}** (Confidence: {volume_signal['confidence']})\n"
            f"Volume Ratio: {volume_signal['volume_ratio']}x\n"
            f"Price Change: {volume_signal['price_change_pct']:+.2f}%\n"
            f"{accelerating_text}"
            f"{action_text}"
        )
        message_queue.put(discord_msg)

        send_to_dashboard("volume", volume_signal)


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

        alerts = check_alerts(symbol=symbol, interval="3m", offset=40000)

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

    # Start main alert monitor (mean reversion, trend, range tests) - runs every 91 seconds
    monitor_thread, stop_event = start_alert_monitor_thread(
        symbol=main_ticker, interval_seconds=301
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

        # Stop both monitor threads
        stop_event.set()
        monitor_thread.join(timeout=2)
        console.print("[dim]✓ Main alert monitor stopped[/dim]")

        await stop_bot()

        console.print("[green]✓ Clean shutdown complete.[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
