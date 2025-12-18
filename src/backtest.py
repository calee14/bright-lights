from src.main import (
    reversion_alert,
    trend_alert,
    volume_anomaly_alert,
    price_range_test_alert,
)
from typing import Any, List, Dict, Optional
from src.util.feed import get_yahoo_finance_data, parse_yahoo_data
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
import numpy as np

console = Console()


# Backtest the std_alert
# algorithm to find best parameters
def backtest_reversion_alert(
    data,
    lookback_range=range(6, 15, 1),
    std_range=np.arange(1.5, 10.0, 0.2),
    min_signals=5,
):
    best_win_rate = 0
    best_params = {}
    total_combinations = len(lookback_range) * len(list(std_range))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Backtesting std_alert...", total=total_combinations
        )
        for lookback in lookback_range:
            for std in std_range:
                wins = 0
                losses = 0
                total_return = 0

                for i in range(0, len(data) - lookback + 1, 5):
                    window = data.iloc[i : i + lookback]
                    signal = reversion_alert(window, std=std)
                    if signal:
                        forward_returns = calculate_forward_returns(
                            data, i + lookback, periods=3
                        )

                        if forward_returns is not None:
                            # Mean reversion: ABOVE means price is high, expect down
                            if signal["direction"] == "ABOVE":
                                expected_return = -forward_returns
                            else:
                                expected_return = forward_returns

                            total_return += expected_return

                            if expected_return > 0:
                                wins += 1
                            else:
                                losses += 1

                signal_count = wins + losses

                # Skip if not enough signals
                if signal_count < min_signals:
                    win_rate = 0
                    adjusted_win_rate = 0
                else:
                    win_rate = wins / signal_count

                    # Apply penalty for low sample sizes
                    sample_penalty = min(signal_count / (min_signals * 2), 1.0)
                    adjusted_win_rate = win_rate * sample_penalty

                if adjusted_win_rate > best_win_rate and signal_count >= min_signals:
                    best_win_rate = adjusted_win_rate
                    best_params = {
                        "lookback": lookback,
                        "std": std,
                        "wins": wins,
                        "losses": losses,
                        "signal_count": signal_count,
                        "win_rate": win_rate,
                        "adjusted_win_rate": adjusted_win_rate,
                        "total_return": total_return,
                        "avg_return": total_return / signal_count
                        if signal_count > 0
                        else 0,
                    }

                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]std_alert: lookback={lookback}, std={std:.1f}, WR={win_rate:.1%} ({signal_count})",
                )

    console.print(
        f"[green]✓ std_alert complete! Best adjusted win rate: {best_win_rate:.2%} "
        f"(raw: {best_params.get('win_rate', 0):.2%}, {best_params.get('wins', 0)}/{best_params.get('signal_count', 0)})[/green]"
    )

    return best_params


# Backtest the trend alert
# algorithm to find best parameters
def backtest_trend_alert(
    data,
    threshold_range=np.arange(0.9, 0.99, 0.02),
    n_candles_range=range(3, 9, 1),
    decay_rate_range=np.arange(0.85, 0.95, 0.02),
    roc_weight_range=np.arange(0.01, 0.5, 0.02),
    candle_size_weight_range=np.arange(0.01, 0.03, 0.01),
    min_signals=10,
):
    best_win_rate = 0
    best_params = {}

    total_combinations = (
        len(list(threshold_range))
        * len(n_candles_range)
        * len(list(decay_rate_range))
        * len(list(roc_weight_range))
        * len(list(candle_size_weight_range))
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[magenta]Backtesting trend_alert...", total=total_combinations
        )

        for threshold in threshold_range:
            for n_candles in n_candles_range:
                for decay_rate in decay_rate_range:
                    for roc_weight in roc_weight_range:
                        for candle_size_weight in candle_size_weight_range:
                            wins = 0
                            losses = 0
                            total_return = 0

                            for i in range(0, len(data) - n_candles + 1, 5):
                                window = data.iloc[i : i + n_candles]
                                signal = trend_alert(
                                    window,
                                    threshold=threshold,
                                    n_candles=n_candles,
                                    decay_rate=decay_rate,
                                    roc_weight=roc_weight,
                                    candle_size_weight=candle_size_weight,
                                )

                                if signal:
                                    forward_returns = calculate_forward_returns(
                                        data, i + n_candles, periods=5
                                    )

                                    if forward_returns is not None:
                                        if signal["direction"] == "UPTREND":
                                            expected_return = forward_returns
                                        else:
                                            expected_return = -forward_returns

                                        total_return += expected_return

                                        if expected_return > 0:
                                            wins += 1
                                        else:
                                            losses += 1

                            signal_count = wins + losses

                            # Skip if not enough signals
                            if signal_count < min_signals:
                                win_rate = 0
                                adjusted_win_rate = 0
                            else:
                                win_rate = wins / signal_count

                                # Apply penalty for low sample sizes
                                sample_penalty = min(
                                    signal_count / (min_signals * 2), 1.0
                                )
                                adjusted_win_rate = win_rate * sample_penalty

                            if (
                                adjusted_win_rate > best_win_rate
                                and signal_count >= min_signals
                            ):
                                best_win_rate = adjusted_win_rate
                                best_params = {
                                    "threshold": threshold,
                                    "n_candles": n_candles,
                                    "decay_rate": decay_rate,
                                    "roc_weight": roc_weight,
                                    "candle_size_weight": candle_size_weight,
                                    "wins": wins,
                                    "losses": losses,
                                    "signal_count": signal_count,
                                    "win_rate": win_rate,
                                    "adjusted_win_rate": adjusted_win_rate,
                                    "total_return": total_return,
                                    "avg_return": total_return / signal_count
                                    if signal_count > 0
                                    else 0,
                                }

                            progress.update(
                                task,
                                advance=1,
                                description=f"[magenta]trend_alert: thresh={threshold:.2f}, n={n_candles}, WR={win_rate:.1%} ({signal_count})",
                            )

    console.print(
        f"[green]✓ trend_alert complete! Best adjusted win rate: {best_win_rate:.2%} "
        f"(raw: {best_params.get('win_rate', 0):.2%}, {best_params.get('wins', 0)}/{best_params.get('signal_count', 0)})[/green]"
    )
    return best_params


# Backtest the resistance/support alert
# algorithm to find best parameters
def backtest_price_range_test_alert(
    data,
    range_pct_range=np.arange(0.02, 0.2, 0.02),
    min_tests_range=range(10, 30, 2),
    lookback_range=range(5, 25, 2),
    min_signals=5,
):
    best_win_rate = 0
    best_params = {}

    total_combinations = (
        len(list(range_pct_range)) * len(min_tests_range) * len(lookback_range)
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[yellow]Backtesting price_range_test_alert...", total=total_combinations
        )

        for range_pct in range_pct_range:
            for min_tests in min_tests_range:
                for lookback in lookback_range:
                    wins = 0
                    losses = 0
                    total_return = 0

                    for i in range(0, len(data) - lookback + 1, 5):
                        window = data.iloc[i : i + lookback]
                        signal = price_range_test_alert(
                            window,
                            range_pct=range_pct,
                            min_tests=min_tests,
                            lookback=lookback,
                        )

                        if signal:
                            forward_returns = calculate_forward_returns(
                                data, i + lookback, periods=3
                            )

                            if forward_returns is not None:
                                # Determine expected direction
                                if (
                                    signal["level_type"] == "RESISTANCE"
                                    and signal["position"] == "BELOW"
                                ):
                                    # Expect breakout up
                                    expected_return = forward_returns
                                elif (
                                    signal["level_type"] == "SUPPORT"
                                    and signal["position"] == "ABOVE"
                                ):
                                    # Expect breakdown down
                                    expected_return = -forward_returns
                                else:
                                    expected_return = 0
                                    # # Mixed signal - count any move as win
                                    # if signal["position"] == "BELOW":
                                    #     expected_return = forward_returns * 0.1
                                    # elif signal["position"] == "ABOVE":
                                    #     expected_return = -forward_returns * 0.1
                                    # else:
                                    #     expected_return = 0

                                total_return += expected_return

                                if expected_return > 0:
                                    wins += 1
                                elif expected_return < 0:
                                    losses += 1
                    signal_count = wins + losses
                    # Skip if not enough signals
                    if signal_count < min_signals:
                        win_rate = 0
                        adjusted_win_rate = 0
                    else:
                        win_rate = wins / signal_count

                        # Apply penalty for low sample sizes using Wilson score or simple penalty
                        # Option 1: Simple linear penalty
                        sample_penalty = min(signal_count / (min_signals * 2), 1.0)
                        adjusted_win_rate = win_rate * sample_penalty

                    if (
                        adjusted_win_rate > best_win_rate
                        and signal_count >= min_signals
                    ):
                        best_win_rate = adjusted_win_rate
                        best_params = {
                            "range_pct": range_pct,
                            "min_tests": min_tests,
                            "lookback": lookback,
                            "wins": wins,
                            "losses": losses,
                            "signal_count": signal_count,
                            "win_rate": adjusted_win_rate,
                            "total_return": total_return,
                            "avg_return": total_return / signal_count
                            if signal_count > 0
                            else 0,
                        }
                    progress.update(
                        task,
                        advance=1,
                        description=f"[yellow]price_range: range={range_pct:.1f}%, tests={min_tests}",
                    )

    console.print(
        f"[green]✓ price_range_test_alert complete! Best win rate: {best_win_rate:.2%} ({best_params.get('wins', 0)}/{best_params.get('signal_count', 0)})[/green]"
    )
    return best_params


# Backtest the volume anomaly alert
# algorithm to find best parameters
def backtest_volume_anomaly_alert(
    data,
    threshold_range=np.arange(1.5, 5.5, 0.5),
    n_candles_range=range(3, 10, 1),
    min_signals=5,
):
    best_win_rate = 0
    best_params = {}

    total_combinations = len(list(threshold_range)) * len(n_candles_range)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[blue]Backtesting volume_anomaly_alert...", total=total_combinations
        )

        for threshold in threshold_range:
            for n_candles in n_candles_range:
                wins = 0
                losses = 0
                total_return = 0
                for i in range(0, len(data) - (n_candles * 2) + 1, 5):
                    window = data.iloc[i : i + (n_candles * 2)]
                    signal = volume_anomaly_alert(
                        window, threshold=threshold, n_candles=n_candles
                    )

                    if signal:
                        forward_returns = calculate_forward_returns(
                            data, i + (n_candles * 2), periods=3
                        )

                        if forward_returns is not None:
                            # Determine expected direction based on interpretation
                            if signal["interpretation"] == "STRONG_UPTREND":
                                expected_return = forward_returns
                            elif signal["interpretation"] == "STRONG_DOWNTREND":
                                expected_return = -forward_returns
                            elif signal["interpretation"] == "UPTREND_WEAKENING":
                                expected_return = -forward_returns
                            elif signal["interpretation"] == "DOWNTREND_WEAKENING":
                                expected_return = forward_returns
                            else:
                                # UNCLEAR - skip
                                continue

                            total_return += expected_return

                            if expected_return > 0:
                                wins += 1
                            else:
                                losses += 1

                signal_count = wins + losses

                # Skip if not enough signals
                if signal_count < min_signals:
                    win_rate = 0
                    adjusted_win_rate = 0
                else:
                    win_rate = wins / signal_count

                    # Apply penalty for low sample sizes
                    sample_penalty = min(signal_count / (min_signals * 2), 1.0)
                    adjusted_win_rate = win_rate * sample_penalty

                if adjusted_win_rate > best_win_rate and signal_count >= min_signals:
                    best_win_rate = adjusted_win_rate
                    best_params = {
                        "threshold": threshold,
                        "n_candles": n_candles,
                        "wins": wins,
                        "losses": losses,
                        "signal_count": signal_count,
                        "win_rate": adjusted_win_rate,
                        "total_return": total_return,
                        "avg_return": total_return / signal_count
                        if signal_count > 0
                        else 0,
                    }

                progress.update(
                    task,
                    advance=1,
                    description=f"[blue]volume: threshold={threshold:.1f}, n={n_candles}, WR={win_rate:.1%}",
                )
    console.print(
        f"[green]✓ volume_anomaly_alert complete! Best win rate: {best_win_rate:.2%} ({best_params.get('wins', 0)}/{best_params.get('signal_count', 0)})[/green]"
    )
    return best_params


def calculate_forward_returns(data, start_idx, periods=3):
    if start_idx + periods >= len(data):
        return 0

    start_price = data.iloc[start_idx]["Close"]
    end_price = data.iloc[start_idx + periods]["Close"]

    return end_price - start_price


if __name__ == "__main__":
    # Get data from past 8 days
    data = parse_yahoo_data(
        get_yahoo_finance_data("IWM", lookback=691200, interval="3m")
    )

    reversion_params = backtest_reversion_alert(data)
    trend_params = backtest_trend_alert(data)
    price_range_params = backtest_price_range_test_alert(data)
    volume_params = backtest_volume_anomaly_alert(data)
    print(reversion_params)
    print(trend_params)
    print(price_range_params)
    print(volume_params)
