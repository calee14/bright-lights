import json
from pathlib import Path
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
from multiprocessing import Pool, cpu_count
from itertools import product
import numpy as np

console = Console()

forward_period = 3


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
                            data, i + lookback, periods=forward_period
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

    save_params_to_json(best_params, "reversion_alert_params.json")

    return best_params


# Backtest the trend alert
# algorithm to find best parameters
def backtest_single_trend_combination(args):
    """Test a single parameter combination for trend_alert"""
    (
        threshold,
        n_candles,
        decay_rate,
        roc_weight,
        volume_weight,
        data_dict,
        forward_period,
        min_signals,
    ) = args

    # Reconstruct DataFrame from dict (needed for multiprocessing)
    import pandas as pd

    data = pd.DataFrame(data_dict)

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
            volume_weight=volume_weight,
        )

        if signal:
            forward_returns = calculate_forward_returns(
                data, i + n_candles, periods=forward_period
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

    if signal_count < min_signals:
        return None

    win_rate = wins / signal_count
    sample_penalty = min(signal_count / (min_signals * 2), 1.0)
    adjusted_win_rate = win_rate * sample_penalty

    return {
        "threshold": float(threshold),
        "n_candles": int(n_candles),
        "decay_rate": float(decay_rate),
        "roc_weight": float(roc_weight),
        "volume_weight": float(volume_weight),
        "wins": wins,
        "losses": losses,
        "signal_count": signal_count,
        "win_rate": win_rate,
        "adjusted_win_rate": adjusted_win_rate,
        "total_return": total_return,
        "avg_return": total_return / signal_count,
    }


def backtest_trend_alert(
    data,
    threshold_range=np.arange(0.9, 0.99, 0.02),
    n_candles_range=range(3, 13, 2),
    decay_rate_range=np.arange(0.80, 0.95, 0.05),
    roc_weight_range=np.arange(0.1, 0.33, 0.08),
    volume_weight_range=np.arange(0.1, 0.33, 0.08),
    min_signals=5,
    forward_period=5,
    use_multiprocessing=True,
):
    best_win_rate = 0
    best_params = {}

    # Create all combinations
    combinations = list(
        product(
            threshold_range,
            n_candles_range,
            decay_rate_range,
            roc_weight_range,
            volume_weight_range,
        )
    )

    total_combinations = len(combinations)

    if use_multiprocessing:
        # Convert DataFrame to dict for multiprocessing (picklable)
        data_dict = data.to_dict("list")

        # Add data and other args to each combination
        args_list = [
            (t, n, d, r, v, data_dict, forward_period, min_signals)
            for t, n, d, r, v in combinations
        ]

        num_cores = cpu_count()
        console.print(
            f"[cyan]Testing {total_combinations} combinations using {num_cores} cores...[/cyan]"
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

            # Use multiprocessing
            with Pool(num_cores) as pool:
                for result in pool.imap_unordered(
                    backtest_single_trend_combination, args_list, chunksize=10
                ):
                    if result and result["adjusted_win_rate"] > best_win_rate:
                        best_win_rate = result["adjusted_win_rate"]
                        best_params = result

                    progress.update(
                        task,
                        advance=1,
                        description=f"[magenta]trend_alert: best WR={best_win_rate:.1%} ({best_params.get('signal_count', 0)} signals)",
                    )
    else:
        # Original single-threaded version
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
                            for volume_weight in volume_weight_range:
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
                                        volume_weight=volume_weight,
                                    )

                                    if signal:
                                        forward_returns = calculate_forward_returns(
                                            data, i + n_candles, periods=forward_period
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

                                if signal_count < min_signals:
                                    win_rate = 0
                                    adjusted_win_rate = 0
                                else:
                                    win_rate = wins / signal_count
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
                                        "volume_weight": volume_weight,
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

    save_params_to_json(best_params, "trend_alert_params.json")
    return best_params


# Backtest the resistance/support alert
# algorithm to find best parameters
def backtest_price_range_test_alert(
    data,
    range_pct_range=np.arange(0.02, 0.2, 0.02),
    min_tests_range=range(5, 30, 2),
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

    save_params_to_json(best_params, "price_range_test_alert_params.json")
    return best_params


# Backtest the volume anomaly alert
# algorithm to find best parameters
def backtest_volume_anomaly_alert(
    data,
    threshold_range=np.arange(1.5, 8, 0.2),
    n_candles_range=range(3, 13, 1),
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
                            data, i + (n_candles * 2), periods=forward_period
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

    save_params_to_json(best_params, "volume_anomaly_alert_params.json")

    return best_params


def calculate_forward_returns(data, start_idx, periods=3):
    if start_idx + periods >= len(data):
        return 0

    start_price = data.iloc[start_idx]["Close"]
    end_price = data.iloc[start_idx + periods]["Close"]

    return end_price - start_price


def save_params_to_json(params, filename):
    """Save parameters to JSON file"""
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(params, f, indent=2)

    console.print(f"[green]✓ Saved parameters to {filepath}[/green]")


if __name__ == "__main__":
    # Get data from past 8 days
    data = parse_yahoo_data(
        get_yahoo_finance_data("MNQ=F", lookback=691200, interval="3m")
    )

    reversion_params = backtest_reversion_alert(data)
    trend_params = backtest_trend_alert(data)
    price_range_params = backtest_price_range_test_alert(data)
    volume_params = backtest_volume_anomaly_alert(data)
    print(reversion_params)
    print(trend_params)
    print(price_range_params)
    print(volume_params)
