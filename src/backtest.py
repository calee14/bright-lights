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

main_ticker = "QQQ"
forward_period = 3
min_signals = 30
train_splits = 2


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

    return best_params


# Backtest the resistance/support alert
# algorithm to find best parameters
def backtest_price_range_test_alert(
    data,
    range_pct_range=np.arange(0.01, 1.0, 0.1),
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


def test_all_alerts(test_data, forward_period=3):
    """
    Test all alerts using previously calculated optimal parameters.
    Loads parameters from JSON files and runs backtests on test data.

    Args:
        test_data: DataFrame with test data
        forward_period: Number of periods to look forward for returns
    """
    import json
    from pathlib import Path
    from rich.table import Table

    results_dir = Path("backtest_results")

    # Dictionary to store all results
    all_results = {}

    # Load parameters and test each alert type
    alert_configs = [
        {
            "name": "reversion_alert",
            "file": "reversion_alert_params.json",
            "function": reversion_alert,
            "test_func": test_reversion_alert,
        },
        {
            "name": "trend_alert",
            "file": "trend_alert_params.json",
            "function": trend_alert,
            "test_func": test_trend_alert,
        },
        {
            "name": "price_range_test_alert",
            "file": "price_range_test_alert_params.json",
            "function": price_range_test_alert,
            "test_func": test_price_range_alert,
        },
        {
            "name": "volume_anomaly_alert",
            "file": "volume_anomaly_alert_params.json",
            "function": volume_anomaly_alert,
            "test_func": test_volume_alert,
        },
    ]

    console.print("\n[bold cyan]Testing Alerts on Out-of-Sample Data[/bold cyan]\n")

    for config in alert_configs:
        param_file = results_dir / config["file"]

        if not param_file.exists():
            console.print(
                f"[yellow]⚠ {config['name']}: Parameters file not found, skipping[/yellow]"
            )
            continue

        # Load parameters
        with open(param_file, "r") as f:
            params = json.load(f)

        # Run test
        console.print(f"[cyan]Testing {config['name']}...[/cyan]")
        result = config["test_func"](test_data, params, forward_period)
        all_results[config["name"]] = result

    # Display summary table
    display_results_table(all_results)

    return all_results


def test_reversion_alert(data, params, forward_period):
    """Test reversion alert with given parameters"""
    lookback = params["lookback"]
    std = params["std"]

    wins = 0
    losses = 0
    total_return = 0
    signals = []

    for i in range(0, len(data) - lookback + 1, 3):
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

                signals.append(
                    {"index": i + lookback, "signal": signal, "return": expected_return}
                )

    signal_count = wins + losses
    win_rate = wins / signal_count if signal_count > 0 else 0
    avg_return = total_return / signal_count if signal_count > 0 else 0

    return {
        "params": {"lookback": lookback, "std": std},
        "wins": wins,
        "losses": losses,
        "signal_count": signal_count,
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_return": avg_return,
        "signals": signals,
    }


def test_trend_alert(data, params, forward_period):
    """Test trend alert with given parameters"""
    threshold = params["threshold"]
    n_candles = params["n_candles"]
    decay_rate = params["decay_rate"]
    roc_weight = params["roc_weight"]
    volume_weight = params["volume_weight"]

    wins = 0
    losses = 0
    total_return = 0
    signals = []

    for i in range(0, len(data) - n_candles + 1, 3):
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

                signals.append(
                    {
                        "index": i + n_candles,
                        "signal": signal,
                        "return": expected_return,
                    }
                )

    signal_count = wins + losses
    win_rate = wins / signal_count if signal_count > 0 else 0
    avg_return = total_return / signal_count if signal_count > 0 else 0

    return {
        "params": {
            "threshold": threshold,
            "n_candles": n_candles,
            "decay_rate": decay_rate,
            "roc_weight": roc_weight,
            "volume_weight": volume_weight,
        },
        "wins": wins,
        "losses": losses,
        "signal_count": signal_count,
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_return": avg_return,
        "signals": signals,
    }


def test_price_range_alert(data, params, forward_period):
    """Test price range alert with given parameters"""
    range_pct = params["range_pct"]
    min_tests = params["min_tests"]
    lookback = params["lookback"]

    wins = 0
    losses = 0
    total_return = 0
    signals = []

    for i in range(0, len(data) - lookback + 1, 3):
        window = data.iloc[i : i + lookback]
        signal = price_range_test_alert(
            window,
            range_pct=range_pct,
            min_tests=min_tests,
            lookback=lookback,
        )

        if signal:
            forward_returns = calculate_forward_returns(
                data, i + lookback, periods=forward_period
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
                    signal["level_type"] == "SUPPORT" and signal["position"] == "ABOVE"
                ):
                    # Expect breakdown down
                    expected_return = -forward_returns
                else:
                    expected_return = 0

                total_return += expected_return

                if expected_return > 0:
                    wins += 1
                elif expected_return < 0:
                    losses += 1

                signals.append(
                    {"index": i + lookback, "signal": signal, "return": expected_return}
                )

    signal_count = wins + losses
    win_rate = wins / signal_count if signal_count > 0 else 0
    avg_return = total_return / signal_count if signal_count > 0 else 0

    return {
        "params": {
            "range_pct": range_pct,
            "min_tests": min_tests,
            "lookback": lookback,
        },
        "wins": wins,
        "losses": losses,
        "signal_count": signal_count,
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_return": avg_return,
        "signals": signals,
    }


def test_volume_alert(data, params, forward_period):
    """Test volume anomaly alert with given parameters"""
    threshold = params["threshold"]
    n_candles = params["n_candles"]

    wins = 0
    losses = 0
    total_return = 0
    signals = []

    for i in range(0, len(data) - (n_candles * 2) + 1, 3):
        window = data.iloc[i : i + (n_candles * 2)]
        signal = volume_anomaly_alert(window, threshold=threshold, n_candles=n_candles)

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

                signals.append(
                    {
                        "index": i + (n_candles * 2),
                        "signal": signal,
                        "return": expected_return,
                    }
                )

    signal_count = wins + losses
    win_rate = wins / signal_count if signal_count > 0 else 0
    avg_return = total_return / signal_count if signal_count > 0 else 0

    return {
        "params": {"threshold": threshold, "n_candles": n_candles},
        "wins": wins,
        "losses": losses,
        "signal_count": signal_count,
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_return": avg_return,
        "signals": signals,
    }


def display_results_table(all_results):
    """Display a formatted table of all test results"""
    from rich.table import Table

    table = Table(title="\n[bold]Out-of-Sample Test Results[/bold]", show_header=True)

    table.add_column("Alert Type", style="cyan", no_wrap=True)
    table.add_column("Signals", justify="right", style="magenta")
    table.add_column("Wins", justify="right", style="green")
    table.add_column("Losses", justify="right", style="red")
    table.add_column("Win Rate", justify="right", style="yellow")
    table.add_column("Total Return", justify="right", style="blue")
    table.add_column("Avg Return", justify="right", style="white")

    for name, result in all_results.items():
        win_rate_pct = f"{result['win_rate']:.1%}"
        total_return = f"{result['total_return']:+.2f}"
        avg_return = f"{result['avg_return']:+.4f}"

        # Color win rate based on performance
        if result["win_rate"] >= 0.55:
            win_rate_style = "[bold green]"
        elif result["win_rate"] >= 0.50:
            win_rate_style = "[green]"
        elif result["win_rate"] >= 0.45:
            win_rate_style = "[yellow]"
        else:
            win_rate_style = "[red]"

        table.add_row(
            name.replace("_", " ").title(),
            str(result["signal_count"]),
            str(result["wins"]),
            str(result["losses"]),
            f"{win_rate_style}{win_rate_pct}[/]",
            total_return,
            avg_return,
        )

    console.print(table)
    console.print()


def walk_forward_optimization(
    data, backtest_func, param_ranges, n_splits=5, train_ratio=0.7
):
    """
    Walk-forward optimization: train on one period, test on next period
    """
    data_len = len(data)
    split_size = data_len // (n_splits + 1)

    all_results = []

    console.print(f"\n[bold cyan]Walk-Forward Analysis ({n_splits} splits)[/bold cyan]")

    for split in range(n_splits):
        train_start = split * split_size
        train_end = train_start + int(split_size * train_ratio)
        test_start = train_end
        test_end = test_start + int(split_size * (1 - train_ratio))

        if test_end > data_len:
            break

        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_start:test_end].copy()

        console.print(f"\n[yellow]Split {split + 1}/{n_splits}:[/yellow]")
        console.print(f"  Train: {len(train_data)} candles")
        console.print(f"  Test:  {len(test_data)} candles")

        # Optimize on training data
        # backtest_func returns dict directly with all params AND metrics together
        train_result = backtest_func(train_data, **param_ranges)

        # Store the entire result
        all_results.append(train_result)

    # Average parameters weighted by performance
    return average_parameters(all_results)


def create_default_params(func_name):
    """Create default parameters when backtesting fails"""
    defaults = {
        "backtest_reversion_alert": {
            "lookback": 10,
            "std": 2.0,
            "wins": 0,
            "losses": 0,
            "signal_count": 0,
            "win_rate": 0.0,
            "adjusted_win_rate": 0.0,
            "total_return": 0.0,
            "avg_return": 0.0,
        },
        "backtest_trend_alert": {
            "threshold": 0.95,
            "n_candles": 7,
            "decay_rate": 0.85,
            "roc_weight": 0.2,
            "volume_weight": 0.2,
            "wins": 0,
            "losses": 0,
            "signal_count": 0,
            "win_rate": 0.0,
            "adjusted_win_rate": 0.0,
            "total_return": 0.0,
            "avg_return": 0.0,
        },
        "backtest_price_range_test_alert": {
            "range_pct": 0.1,
            "min_tests": 15,
            "lookback": 15,
            "wins": 0,
            "losses": 0,
            "signal_count": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "avg_return": 0.0,
        },
        "backtest_volume_anomaly_alert": {
            "threshold": 3.0,
            "n_candles": 7,
            "wins": 0,
            "losses": 0,
            "signal_count": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "avg_return": 0.0,
        },
    }

    return defaults.get(func_name, {})


def average_parameters(results):
    """
    Average parameters across splits, weighted by performance.
    """
    if not results:
        console.print("[yellow]Warning: No results to average[/yellow]")
        return {}

    results = list(filter(lambda x: x != {}, results))

    # Keys that are metrics (not parameters to average)
    metric_keys = {
        "wins",
        "losses",
        "signal_count",
        "win_rate",
        "adjusted_win_rate",
        "total_return",
        "avg_return",
    }

    # Get parameter keys (everything that's NOT a metric)
    all_keys = results[0].keys()
    param_keys = [k for k in all_keys if k not in metric_keys]

    console.print(f"\n[cyan]Averaging parameters: {param_keys}[/cyan]")

    averaged = {}

    # Calculate weights (use adjusted_win_rate if available, else win_rate)
    weights = []
    for r in results:
        weight = r.get("adjusted_win_rate", r.get("win_rate", 0))
        weights.append(max(weight, 0))  # Ensure non-negative

    total_weight = sum(weights)

    if total_weight == 0 or len(results) == 0:
        console.print(
            "[yellow]Warning: All weights are zero, using simple average[/yellow]"
        )
        # Fall back to simple average
        for key in param_keys:
            values = [r.get(key, 0) for r in results]  # Use .get() with default
            averaged[key] = np.mean(values) if values else 0
    else:
        # Weighted average
        for key in param_keys:
            weighted_sum = sum(
                r.get(key, 0) * weights[i] for i, r in enumerate(results)
            )
            averaged[key] = weighted_sum / total_weight

    # Round integer parameters
    int_params = ["lookback", "n_candles", "min_tests"]
    for key in int_params:
        if key in averaged:
            averaged[key] = int(round(averaged[key]))

    # Add stability metrics with safe defaults
    win_rates = [r.get("win_rate", 0) for r in results]
    averaged["avg_win_rate"] = np.mean(win_rates) if win_rates else 0.0
    averaged["std_win_rate"] = np.std(win_rates) if len(win_rates) > 1 else 0.0
    averaged["stability_score"] = averaged["avg_win_rate"] - averaged["std_win_rate"]

    # Store individual results for reference
    averaged["split_results"] = results

    console.print(f"\n[green]✓ Walk-Forward Complete:[/green]")
    console.print(f"  Average Win Rate: {averaged['avg_win_rate']:.2%}")
    console.print(f"  Consistency (±): {averaged['std_win_rate']:.2%}")
    console.print(f"  Stability Score: {averaged['stability_score']:.2%}")

    # Show averaged parameter values
    console.print(f"\n[green]Averaged Parameters:[/green]")
    for key in param_keys:
        if isinstance(averaged.get(key), float):
            console.print(f"  {key}: {averaged[key]:.4f}")
        else:
            console.print(f"  {key}: {averaged.get(key, 'N/A')}")

    return averaged


if __name__ == "__main__":
    # Get data from past 8 days (691200 seconds)
    data = parse_yahoo_data(
        get_yahoo_finance_data(main_ticker, lookback=691200, interval="3m", offset=0)
    )

    reversion_params = backtest_reversion_alert(data, min_signals=min_signals)
    reversion_params = walk_forward_optimization(
        data=data,
        backtest_func=backtest_reversion_alert,
        param_ranges={
            "lookback_range": range(6, 15, 1),
            "std_range": np.arange(1.5, 10.0, 0.2),
        },
        n_splits=train_splits,
    )
    # trend_params = backtest_trend_alert(data, min_signals=min_signals)
    trend_params = walk_forward_optimization(
        data=data,
        backtest_func=backtest_trend_alert,
        param_ranges={
            "threshold_range": np.arange(0.9, 0.99, 0.02),
            "n_candles_range": range(3, 13, 2),
            "decay_rate_range": np.arange(0.80, 0.95, 0.05),
            "roc_weight_range": np.arange(0.1, 0.33, 0.08),
            "volume_weight_range": np.arange(0.1, 0.33, 0.08),
            "forward_period": 5,
            "use_multiprocessing": True,
        },
        n_splits=train_splits,
    )
    # price_range_params = backtest_price_range_test_alert(data, min_signals=min_signals)
    price_range_params = walk_forward_optimization(
        data=data,
        backtest_func=backtest_price_range_test_alert,
        param_ranges={
            "range_pct_range": np.arange(0.02, 0.2, 0.02),
            "min_tests_range": range(5, 30, 2),
            "lookback_range": range(5, 25, 2),
        },
        n_splits=train_splits,
    )
    # volume_params = backtest_volume_anomaly_alert(data, min_signals=min_signals)
    volume_params = walk_forward_optimization(
        data=data,
        backtest_func=backtest_volume_anomaly_alert,
        param_ranges={
            "threshold_range": np.arange(1.5, 8, 0.2),
            "n_candles_range": range(3, 13, 1),
        },
        n_splits=train_splits,
    )

    save_params_to_json(reversion_params, "reversion_alert_params.json")
    save_params_to_json(trend_params, "trend_alert_params.json")
    save_params_to_json(price_range_params, "price_range_test_alert_params.json")
    save_params_to_json(volume_params, "volume_anomaly_alert_params.json")
    # print(reversion_params)
    # print(trend_params)
    # print(price_range_params)
    # print(volume_params)

    test_data = parse_yahoo_data(
        get_yahoo_finance_data(main_ticker, lookback=691200, interval="3m", offset=0)
    )

    all_results = test_all_alerts(test_data, forward_period=1)
    all_results = test_all_alerts(test_data, forward_period=3)
    all_results = test_all_alerts(test_data, forward_period=5)
