# src/main.py
from time import sleep
from typing import List, Dict
from anthropic.types import ContentBlock
from src.util.bot import (
    message_queue,
    messenger,
    start_bot,
    stop_bot,
    is_bot_ready,
)
from src.util.feed import get_yahoo_finance_data, parse_yahoo_data
from src.util.plot import plot_recent_candlesticks
from src.util.vibe import chat, build_msg
from src.util.prompt import (
    STAGE1_TREND_ANALYSIS,
    STAGE2_LEVELS_ANALYSIS,
    STAGE3_REVERSAL_WATCH,
    generate_final_summary_prompt,
)

from concurrent.futures import Future, ThreadPoolExecutor
from rich.markdown import Markdown
from rich.console import Console
from datetime import datetime
from threading import Event
import asyncio
import json
import re
import argparse

console = Console()

# Thresholds for information alerts (not trade signals)
TREND_STRENGTH_ALERT_LEVEL = 70  # Alert when trend is particularly strong
REVERSAL_ALERT_LEVEL = 60  # Alert when reversal signals are meaningful
LEVEL_TEST_DISTANCE_PCT = 1.0  # Alert when within 1% of key level


# Single analysis call
def analyze_stage(prompt: str, charts: list[str] = [], context: str = "") -> str:
    """Execute a single stage of analysis"""
    if context:
        full_prompt = f"{context}\n\n{prompt}"
    else:
        full_prompt = prompt

    msg = build_msg([full_prompt], charts)
    res = chat(msg)

    return res[0].text


# Multi-stage pipeline for a single analyst
def pipeline_analyst(charts: list[str], analyst_id: int) -> Dict[str, str]:
    """
    Run full 3-stage analysis pipeline for one analyst.
    Returns dict with all stage outputs.
    """
    console.print(f"[dim]  → Analyst {analyst_id}: Stage 1 - Trend Analysis[/dim]")

    # STAGE 1: Trend Analysis
    stage1_output = analyze_stage(STAGE1_TREND_ANALYSIS, charts)

    console.print(f"[dim]  → Analyst {analyst_id}: Stage 2 - Key Levels Analysis[/dim]")

    # STAGE 2: Key Levels Analysis (with Stage 1 context)
    stage2_context = f"PREVIOUS ANALYSIS (Trend Characterization):\n{stage1_output}"
    stage2_output = analyze_stage(STAGE2_LEVELS_ANALYSIS, charts, stage2_context)

    console.print(f"[dim]  → Analyst {analyst_id}: Stage 3 - Reversal Watch[/dim]")

    # STAGE 3: Reversal Watch (with full context)
    stage3_context = (
        f"STAGE 1 - TREND:\n{stage1_output}\n\nSTAGE 2 - KEY LEVELS:\n{stage2_output}"
    )
    stage3_output = analyze_stage(STAGE3_REVERSAL_WATCH, charts, stage3_context)

    return {
        "analyst_id": analyst_id,
        "stage1_trend": stage1_output,
        "stage2_levels": stage2_output,
        "stage3_reversal": stage3_output,
    }


# Run multiple analysts in parallel
def run_pack_pipeline(charts: list[str], num_analysts: int = 2):
    """
    Run multiple analysts through the 3-stage pipeline concurrently.
    Each analyst completes all 3 stages independently.
    """
    console.print(
        f"[cyan]📊 Running {num_analysts} analysts through pipeline...[/cyan]"
    )

    with ThreadPoolExecutor(max_workers=num_analysts) as executor:
        futures = [
            executor.submit(pipeline_analyst, charts, i + 1)
            for i in range(num_analysts)
        ]

        results = [future.result() for future in futures]

    console.print(f"[green]✓ All analysts completed pipeline[/green]")
    return results


# Fetch data and generate charts
def ticker_scent(ticker: str):
    """Fetch data and plot charts for analysis"""
    charts = ["charts/tminus60.jpeg"]

    # 1-hour chart (1-minute candles, last 1 hour)
    tminus60 = parse_yahoo_data(
        get_yahoo_finance_data(ticker, lookback=3600, interval="1m")
    )
    plot_recent_candlesticks(tminus60, last_n_periods=len(tminus60), filename=charts[0])

    return charts


def extract_trend_data(trend_text: str) -> Dict:
    """
    Extract trend information from Stage 1 analysis.
    Returns dict with direction, strength, quality, duration.
    """
    data = {
        "direction": "SIDEWAYS",
        "strength": 0,
        "quality": "Unknown",
        "duration": 0,
    }

    # Extract trend direction
    if re.search(r"Trend Direction[:\*\s]*Uptrend", trend_text, re.IGNORECASE):
        data["direction"] = "UPTREND"
    elif re.search(r"Trend Direction[:\*\s]*Downtrend", trend_text, re.IGNORECASE):
        data["direction"] = "DOWNTREND"
    elif re.search(r"Trend Direction[:\*\s]*Sideways", trend_text, re.IGNORECASE):
        data["direction"] = "SIDEWAYS"

    # Extract trend strength
    strength_match = re.search(
        r"Total Score[:\*\s]*(\d{1,3})/100", trend_text, re.IGNORECASE
    )
    if strength_match:
        data["strength"] = int(strength_match.group(1))

    # Extract quality
    if re.search(r"Trend Quality[:\*\s]*Clean", trend_text, re.IGNORECASE):
        data["quality"] = "Clean"
    elif re.search(r"Trend Quality[:\*\s]*Choppy", trend_text, re.IGNORECASE):
        data["quality"] = "Choppy"
    elif re.search(r"Trend Quality[:\*\s]*Pausing", trend_text, re.IGNORECASE):
        data["quality"] = "Pausing"

    # Extract duration (number of candles)
    duration_match = re.search(
        r"(\d+)\s+candles in current trend", trend_text, re.IGNORECASE
    )
    if duration_match:
        data["duration"] = int(duration_match.group(1))

    return data


def extract_levels_data(levels_text: str) -> Dict:
    """
    Extract support/resistance levels from Stage 2 analysis.
    Returns dict with support/resistance levels and probabilities.
    """
    data = {
        "support_levels": [],
        "resistance_levels": [],
        "critical_level": None,
    }

    # Extract support levels
    support_pattern = r"Support \d+:\s*\$?(\d+\.?\d*)\s*.*?Strength Rating:\s*(\d+)/100.*?Hold Probability:\s*(\d+)%"
    for match in re.finditer(support_pattern, levels_text, re.IGNORECASE | re.DOTALL):
        data["support_levels"].append(
            {
                "price": float(match.group(1)),
                "strength": int(match.group(2)),
                "hold_probability": int(match.group(3)),
            }
        )

    # Extract resistance levels
    resistance_pattern = r"Resistance \d+:\s*\$?(\d+\.?\d*)\s*.*?Strength Rating:\s*(\d+)/100.*?Hold Probability:\s*(\d+)%"
    for match in re.finditer(
        resistance_pattern, levels_text, re.IGNORECASE | re.DOTALL
    ):
        data["resistance_levels"].append(
            {
                "price": float(match.group(1)),
                "strength": int(match.group(2)),
                "hold_probability": int(match.group(3)),
            }
        )

    return data


def extract_reversal_data(reversal_text: str) -> Dict:
    """
    Extract reversal information from Stage 3 analysis.
    Returns dict with reversal score and probability.
    """
    data = {
        "reversal_score": 0,
        "reversal_probability": 0,
        "warning_signs": [],
    }

    # Extract reversal score
    score_match = re.search(
        r"Reversal Score:\s*(\d{1,3})/100", reversal_text, re.IGNORECASE
    )
    if score_match:
        data["reversal_score"] = int(score_match.group(1))

    # Extract reversal probability
    prob_match = re.search(
        r"Reversal Probability:\s*(\d{1,3})%", reversal_text, re.IGNORECASE
    )
    if prob_match:
        data["reversal_probability"] = int(prob_match.group(1))

    return data


# Aggregate analysis from multiple analysts
def aggregate_analysis(analyst_results: List[Dict]) -> Dict:
    """
    Parse analysis from all analysts and create consensus view.
    Returns aggregate statistics for informational purposes.
    """
    trend_data = []
    levels_data = []
    reversal_data = []

    for result in analyst_results:
        # Extract data from each stage
        trend_info = extract_trend_data(result["stage1_trend"])
        levels_info = extract_levels_data(result["stage2_levels"])
        reversal_info = extract_reversal_data(result["stage3_reversal"])

        trend_data.append(trend_info)
        levels_data.append(levels_info)
        reversal_data.append(reversal_info)

    # Calculate trend consensus
    directions = [t["direction"] for t in trend_data]
    direction_counts = {d: directions.count(d) for d in set(directions)}
    consensus_direction = max(direction_counts, key=direction_counts.get)
    direction_agreement = direction_counts[consensus_direction] / len(directions)

    avg_trend_strength = sum(t["strength"] for t in trend_data) / len(trend_data)

    # Get most common quality
    qualities = [t["quality"] for t in trend_data]
    consensus_quality = max(set(qualities), key=qualities.count)

    # Calculate levels consensus
    all_supports = []
    all_resistances = []
    for levels in levels_data:
        all_supports.extend(levels["support_levels"])
        all_resistances.extend(levels["resistance_levels"])

    # Find strongest support (closest to current price with highest strength)
    strongest_support = None
    if all_supports:
        strongest_support = max(all_supports, key=lambda x: x["strength"])

    # Find strongest resistance
    strongest_resistance = None
    if all_resistances:
        strongest_resistance = max(all_resistances, key=lambda x: x["strength"])

    # Calculate reversal consensus
    avg_reversal_score = sum(r["reversal_score"] for r in reversal_data) / len(
        reversal_data
    )
    avg_reversal_prob = sum(r["reversal_probability"] for r in reversal_data) / len(
        reversal_data
    )

    return {
        "trend_consensus": {
            "direction": consensus_direction,
            "direction_agreement": direction_agreement * 100,
            "avg_strength": avg_trend_strength,
            "quality": consensus_quality,
        },
        "levels_consensus": {
            "strongest_support": strongest_support["price"] if strongest_support else 0,
            "support_strength": strongest_support["strength"]
            if strongest_support
            else 0,
            "support_hold_prob": strongest_support["hold_probability"]
            if strongest_support
            else 0,
            "strongest_resistance": strongest_resistance["price"]
            if strongest_resistance
            else 0,
            "resistance_strength": strongest_resistance["strength"]
            if strongest_resistance
            else 0,
            "resistance_hold_prob": strongest_resistance["hold_probability"]
            if strongest_resistance
            else 0,
            "current_price": 0,  # Will be filled in with actual current price
        },
        "reversal_consensus": {
            "avg_score": avg_reversal_score,
            "avg_probability": avg_reversal_prob,
        },
        "raw_data": {
            "trend_data": trend_data,
            "levels_data": levels_data,
            "reversal_data": reversal_data,
        },
    }


# Generate final informational summary
def generate_final_summary(
    aggregate: Dict, analyst_results: List[Dict], current_price: float
) -> str:
    """
    Create final informational summary without trade recommendations.
    """
    # Update current price in aggregate
    aggregate["levels_consensus"]["current_price"] = current_price

    # Build context from all analysts
    analyst_summaries = []
    for result in analyst_results:
        summary = (
            f"**Analyst {result['analyst_id']}:**\n\n"
            f"Trend: {result['stage1_trend']}\n\n"
            f"Levels: {result['stage2_levels']}\n\n"
            f"Reversal: {result['stage3_reversal']}\n"
        )
        analyst_summaries.append(summary)

    summary_prompt = generate_final_summary_prompt(
        trend_consensus=aggregate["trend_consensus"],
        levels_consensus=aggregate["levels_consensus"],
        reversal_consensus=aggregate["reversal_consensus"],
        analyst_summaries="\n---\n".join(analyst_summaries),
    )

    msg = build_msg([summary_prompt], [])
    final_output = chat(msg)[0].text

    return final_output


def generate_information_alerts(aggregate: Dict, current_price: float) -> List[str]:
    """
    Generate information alerts based on market conditions.
    These are NOT trade signals, but important market state notifications.
    """
    alerts = []

    trend = aggregate["trend_consensus"]
    levels = aggregate["levels_consensus"]
    reversal = aggregate["reversal_consensus"]

    # Alert 1: Strong trend detected
    if trend["avg_strength"] >= TREND_STRENGTH_ALERT_LEVEL:
        alerts.append(
            f"📊 Strong {trend['direction']} detected ({trend['avg_strength']:.0f}/100 strength)"
        )

    # Alert 2: Approaching key level
    support_price = levels["strongest_support"]
    resistance_price = levels["strongest_resistance"]

    if support_price > 0:
        distance_to_support_pct = abs(
            (current_price - support_price) / current_price * 100
        )
        if distance_to_support_pct <= LEVEL_TEST_DISTANCE_PCT:
            alerts.append(
                f"⚠️  Approaching key support at ${support_price:.2f} "
                f"({levels['support_hold_prob']:.0f}% hold probability)"
            )

    if resistance_price > 0:
        distance_to_resistance_pct = abs(
            (resistance_price - current_price) / current_price * 100
        )
        if distance_to_resistance_pct <= LEVEL_TEST_DISTANCE_PCT:
            alerts.append(
                f"⚠️  Approaching key resistance at ${resistance_price:.2f} "
                f"({levels['resistance_hold_prob']:.0f}% hold probability)"
            )

    # Alert 3: Reversal signals
    if reversal["avg_score"] >= REVERSAL_ALERT_LEVEL:
        alerts.append(
            f"🔄 Reversal signals present (Score: {reversal['avg_score']:.0f}/100, "
            f"Probability: {reversal['avg_probability']:.0f}%)"
        )

    return alerts


# Main monitoring loop
def pack_hunt(stop_event: Event):
    """Continuous loop running informational analysis"""
    while not stop_event.is_set():
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            console.print(
                f"\n[bold cyan]═══ Analysis Cycle: {current_time} ═══[/bold cyan]"
            )

            # 1. Fetch data and generate charts
            console.print("[cyan]📈 Fetching market data...[/cyan]")
            charts = ticker_scent("QQQ")

            # Get current price
            current_data = parse_yahoo_data(
                get_yahoo_finance_data("QQQ", lookback=3600, interval="1m")
            )
            current_price = (
                float(current_data.iloc[-1]["Close"]) if len(current_data) > 0 else 0
            )

            # 2. Run 2 analysts through full pipeline
            analyst_results = run_pack_pipeline(charts, num_analysts=2)

            # 3. Aggregate analysis
            console.print("[cyan]🧮 Aggregating analysis...[/cyan]")
            aggregate = aggregate_analysis(analyst_results)

            # Display aggregate stats
            console.print(
                f"[yellow]Trend: {aggregate['trend_consensus']['direction']} "
                f"(Strength: {aggregate['trend_consensus']['avg_strength']:.0f}/100, "
                f"Quality: {aggregate['trend_consensus']['quality']})[/yellow]"
            )
            console.print(
                f"[yellow]Reversal Watch: {aggregate['reversal_consensus']['avg_score']:.0f}/100 score, "
                f"{aggregate['reversal_consensus']['avg_probability']:.0f}% probability[/yellow]"
            )

            # 4. Generate information alerts
            alerts = generate_information_alerts(aggregate, current_price)

            if alerts:
                console.print(f"[bold yellow]📢 Information Alerts:[/bold yellow]")
                for alert in alerts:
                    console.print(f"[yellow]  • {alert}[/yellow]")
                    message_queue.put(alert)

            # 5. Generate and display final summary
            console.print("[cyan]📝 Generating market summary...[/cyan]")
            final_summary = generate_final_summary(
                aggregate, analyst_results, current_price
            )

            console.print(f"\n[bold green]═══ Market Summary ═══[/bold green]")
            console.print(Markdown(final_summary))

            # Send summary to bot
            message_queue.put(f"\n{final_summary}")

            # Wait before next cycle
            stop_event.wait(timeout=170)  # ~3 minutes between analyses

        except Exception as e:
            console.print(f"[red]✗ Error in analysis cycle: {e}[/red]")
            console.print(f"[red]{type(e).__name__}: {str(e)}[/red]")
            stop_event.wait(timeout=60)


def on_hunt_complete(_: Future):
    """Callback when pack_hunt completes"""
    try:
        console.print("\n[bold green]═══ ANALYSIS PERIOD ENDED ═══[/bold green]")
    except Exception as e:
        console.print(f"[red]✗ Analysis failed: {e}[/red]")


async def main():
    console.print("[bold cyan]🚀 Market Information System Starting...[/bold cyan]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")

    bot_task = asyncio.create_task(start_bot())

    while not is_bot_ready():
        await asyncio.sleep(0.5)

    messenger_task = asyncio.create_task(messenger())

    stop_event = Event()
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        hunt_future = executor.submit(pack_hunt, stop_event)
        hunt_future.add_done_callback(on_hunt_complete)

        while not stop_event.is_set():
            if hunt_future.done():
                console.print("[yellow]Analysis worker stopped[/yellow]")
                break

            await asyncio.sleep(5)

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]⚠️  Shutting down gracefully...[/yellow]")

    finally:
        console.print("[dim]Starting cleanup...[/dim]")

        stop_event.set()
        executor.shutdown(wait=True)
        console.print("[dim]✓ Executor shut down[/dim]")

        messenger_task.cancel()
        try:
            await messenger_task
        except asyncio.CancelledError:
            console.print("[dim]✓ Messenger cancelled[/dim]")

        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            console.print("[dim]✓ Bot task cancelled[/dim]")

        try:
            await stop_bot()
            console.print("[dim]✓ Bot stopped[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Error stopping bot: {e}[/yellow]")

        console.print("[bold green]✓ Shutdown complete[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Information System")

    parser.add_argument(
        "-ts",
        "--trend-strength",
        type=float,
        help="Alert threshold for trend strength (0-100, default 70)",
        default=70,
    )
    parser.add_argument(
        "-rs",
        "--reversal-score",
        type=float,
        help="Alert threshold for reversal score (0-100, default 60)",
        default=60,
    )
    parser.add_argument(
        "-ld",
        "--level-distance",
        type=float,
        help="Alert when within X%% of key level (default 1.0)",
        default=1.0,
    )

    args = parser.parse_args()

    TREND_STRENGTH_ALERT_LEVEL = args.trend_strength
    REVERSAL_ALERT_LEVEL = args.reversal_score
    LEVEL_TEST_DISTANCE_PCT = args.level_distance

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
