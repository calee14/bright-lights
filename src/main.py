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
    STAGE1_PATTERN_RECOGNITION,
    STAGE2_CONFLUENCE_ANALYSIS,
    STAGE3_PREDICTION,
    generate_final_alert_prompt,
)
from concurrent.futures import Future, ThreadPoolExecutor
from rich.markdown import Markdown
from rich.console import Console
from datetime import datetime
from threading import Event
import asyncio
import json

console = Console()


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
    console.print(f"[dim]  → Analyst {analyst_id}: Stage 1 - Pattern Recognition[/dim]")

    # STAGE 1: Pattern Recognition
    stage1_output = analyze_stage(STAGE1_PATTERN_RECOGNITION, charts)

    console.print(f"[dim]  → Analyst {analyst_id}: Stage 2 - Confluence Analysis[/dim]")

    # STAGE 2: Confluence Analysis (with Stage 1 context)
    stage2_context = f"PREVIOUS ANALYSIS (Patterns Identified):\n{stage1_output}"
    stage2_output = analyze_stage(STAGE2_CONFLUENCE_ANALYSIS, charts, stage2_context)

    console.print(f"[dim]  → Analyst {analyst_id}: Stage 3 - Prediction[/dim]")

    # STAGE 3: Prediction (with full context)
    stage3_context = (
        f"STAGE 1 - PATTERNS:\n{stage1_output}\n\n"
        f"STAGE 2 - CONFLUENCE:\n{stage2_output}"
    )
    stage3_output = analyze_stage(STAGE3_PREDICTION, charts, stage3_context)

    return {
        "analyst_id": analyst_id,
        "stage1_patterns": stage1_output,
        "stage2_confluence": stage2_output,
        "stage3_prediction": stage3_output,
    }


# Run multiple analysts in parallel (each doing full 3-stage pipeline)
def run_pack_pipeline(charts: list[str], num_analysts: int = 3):
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
    charts = ["charts/tminus60.jpeg", "charts/tminus360.jpeg"]

    # 1-hour chart (1-minute candles, last 60)
    tminus60 = parse_yahoo_data(
        get_yahoo_finance_data(ticker, lookback=3600, interval="1m")
    )
    plot_recent_candlesticks(tminus60, last_n_periods=len(tminus60), filename=charts[0])

    # 5-hour chart (5-minute candles)
    tminus360 = parse_yahoo_data(
        get_yahoo_finance_data(ticker, lookback=10800, interval="5m")
    )
    plot_recent_candlesticks(
        tminus360, last_n_periods=len(tminus360), filename=charts[1]
    )

    return charts


# Aggregate predictions from multiple analysts
def aggregate_predictions(analyst_results: List[Dict]) -> Dict:
    """
    Parse predictions from all analysts and create consensus view.
    Returns aggregate statistics and most confident prediction.
    """
    predictions = []

    for result in analyst_results:
        prediction_text = result["stage3_prediction"]

        # Simple parsing - you can make this more sophisticated
        direction = None
        confidence = 0

        if "Direction: Up" in prediction_text or "Direction:** Up" in prediction_text:
            direction = "UP"
        elif (
            "Direction: Down" in prediction_text
            or "Direction:** Down" in prediction_text
        ):
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # Extract confidence percentage (look for patterns like "70%" or "Confidence: 65%")
        import re

        conf_match = re.search(r"(\d{1,3})%", prediction_text)
        if conf_match:
            confidence = int(conf_match.group(1))

        predictions.append(
            {
                "analyst_id": result["analyst_id"],
                "direction": direction,
                "confidence": confidence,
                "full_analysis": result,
            }
        )

    # Calculate consensus
    up_votes = sum(1 for p in predictions if p["direction"] == "UP")
    down_votes = sum(1 for p in predictions if p["direction"] == "DOWN")
    neutral_votes = sum(1 for p in predictions if p["direction"] == "NEUTRAL")

    avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)

    # Determine consensus direction
    if up_votes > down_votes and up_votes > neutral_votes:
        consensus = "UP"
    elif down_votes > up_votes and down_votes > neutral_votes:
        consensus = "DOWN"
    else:
        consensus = "NEUTRAL"

    return {
        "consensus_direction": consensus,
        "vote_breakdown": {
            "UP": up_votes,
            "DOWN": down_votes,
            "NEUTRAL": neutral_votes,
        },
        "avg_confidence": avg_confidence,
        "predictions": predictions,
        "agreement_level": max(up_votes, down_votes, neutral_votes) / len(predictions),
    }


# Generate final summary with consensus
def generate_final_summary(aggregate: Dict, analyst_results: List[Dict]) -> str:
    """
    Create final summary prompt and get consolidated output.
    Only triggers BLUEHORSESHOE if high confidence + agreement.
    """
    consensus = aggregate["consensus_direction"]
    agreement = aggregate["agreement_level"]
    avg_conf = aggregate["avg_confidence"]

    # Build context from all analysts
    analyst_summaries = []
    for result in analyst_results:
        summary = (
            f"**Analyst {result['analyst_id']}:**\n{result['stage3_prediction']}\n"
        )
        analyst_summaries.append(summary)

    summary_prompt = generate_final_alert_prompt(
        consensus=consensus,
        agreement_pct=agreement * 100,
        avg_confidence=avg_conf,
        analyst_summaries="\n---\n".join(analyst_summaries),
    )

    msg = build_msg([summary_prompt], [])
    final_output = chat(msg)[0].text

    # Only add BLUEHORSESHOE prefix if criteria met
    if agreement >= 0.67 and avg_conf >= 65:  # 2/3 agreement + 65% avg confidence
        return f"BLUEHORSESHOE\n\n{final_output}"
    else:
        return final_output


# Main hunting loop
def pack_hunt(stop_event: Event):
    """Continuous loop running analysis pipeline"""
    while not stop_event.is_set():
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            console.print(
                f"\n[bold cyan]═══ Analysis Cycle: {current_time} ═══[/bold cyan]"
            )

            # 1. Fetch data and generate charts
            console.print("[cyan]📈 Fetching market data...[/cyan]")
            charts = ticker_scent("QQQ")

            # 2. Run 3 analysts through full pipeline
            analyst_results = run_pack_pipeline(charts, num_analysts=3)

            # 3. Aggregate predictions
            console.print("[cyan]🧮 Aggregating predictions...[/cyan]")
            aggregate = aggregate_predictions(analyst_results)

            console.print(
                f"[yellow]Consensus: {aggregate['consensus_direction']} | "
                f"Agreement: {aggregate['agreement_level'] * 100:.0f}% | "
                f"Avg Confidence: {aggregate['avg_confidence']:.0f}%[/yellow]"
            )

            # 4. Generate final summary
            console.print("[cyan]📝 Generating final summary...[/cyan]")
            final_summary = generate_final_summary(aggregate, analyst_results)

            # 5. Output and alert if criteria met
            if "BLUEHORSESHOE" in final_summary:
                console.print(f"\n[bold green]🎯 HIGH CONFIDENCE SIGNAL[/bold green]")
                console.print(Markdown(final_summary))
                message_queue.put(final_summary)
            else:
                console.print(f"[dim]ℹ️  No high-confidence signal this cycle[/dim]")

            # Wait before next cycle
            stop_event.wait(timeout=130)

        except Exception as e:
            console.print(f"[red]✗ Error in hunt: {e}[/red]")
            console.print(f"[red]{type(e).__name__}: {str(e)}[/red]")
            stop_event.wait(timeout=60)  # Wait before retry


def on_hunt_complete(_: Future):
    """Callback when pack_hunt completes"""
    try:
        console.print("\n[bold green]═══ ANALYSIS PERIOD ENDED ═══[/bold green]")
    except Exception as e:
        console.print(f"[red]✗ Pack hunt failed: {e}[/red]")


async def main():
    console.print("[bold cyan]🚀 Pack Analysis System Starting...[/bold cyan]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")

    bot_task = asyncio.create_task(start_bot())

    while not is_bot_ready():
        await asyncio.sleep(0.5)

    messenger_task = asyncio.create_task(messenger())

    stop_event = Event()
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        hunt_future = executor.submit(pack_hunt, stop_event)
        hunt_future.add_done_callback(on_hunt_complete)

        while not stop_event.is_set():
            if hunt_future.done():
                console.print("[yellow]Hunt worker stopped[/yellow]")
                break

            await asyncio.sleep(5)  # Check every 5 seconds

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
