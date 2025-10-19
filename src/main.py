from time import sleep
from typing import List
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
from src.util.prompt import ANALYZE_PROMPT, generate_summary_prompt
from concurrent.futures import Future, ThreadPoolExecutor
from rich.markdown import Markdown
from rich.console import Console
from datetime import datetime
from threading import Event
import asyncio

console = Console()


# chat with model
def pack_member(prompt: str, charts: list[str] = []) -> List[ContentBlock]:
    msg = build_msg([prompt], charts)

    res = chat(msg)

    return res


# exec concurrent model api calls
def run_pack(prompt: str, charts: list[str], num_calls: int):
    with ThreadPoolExecutor(max_workers=num_calls) as executor:
        futures = [
            executor.submit(pack_member, prompt, charts) for _ in range(num_calls)
        ]

        results = [future.result() for future in futures]

    return results


# fetch data
def ticker_scent(ticker: str):
    charts = ["charts/tminus60.jpeg", "charts/tminus360.jpeg"]
    tminus60 = parse_yahoo_data(
        get_yahoo_finance_data(ticker, lookback=3600, interval="1m")
    )

    plot_recent_candlesticks(tminus60, last_n_periods=len(tminus60), filename=charts[0])

    tminus360 = parse_yahoo_data(
        get_yahoo_finance_data(ticker, lookback=10800, interval="5m")
    )

    plot_recent_candlesticks(
        tminus360, last_n_periods=len(tminus360), filename=charts[1]
    )

    return charts


# continuous loop to run hunts
def pack_hunt(stop_event: Event):
    while not stop_event.is_set():
        try:
            files = ticker_scent("SPY")

            results = run_pack(ANALYZE_PROMPT, files, num_calls=3)

            analysis = []
            for r in results:
                cb = r[0]
                analysis.append(cb.text)

            summary_prompt = generate_summary_prompt(analysis)

            final_summary = pack_member(summary_prompt)[0].text

            # process summary and analysis
            if "BLUEHORSESHOE" in final_summary:
                current_time = datetime.now().strftime("%H:%M:%S")
                console.print(f"[dim]⏰ {current_time}")
                console.print(Markdown(final_summary))

                # add msg to queue for bot to send
                message_queue.put(final_summary)

            stop_event.wait(timeout=130)
        except Exception as e:
            console.log(f"[red]Error in hunt: {e}[/red]")
            break


def on_hunt_complete(_: Future):
    """Callback when pack_hunt completes"""
    try:
        console.print("\n[bold green]═══ ANALYSIS PERIOD ENDED ═══[/bold green]")
        console.print("[bold green]═════════════════════════[/bold green]\n")
    except Exception as e:
        console.print(f"[red]✗ Pack hunt failed: {e}[/red]")


async def main():
    console.print("[bold cyan]🚀[/bold cyan]")
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

            # continue if hunt is still going
            current_time = datetime.now().strftime("%H:%M:%S")
            console.print(f"[dim]⏰ {current_time} - System running...[/dim]", end="\r")

            # sleep to avoid busy-waiting
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    finally:
        console.print("[dim]Starting cleanup...[/dim]")

        stop_event.set()
        executor.shutdown(wait=True)
        console.print("[dim]Executor shut down[/dim]")

        # Cancel messenger
        messenger_task.cancel()
        try:
            await messenger_task
        except asyncio.CancelledError:
            console.print("[dim]Messenger cancelled[/dim]")

        # Cancel bot
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            console.print("[dim]Bot task cancelled[/dim]")
        except Exception as e:
            console.print(f"[red]Bot error: {e}[/red]")

        # Stop bot gracefully
        try:
            await stop_bot()
            console.print("[dim]Bot stopped[/dim]")
        except Exception as e:
            console.print(f"[yellow]Error stopping bot: {e}[/yellow]")

        console.print("[green]✓ Shutdown complete[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[green]Program terminated[/green]")
