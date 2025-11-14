from concurrent.futures import ThreadPoolExecutor
from threading import Event
from rich.console import Console
from src.util.bot import is_bot_ready, send_to_general, start_bot
from src.util.feed import get_yahoo_finance_data, parse_yahoo_data
import argparse
import asyncio

console = Console()


async def messenger(alert_args):
    while True:
        df = parse_yahoo_data(
            get_yahoo_finance_data("QQQ", lookback=3600 * 5, interval="2m")
        )

        close_price = df["Close"].iloc[-1]

        if alert_args["above"]:
            if close_price >= alert_args["above"]:
                await send_to_general(
                    f"ALERT: The price has broken above the {close_price} target!",
                    "robin",
                )
                break
        if alert_args["below"]:
            if close_price <= alert_args["below"]:
                await send_to_general(
                    f"ALERT: The price has broken below the {close_price} target!",
                    "robin",
                )
                break
        await asyncio.sleep(13)


async def main(alert_args):
    bot_task = asyncio.create_task(start_bot())
    while not is_bot_ready():
        await asyncio.sleep(0.5)

    messenger_task = asyncio.create_task(messenger(alert_args))

    try:
        while not messenger_task.done():
            await asyncio.sleep(5)  # Check every 5 seconds
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]⚠️  Shutting down gracefully...[/yellow]")
    finally:
        console.print("[dim]Starting cleanup...[/dim]")
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

        console.print("[bold green]✓ Shutdown complete[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="alerts for price")

    parser.add_argument("-a", "--above", type=float, help="close above price")
    parser.add_argument("-b", "--below", type=float, help="close below price")
    args = parser.parse_args()
    alert_args = {"above": args.above, "below": args.below}
    try:
        asyncio.run(main(alert_args))
    except KeyboardInterrupt:
        pass
