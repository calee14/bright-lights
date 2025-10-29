from concurrent.futures import ThreadPoolExecutor
from threading import Event
from rich.console import Console
from src.util.bot import is_bot_ready, send_to_general, start_bot
from src.util.feed import get_yahoo_finance_data, parse_yahoo_data
import argparse
import asyncio

console = Console()


async def messenger(alert_price, above):
    while True:
        df = parse_yahoo_data(
            get_yahoo_finance_data("QQQ", lookback=3600 * 5, interval="2m")
        )

        close_price = df["Close"].iloc[-1]

        if above:
            if close_price >= alert_price:
                await send_to_general(
                    f"ALERT: The price has broken above the {close_price} target!",
                    "robin",
                )
                break
        else:
            if close_price <= alert_price:
                await send_to_general(
                    f"ALERT: The price has broken below the {close_price} target!",
                    "robin",
                )
                break
        await asyncio.sleep(39)


async def main(alert_price, above):
    bot_task = asyncio.create_task(start_bot())
    while not is_bot_ready():
        await asyncio.sleep(0.5)

    messenger_task = asyncio.create_task(messenger(alert_price, above))

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
    if args.above:
        alert_price = args.above
        above = True
    elif args.below:
        alert_price = args.below
        above = False
    try:
        asyncio.run(main(alert_price, above))
    except KeyboardInterrupt:
        pass
