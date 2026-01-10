# src/trader.py
import threading
from time import time, sleep
from typing import Optional, Dict, Any
from datetime import datetime
from rich.console import Console
from src.util.feed import get_yahoo_finance_data, parse_yahoo_data

console = Console()


class Position:
    """Represents an open trading position"""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_time: float,
        stop_loss: float,
        signal_type: str,
    ):
        self.symbol = symbol
        self.direction = direction  # "LONG" or "SHORT"
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.signal_type = signal_type
        self.profit_target = self._calculate_profit_target()

    def _calculate_profit_target(self) -> float:
        """Calculate 3:1 profit target based on stop loss distance"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = risk * 3

        if self.direction == "LONG":
            return self.entry_price + reward
        else:
            return self.entry_price - reward

    def time_in_position(self) -> float:
        """Returns time in position in minutes"""
        return (time() - self.entry_time) / 60

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current profit/loss percentage"""
        if self.direction == "LONG":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


class Trader:
    """
    Trader class that receives signals and manages positions with risk management.

    Entry Logic:
    - TREND signals: Immediate entry
    - PRICE_RANGE_TEST signals: Entry when price breaks resistance/support by threshold

    Exit Logic:
    - Stop loss hit (absolute dollar amount)
    - 3:1 profit target reached
    - 50 minutes in position

    Args:
        symbol: Trading symbol (default: "QQQ")
        stop_loss_amount: Stop loss in dollars (default: $2.50)
        price_range_threshold_amount: Breakout threshold in dollars (default: $0.50)
        max_position_time_minutes: Maximum time in position (default: 50 minutes)
        poll_interval_seconds: Price polling frequency (default: 1 second)
    """

    def __init__(
        self,
        symbol: str = "QQQ",
        stop_loss_amount: float = 2.50,
        price_range_threshold_amount: float = 0.50,
        max_position_time_minutes: float = 50,
        poll_interval_seconds: float = 1.0,
        time_offset: int = 0,
    ):
        self.symbol = symbol
        self.stop_loss_amount = stop_loss_amount
        self.price_range_threshold_amount = price_range_threshold_amount
        self.max_position_time_minutes = max_position_time_minutes
        self.poll_interval_seconds = poll_interval_seconds
        self.time_offset = time_offset

        self.position: Optional[Position] = None
        self.current_price: Optional[float] = None
        self.pending_range_signal: Optional[Dict[str, Any]] = None

        # Session statistics
        self.session_trades: list[Dict[str, Any]] = []
        self.session_total_pnl: float = 0.0
        self.winning_trades: int = 0
        self.losing_trades: int = 0

        self._stop_event = threading.Event()
        self._price_monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the trader's price monitoring thread"""
        if (
            self._price_monitor_thread is None
            or not self._price_monitor_thread.is_alive()
        ):
            self._stop_event.clear()
            self._price_monitor_thread = threading.Thread(
                target=self._price_monitor_loop, daemon=True
            )
            self._price_monitor_thread.start()
            console.print(f"[green]Trader started for {self.symbol}[/green]")

    def stop(self):
        """Stop the trader and exit any open positions"""
        self._stop_event.set()
        if self._price_monitor_thread:
            self._price_monitor_thread.join(timeout=5)
        console.print("[yellow]Trader stopped[/yellow]")

    def _price_monitor_loop(self):
        """Continuously monitor price and check exit conditions"""
        while not self._stop_event.is_set():
            try:
                # Fetch current price
                data = parse_yahoo_data(
                    get_yahoo_finance_data(
                        self.symbol,
                        lookback=300,
                        interval="1m",
                        offset=self.time_offset,
                    )
                )

                # Check if we have valid data
                if data is None or len(data) == 0:
                    console.print("[yellow]No data received, retrying...[/yellow]")
                    sleep(self.poll_interval_seconds)
                    continue

                self.current_price = data["Close"].iloc[-1]

                # Check if we have a position and need to exit
                if self.position:
                    self._check_exit_conditions()

                # Check if we have a pending range signal and should enter
                if self.pending_range_signal and not self.position:
                    self._check_range_signal_entry()

            except IndexError as e:
                console.print(f"[yellow]No price data available, retrying...[/yellow]")
            except Exception as e:
                console.print(f"[red]Error in price monitor: {e}[/red]")

            sleep(self.poll_interval_seconds)

    def receive_signal(self, signal: Dict[str, Any]):
        """
        Receive a signal from main.py and decide whether to enter a position

        Args:
            signal: Dictionary containing signal data from check_alerts
        """
        with self._lock:
            if signal.get("trend_signal"):
                self._handle_trend_signal(signal["trend_signal"])
            elif signal.get("range_test_signal"):
                self._handle_range_signal(signal["range_test_signal"])

    def _handle_trend_signal(self, trend_signal: Dict[str, Any]):
        """Handle TREND signal - immediate entry"""
        if self.position:
            console.print("[yellow]Already in position, ignoring trend signal[/yellow]")
            return

        if self.current_price is None or self.current_price <= 0:
            console.print(
                "[red]No valid current price available, cannot enter position[/red]"
            )
            return

        direction = "LONG" if trend_signal["direction"] == "UPTREND" else "SHORT"

        # Calculate stop loss
        if direction == "LONG":
            stop_loss = self.current_price - self.stop_loss_amount
        else:
            stop_loss = self.current_price + self.stop_loss_amount

        # Enter position
        self.position = Position(
            symbol=self.symbol,
            direction=direction,
            entry_price=self.current_price,
            entry_time=time(),
            stop_loss=stop_loss,
            signal_type="TREND",
        )

        console.print(f"\n[bold green]🚀 ENTERED {direction} POSITION[/bold green]")
        console.print(f"Entry Price: ${self.position.entry_price:.2f}")
        console.print(f"Stop Loss: ${self.position.stop_loss:.2f}")
        console.print(f"Profit Target: ${self.position.profit_target:.2f}")
        console.print(
            f"Signal: {trend_signal['direction']} ({trend_signal['strength']})\n"
        )

    def _handle_range_signal(self, range_signal: Dict[str, Any]):
        """Handle PRICE_RANGE_TEST signal - wait for breakout"""
        if self.position:
            console.print("[yellow]Already in position, ignoring range signal[/yellow]")
            return

        # Store the pending signal
        self.pending_range_signal = range_signal

        console.print(f"\n[bold yellow]⏳ MONITORING RANGE BREAKOUT[/bold yellow]")
        console.print(f"Level Type: {range_signal['level_type']}")
        console.print(f"Level Price: ${range_signal['level_price']:.2f}")
        console.print(
            f"Waiting for ${self.price_range_threshold_amount:.2f} breakout threshold\n"
        )

    def _check_range_signal_entry(self):
        """Check if price has broken through resistance/support by threshold"""
        if (
            not self.pending_range_signal
            or self.current_price is None
            or self.current_price <= 0
        ):
            return

        signal = self.pending_range_signal
        level_price = signal["level_price"]
        level_type = signal["level_type"]

        # Use absolute dollar threshold
        threshold_amount = self.price_range_threshold_amount

        entered = False
        direction = None
        stop_loss = None

        if level_type == "RESISTANCE":
            # Enter LONG if price breaks above resistance + threshold
            if self.current_price >= level_price + threshold_amount:
                direction = "LONG"
                stop_loss = self.current_price - self.stop_loss_amount
                entered = True

        elif level_type == "SUPPORT":
            # Enter SHORT if price breaks below support - threshold
            if self.current_price <= level_price - threshold_amount:
                direction = "SHORT"
                stop_loss = self.current_price + self.stop_loss_amount
                entered = True

        if entered and direction and stop_loss:
            self.position = Position(
                symbol=self.symbol,
                direction=direction,
                entry_price=self.current_price,
                entry_time=time(),
                stop_loss=stop_loss,
                signal_type="PRICE_RANGE_TEST",
            )

            console.print(
                f"\n[bold green]🚀 ENTERED {direction} POSITION (BREAKOUT)[/bold green]"
            )
            console.print(f"Entry Price: ${self.position.entry_price:.2f}")
            console.print(f"Stop Loss: ${self.position.stop_loss:.2f}")
            console.print(f"Profit Target: ${self.position.profit_target:.2f}")
            console.print(f"Signal: {level_type} breakout at ${level_price:.2f}\n")

            # Clear pending signal
            self.pending_range_signal = None

    def _check_exit_conditions(self):
        """Check all exit conditions and close position if any are met"""
        if not self.position or self.current_price is None:
            return

        position = self.position
        current_pnl = position.calculate_pnl(self.current_price)
        time_in_position = position.time_in_position()

        exit_reason = None

        # Check stop loss
        if position.direction == "LONG":
            if self.current_price <= position.stop_loss:
                exit_reason = "STOP_LOSS"
        else:  # SHORT
            if self.current_price >= position.stop_loss:
                exit_reason = "STOP_LOSS"

        # Check profit target (3:1)
        if not exit_reason:
            if position.direction == "LONG":
                if self.current_price >= position.profit_target:
                    exit_reason = "PROFIT_TARGET"
            else:  # SHORT
                if self.current_price <= position.profit_target:
                    exit_reason = "PROFIT_TARGET"

        # Check time limit (50 minutes)
        if not exit_reason:
            if time_in_position >= self.max_position_time_minutes:
                exit_reason = "TIME_LIMIT"

        if exit_reason:
            self._exit_position(exit_reason, current_pnl)

    def _exit_position(self, reason: str, pnl: float):
        """Exit the current position"""
        if not self.position:
            return

        position = self.position
        exit_color = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""

        # Update session statistics
        self.session_total_pnl += pnl
        if pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Record trade details
        trade_record = {
            "direction": position.direction,
            "signal_type": position.signal_type,
            "entry_price": position.entry_price,
            "exit_price": self.current_price,
            "pnl": pnl,
            "exit_reason": reason,
            "time_in_position": position.time_in_position(),
            "timestamp": datetime.now(),
        }
        self.session_trades.append(trade_record)

        # Display exit information
        console.print(
            f"\n[bold {exit_color}]📤 EXITED {position.direction} POSITION[/bold {exit_color}]"
        )
        console.print(f"Exit Reason: {reason}")
        console.print(f"Entry Price: ${position.entry_price:.2f}")
        console.print(f"Exit Price: ${self.current_price:.2f}")
        console.print(f"PnL: [{exit_color}]{pnl_sign}{pnl:.2f}%[/{exit_color}]")
        console.print(f"Time in Position: {position.time_in_position():.1f} minutes")

        # Display session statistics
        console.print(f"\n[bold cyan]📊 SESSION STATISTICS[/bold cyan]")
        console.print(f"Total Trades: {len(self.session_trades)}")
        console.print(f"Winning Trades: [green]{self.winning_trades}[/green]")
        console.print(f"Losing Trades: [red]{self.losing_trades}[/red]")

        win_rate = (
            (self.winning_trades / len(self.session_trades) * 100)
            if self.session_trades
            else 0
        )
        console.print(f"Win Rate: {win_rate:.1f}%")

        session_color = "green" if self.session_total_pnl >= 0 else "red"
        session_sign = "+" if self.session_total_pnl >= 0 else ""
        console.print(
            f"Session Total P&L: [{session_color}]{session_sign}{self.session_total_pnl:.2f}%[/{session_color}]\n"
        )

        # Clear position
        self.position = None

    def get_status(self) -> Dict[str, Any]:
        """Get current trader status"""
        status = {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "has_position": self.position is not None,
            "has_pending_signal": self.pending_range_signal is not None,
            "session_stats": {
                "total_trades": len(self.session_trades),
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": (self.winning_trades / len(self.session_trades) * 100)
                if self.session_trades
                else 0,
                "session_total_pnl": self.session_total_pnl,
            },
        }

        if self.position:
            status["position"] = {
                "direction": self.position.direction,
                "entry_price": self.position.entry_price,
                "stop_loss": self.position.stop_loss,
                "profit_target": self.position.profit_target,
                "time_in_position": self.position.time_in_position(),
                "current_pnl": self.position.calculate_pnl(self.current_price)
                if self.current_price
                else 0,
            }

        if self.pending_range_signal:
            status["pending_signal"] = {
                "level_type": self.pending_range_signal["level_type"],
                "level_price": self.pending_range_signal["level_price"],
            }

        return status

    def get_session_summary(self) -> str:
        """Get a formatted summary of the trading session"""
        if not self.session_trades:
            return "No trades executed this session."

        summary = "\n" + "=" * 50 + "\n"
        summary += "📊 TRADING SESSION SUMMARY\n"
        summary += "=" * 50 + "\n"
        summary += f"Total Trades: {len(self.session_trades)}\n"
        summary += f"Winning Trades: {self.winning_trades}\n"
        summary += f"Losing Trades: {self.losing_trades}\n"

        win_rate = self.winning_trades / len(self.session_trades) * 100
        summary += f"Win Rate: {win_rate:.1f}%\n"

        summary += f"Session Total P&L: {self.session_total_pnl:+.2f}%\n"
        summary += "\nTrade History:\n"
        summary += "-" * 50 + "\n"

        for i, trade in enumerate(self.session_trades, 1):
            summary += f"{i}. {trade['direction']} {trade['signal_type']}\n"
            summary += f"   Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f}\n"
            summary += (
                f"   P&L: {trade['pnl']:+.2f}% | Reason: {trade['exit_reason']}\n"
            )
            summary += f"   Time: {trade['time_in_position']:.1f} min\n"

        summary += "=" * 50 + "\n"

        return summary
