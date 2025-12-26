"""
WebSocket utility for sending alerts to the dashboard.
Add this to your project and import it in main.py
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp


class DashboardWebSocket:
    """Simple WebSocket client to send messages to the dashboard"""

    def __init__(self, dashboard_url: str = "http://localhost:8000"):
        self.dashboard_url = dashboard_url
        self.api_url = f"{dashboard_url}/api/broadcast"

    async def send_alert(self, alert_type: str, alert_data: Dict[Any, Any]):
        """
        Send an alert to the dashboard

        Args:
            alert_type: Type of alert ('reversion', 'trend', 'range', 'volume')
            alert_data: Alert data dictionary
        """
        try:
            message = {
                "type": "alert_update",
                "timestamp": datetime.now().isoformat(),
                "alert_type": alert_type,
                "data": alert_data,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=message) as resp:
                    if resp.status != 200:
                        print(f"Failed to send alert: {resp.status}")
        except Exception as e:
            print(f"Error sending alert to dashboard: {e}")

    async def send_chart_update(self, image_path: str):
        """
        Send chart update to dashboard

        Args:
            image_path: Path to chart image
        """
        import base64
        from pathlib import Path

        try:
            chart_file = Path(image_path)
            if not chart_file.exists():
                return

            with open(chart_file, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode()

            message = {
                "type": "chart_update",
                "image": f"data:image/jpeg;base64,{image_base64}",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=message) as resp:
                    if resp.status != 200:
                        print(f"Failed to send chart: {resp.status}")
        except Exception as e:
            print(f"Error sending chart to dashboard: {e}")

    def send_alert_sync(self, alert_type: str, alert_data: Dict[Any, Any]):
        """Synchronous wrapper for send_alert"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, create a task
                asyncio.create_task(self.send_alert(alert_type, alert_data))
            else:
                # If no loop is running, run it
                loop.run_until_complete(self.send_alert(alert_type, alert_data))
        except Exception as e:
            print(f"Error in sync send: {e}")

    def send_chart_update_sync(self, image_path: str):
        """Synchronous wrapper for send_chart_update"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_chart_update(image_path))
            else:
                loop.run_until_complete(self.send_chart_update(image_path))
        except Exception as e:
            print(f"Error in sync chart send: {e}")


# Global instance
dashboard_ws = DashboardWebSocket()


def send_to_dashboard(alert_type: str, alert_data: Optional[Dict[Any, Any]] = None):
    """
    Simple function to send alerts to dashboard.
    Call this from your display_alerts() function.

    Args:
        alert_type: 'reversion', 'trend', 'range', 'volume', or 'chart'
        alert_data: Alert data dictionary (not needed for chart updates)

    Example:
        send_to_dashboard('reversion', reversion_signal)
        send_to_dashboard('chart')  # Send chart update
    """
    if alert_type == "chart":
        dashboard_ws.send_chart_update_sync("charts/tminus60.jpeg")
    elif alert_data:
        dashboard_ws.send_alert_sync(alert_type, alert_data)
