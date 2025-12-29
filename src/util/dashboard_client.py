"""
WebSocket client for sending alerts to the dashboard
Improved version with better error handling and async support
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp
import requests
from pathlib import Path
import base64
import numpy as np


class DashboardClient:
    """HTTP client to send messages to the dashboard"""

    def __init__(self, dashboard_url: str = "http://localhost:8179"):
        self.dashboard_url = dashboard_url
        self.broadcast_url = f"{dashboard_url}/api/broadcast"
        self.health_url = f"{dashboard_url}/api/health"

    def _serialize_data(self, data: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Convert datetime objects to ISO format strings for JSON serialization

        Args:
            data: Dictionary that may contain datetime objects

        Returns:
            Dictionary with datetime objects converted to strings
        """
        serialized = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (np.bool_, np.bool)):  # Handle numpy booleans
                serialized[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):  # Handle numpy numbers
                serialized[key] = value.item()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_data(value)
            elif isinstance(value, list):
                serialized[key] = [
                    item.isoformat()
                    if isinstance(item, datetime)
                    else bool(item)
                    if isinstance(item, (np.bool_, np.bool))
                    else item.item()
                    if isinstance(item, (np.integer, np.floating))
                    else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    async def send_alert(self, alert_type: str, alert_data: Dict[Any, Any]):
        """
        Send an alert to the dashboard (async version)

        Args:
            alert_type: Type of alert ('reversion', 'trend', 'range', 'volume')
            alert_data: Alert data dictionary with timestamp
        """
        try:
            # Convert datetime objects to ISO format strings
            alert_data = self._serialize_data(alert_data)

            # Ensure timestamp is present
            if "timestamp" not in alert_data:
                alert_data["timestamp"] = datetime.now().isoformat()

            message = {
                "type": "alert_update",
                "alert_type": alert_type,
                "data": alert_data,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.broadcast_url,
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        connections = result.get("connections", 0)
                        print(f"✓ Alert sent: {alert_type} → {connections} client(s)")
                    else:
                        print(f"✗ Failed to send alert: HTTP {resp.status}")

        except asyncio.TimeoutError:
            print(f"✗ Timeout sending alert to dashboard")
        except aiohttp.ClientError as e:
            print(f"✗ Connection error: {e}")
        except Exception as e:
            print(f"✗ Error sending alert: {e}")

    async def send_chart_update(self, image_path: str):
        """
        Send chart update to dashboard (async version)

        Args:
            image_path: Path to chart image file
        """
        try:
            chart_file = Path(image_path)
            if not chart_file.exists():
                print(f"✗ Chart file not found: {image_path}")
                return

            # Read and encode image
            with open(chart_file, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode()

            message = {
                "type": "chart_update",
                "image": f"data:image/jpeg;base64,{image_base64}",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.broadcast_url,
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        connections = result.get("connections", 0)
                        print(f"✓ Chart updated → {connections} client(s)")
                    else:
                        print(f"✗ Failed to send chart: HTTP {resp.status}")

        except asyncio.TimeoutError:
            print(f"✗ Timeout sending chart to dashboard")
        except aiohttp.ClientError as e:
            print(f"✗ Connection error: {e}")
        except Exception as e:
            print(f"✗ Error sending chart: {e}")

    def send_alert_sync(self, alert_type: str, alert_data: Dict[Any, Any]):
        """
        Send an alert to the dashboard (synchronous version)
        Use this in non-async code

        Args:
            alert_type: Type of alert ('reversion', 'trend', 'range', 'volume')
            alert_data: Alert data dictionary
        """
        try:
            # Convert datetime objects to ISO format strings
            alert_data = self._serialize_data(alert_data)

            # Ensure timestamp is present
            if "timestamp" not in alert_data:
                alert_data["timestamp"] = datetime.now().isoformat()

            message = {
                "type": "alert_update",
                "alert_type": alert_type,
                "data": alert_data,
            }

            response = requests.post(self.broadcast_url, json=message, timeout=5)

            if response.status_code == 200:
                result = response.json()
                connections = result.get("connections", 0)
                print(f"✓ Alert sent: {alert_type} → {connections} client(s)")
            else:
                print(f"✗ Failed to send alert: HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"✗ Timeout sending alert to dashboard")
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to dashboard at {self.dashboard_url}")
            print(f"  Make sure the server is running: python app_websocket.py")
        except Exception as e:
            print(f"✗ Error sending alert: {e}, {alert_data}")

    def send_chart_update_sync(self, image_path: str):
        """
        Send chart update to dashboard (synchronous version)
        Use this in non-async code

        Args:
            image_path: Path to chart image file
        """
        try:
            chart_file = Path(image_path)
            if not chart_file.exists():
                print(f"✗ Chart file not found: {image_path}")
                return

            # Read and encode image
            with open(chart_file, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode()

            message = {
                "type": "chart_update",
                "image": f"data:image/jpeg;base64,{image_base64}",
            }

            response = requests.post(self.broadcast_url, json=message, timeout=5)

            if response.status_code == 200:
                result = response.json()
                connections = result.get("connections", 0)
                print(f"✓ Chart updated → {connections} client(s)")
            else:
                print(f"✗ Failed to send chart: HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"✗ Timeout sending chart to dashboard")
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to dashboard at {self.dashboard_url}")
            print(f"  Make sure the server is running: python app_websocket.py")
        except Exception as e:
            print(f"✗ Error sending chart: {e}")

    def check_health_sync(self) -> bool:
        """
        Check if the dashboard server is running and healthy

        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = requests.get(self.health_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                connections = data.get("websocket_connections", 0)
                print(f"✓ Dashboard is healthy ({connections} WebSocket connection(s))")
                return True
            else:
                print(f"✗ Dashboard returned HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to dashboard at {self.dashboard_url}")
            return False
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False


# Global instance
dashboard_cli = DashboardClient()


def send_to_dashboard(alert_type: str, alert_data: Optional[Dict[Any, Any]] = None):
    """
    Simple function to send alerts to dashboard via WebSocket broadcast.

    Args:
        alert_type: 'reversion', 'trend', 'range', 'volume', or 'chart'
        alert_data: Alert data dictionary (not needed for chart updates)

    Examples:
        # Send mean reversion signal
        send_to_dashboard('reversion', {
            "timestamp": "2024-01-15T10:30:00",
            "direction": "LONG",
            "current_price": 450.25,
            "deviation": 2.5,
            "volume_ratio": 1.8
        })

        # Send trend signal
        send_to_dashboard('trend', {
            "timestamp": "2024-01-15T10:30:00",
            "direction": "BULLISH",
            "strength": "STRONG",
            "ratio": 0.75,
            "momentum": "INCREASING"
        })

        # Send range test signal
        send_to_dashboard('range', {
            "timestamp": "2024-01-15T10:30:00",
            "level_type": "SUPPORT",
            "level_price": 445.50,
            "num_tests": 3,
            "urgency": "HIGH"
        })

        # Send volume anomaly signal
        send_to_dashboard('volume', {
            "timestamp": "2024-01-15T10:30:00",
            "volume_direction": "UP",
            "interpretation": "Accumulation detected",
            "volume_ratio": 2.5,
            "confidence": "HIGH"
        })

        # Send chart update
        send_to_dashboard('chart')
    """
    # Always update charts
    # to visualize the context
    # of most recent alert
    dashboard_cli.send_chart_update_sync("charts/tminus60.jpeg")

    if alert_data:
        dashboard_cli.send_alert_sync(alert_type, alert_data)
