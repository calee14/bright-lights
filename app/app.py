"""
FastAPI Web Application for Market Alerts Dashboard
WebSocket-based real-time architecture with improved connection handling
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import json
import base64
from pydantic import BaseModel

app = FastAPI(title="Market Alerts Dashboard")


# WebSocket Connection Manager with improved reliability
class ConnectionManager:
    """Manages WebSocket connections with heartbeat and error recovery"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_ids: Dict[WebSocket, str] = {}
        self._next_id = 0

    async def connect(self, websocket: WebSocket) -> str:
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        connection_id = f"conn_{self._next_id}"
        self._next_id += 1

        self.active_connections.append(websocket)
        self.connection_ids[websocket] = connection_id

        print(
            f"✓ Client connected: {connection_id} (Total: {len(self.active_connections)})"
        )
        return connection_id

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            connection_id = self.connection_ids.get(websocket, "unknown")
            self.active_connections.remove(websocket)
            if websocket in self.connection_ids:
                del self.connection_ids[websocket]
            print(
                f"✗ Client disconnected: {connection_id} (Total: {len(self.active_connections)})"
            )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending to client: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients with error handling"""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


# Store latest data for new connections
class DataStore:
    """Store latest alerts and chart for new connections"""

    def __init__(self):
        self.latest_chart = None
        self.latest_alerts = {
            "reversion": None,
            "trend": None,
            "range": None,
            "volume": None,
        }
        self.last_update = None
        self.alert_count = 0

    def update_chart(self, chart_data: str):
        """Store latest chart"""
        self.latest_chart = chart_data
        self.last_update = datetime.now().isoformat()
        print(f"  ✓ Chart stored (size: {len(chart_data)} bytes)")

    def update_alert(self, alert_type: str, alert_data: Dict[Any, Any]):
        """Store latest alert"""
        if alert_type not in self.latest_alerts:
            print(f"  ✗ Invalid alert type: {alert_type}")
            raise ValueError(f"Invalid alert type: {alert_type}")

        self.latest_alerts[alert_type] = alert_data
        self.last_update = datetime.now().isoformat()
        self.alert_count += 1

        print(f"  ✓ Alert stored: {alert_type}")
        print(f"    - Timestamp: {alert_data.get('timestamp', 'N/A')}")
        print(f"    - Total alerts stored: {self.alert_count}")

    def get_initial_state(self) -> dict:
        """Get all current data for new connections"""
        active_count = sum(
            1 for alert in self.latest_alerts.values() if alert is not None
        )
        print(
            f"  → Sending initial state: {active_count} active alerts, chart: {self.latest_chart is not None}"
        )

        return {
            "chart": self.latest_chart,
            "alerts": self.latest_alerts,
            "last_update": self.last_update,
        }

    def get_stats(self) -> dict:
        """Get statistics about stored data"""
        return {
            "total_alerts_received": self.alert_count,
            "active_alerts": {
                alert_type: alert is not None
                for alert_type, alert in self.latest_alerts.items()
            },
            "has_chart": self.latest_chart is not None,
            "last_update": self.last_update,
        }


# Global data store
data_store = DataStore()


# Pydantic models
class BroadcastMessage(BaseModel):
    type: str
    data: Optional[Dict[Any, Any]] = None
    alert_type: Optional[str] = None
    image: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def home():
    """Render the main dashboard page"""
    html_path = Path("app/templates/dashboard.html")
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Dashboard HTML not found</h1><p>Please create templates/dashboard.html</p>",
            status_code=404,
        )

    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with heartbeat and error recovery"""
    connection_id = await manager.connect(websocket)

    try:
        # Send initial state to newly connected client
        initial_state = data_store.get_initial_state()

        # Send chart if available
        if initial_state["chart"]:
            await manager.send_personal_message(
                {"type": "chart_update", "image": initial_state["chart"]}, websocket
            )

        # Send each alert if available
        for alert_type, alert_data in initial_state["alerts"].items():
            if alert_data:
                await manager.send_personal_message(
                    {
                        "type": "alert_update",
                        "alert_type": alert_type,
                        "data": alert_data,
                    },
                    websocket,
                )

        # Send connection success message
        await manager.send_personal_message(
            {
                "type": "connected",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # 30 second timeout
                )

                # Handle ping/pong
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        websocket,
                    )

            except asyncio.TimeoutError:
                # Send heartbeat ping
                await manager.send_personal_message(
                    {"type": "ping", "timestamp": datetime.now().isoformat()}, websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error for {connection_id}: {e}")
        manager.disconnect(websocket)


@app.post("/api/broadcast")
async def broadcast_message(message: BroadcastMessage):
    """
    Receive messages from trading system and broadcast to all WebSocket clients
    Also stores data for new connections
    """
    try:
        broadcast_data = {"type": message.type}

        if message.type == "chart_update" and message.image:
            # Store the chart
            data_store.update_chart(message.image)
            broadcast_data["image"] = message.image
            print(f"📸 Chart update received and stored")

        elif message.type == "alert_update" and message.alert_type and message.data:
            # Store the alert
            data_store.update_alert(message.alert_type, message.data)
            broadcast_data["alert_type"] = message.alert_type
            broadcast_data["data"] = message.data
            print(
                f"🚨 Alert received: {message.alert_type} - {message.data.get('direction', 'N/A')}"
            )
        else:
            print(f"⚠️  Unknown message type or missing data: {message.type}")
            return {"status": "error", "message": "Invalid message format"}

        # Broadcast to all connected WebSocket clients
        await manager.broadcast(broadcast_data)

        return {
            "status": "success",
            "connections": manager.get_connection_count(),
            "timestamp": datetime.now().isoformat(),
            "message_type": message.type,
        }

    except Exception as e:
        print(f"❌ Error broadcasting: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/chart")
async def get_chart():
    """Get the latest chart image"""
    # Try to get from data store first
    if data_store.latest_chart:
        return {"image": data_store.latest_chart}

    # Fall back to file system
    chart_path = Path("charts/tminus60.jpeg")
    if not chart_path.exists():
        return {"error": "Chart not available"}

    try:
        with open(chart_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode()

        chart_data = f"data:image/jpeg;base64,{image_base64}"
        data_store.update_chart(chart_data)
        return {"image": chart_data}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/alerts")
async def get_alerts():
    """Get all current alerts (for debugging or initial load)"""
    return {
        "reversion_signal": data_store.latest_alerts["reversion"],
        "trend_signal": data_store.latest_alerts["trend"],
        "range_test_signal": data_store.latest_alerts["range"],
        "volume_signal": data_store.latest_alerts["volume"],
        "last_update": data_store.last_update,
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "websocket_connections": manager.get_connection_count(),
        "has_chart": data_store.latest_chart is not None,
        "active_alerts": sum(
            1 for alert in data_store.latest_alerts.values() if alert is not None
        ),
    }


@app.get("/api/connections")
async def get_connections():
    """Get WebSocket connection status"""
    return {
        "active_connections": manager.get_connection_count(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/stats")
async def get_stats():
    """Get detailed statistics about stored data"""
    return {
        **data_store.get_stats(),
        "websocket_connections": manager.get_connection_count(),
        "timestamp": datetime.now().isoformat(),
    }


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    # Create directories
    Path("charts").mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("Market Alerts Dashboard - WebSocket Server")
    print("=" * 70)
    print("\n📊 Dashboard: http://localhost:8179")
    print("🔌 WebSocket: ws://localhost:8179/ws")
    print("❤️  Health: http://localhost:8179/api/health")
    print("📡 Connections: http://localhost:8179/api/connections")
    print("\n" + "=" * 70)
    print("\nFeatures:")
    print("  ✓ Real-time WebSocket updates")
    print("  ✓ Automatic reconnection")
    print("  ✓ Heartbeat ping/pong")
    print("  ✓ Initial state sync for new connections")
    print("  ✓ Graceful error handling")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8179, log_level="info")

