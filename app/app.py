"""
FastAPI Web Application for Market Alerts Dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import asyncio
import json
from datetime import datetime
from typing import List
import base64


app = FastAPI(title="Market Alerts Dashboard")

# Create templates directory
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="app/templates")


# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/chart")
async def get_chart():
    """Get the chart image as base64"""
    chart_path = Path("charts/tminus60.jpeg")

    if not chart_path.exists():
        return {"error": "Chart not available"}

    with open(chart_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode()

    return {"image": f"data:image/jpeg;base64,{image_base64}"}


@app.post("/api/broadcast")
async def broadcast_message(message: dict):
    """
    Receive messages from main.py and broadcast to all WebSocket clients
    """
    await manager.broadcast(message)
    return {"status": "success"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    await manager.connect(websocket)

    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)

            # Send a ping to keep connection alive
            await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8179)
