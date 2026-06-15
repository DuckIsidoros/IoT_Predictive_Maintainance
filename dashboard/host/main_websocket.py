import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from contextlib import asynccontextmanager
import asyncio
import json
import aiomqtt
import aiosqlite

clients: List[WebSocket] = []
latest_data = {}
async def broadcast(payload: object) -> None:
    for ws in list(clients):
        try:
            await ws.send_json(payload)
        except Exception:
            if ws in clients: clients.remove(ws)

async def mqtt_listener():
    global latest_data
    await init_db()
    while True:
        try:
            async with aiomqtt.Client("localhost") as client:
                await client.subscribe("vibration/inference")
                async for message in client.messages:
                    try:
                        data = json.loads(message.payload.decode())
                        latest_data = data

                        async with aiosqlite.connect("vibration_data.db") as db:
                            await db.execute(
                                "INSERT INTO vibration_logs (timestamp, prediction, confidence, fan_state) VALUES (?, ?, ?, ?)",
                                (data.get("timestamp"),
                                data["inference"]["prediction"],
                                data["inference"]["confidence"],
                                data["status"]["fan_state"])
                            )
                            await db.commit()

                        await broadcast({"type": "snapshot", "data": latest_data})
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Bad payload, skipping: {e}")
        except Exception as e:
            print(f"MQTT connection error: {e}, retrying in 3s")
            await asyncio.sleep(3)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(mqtt_listener())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/dashboard")
async def dashboard_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        # Send initial data if available
        if latest_data:
            await websocket.send_json({"type": "snapshot", "data": latest_data})
        
        # Keep connection open
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.remove(websocket)
        
async def init_db():
    async with aiosqlite.connect("vibration_data.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS vibration_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prediction TEXT,
                confidence REAL,
                fan_state TEXT
            )
        """)
        await db.commit()

async def mqtt_listener():
    global latest_data
    # Ensure table exists
    await init_db()
    
    async with aiomqtt.Client("localhost") as client:
        await client.subscribe("vibration/inference")
        async for message in client.messages:
            payload_str = message.payload.decode()
            data = json.loads(payload_str)
            latest_data = data
            
            # --- ADD THIS TO SAVE TO SQLITE ---
            async with aiosqlite.connect("vibration_data.db") as db:
                await db.execute(
                    "INSERT INTO vibration_logs (timestamp, prediction, confidence, fan_state) VALUES (?, ?, ?, ?)",
                    (data.get("timestamp"), 
                    data["inference"]["prediction"], 
                    data["inference"]["confidence"], 
                    data["status"]["fan_state"])
                )
                await db.commit()
            # ----------------------------------
            
            await broadcast({"type": "snapshot", "data": latest_data})
            
@app.get("/api/history")
async def get_history():
    async with aiosqlite.connect("vibration_data.db") as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM vibration_logs ORDER BY id DESC LIMIT 50") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]            
            
'''
DEPRECATED: We are now using MQTT to get real-time updates, so we no longer need to read from snapshot files

async def _read_snapshot_files() -> dict:
    def read_files() -> dict:
        out: dict = {}
        for name in ("summary.json", "features.json", "feature_extraction_report.json"):
            p = DATA_DIR / name
            key = name.replace(".json", "")
            if p.exists():
                try:
                    out[key] = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    out[key] = None
            else:
                out[key] = None
        return out
    return await asyncio.to_thread(read_files)

''' 

'''
DEPRECATED: We are now using MQTT to get real-time updates, so we no longer need to poll snapshot files

async def poll_and_broadcast():
    last_payload = None
    while True:
        snapshot = await _read_snapshot_files()
        payload = build_payload(snapshot)
        if payload != last_payload:
            await broadcast({"type": "snapshot", "data": payload})
            last_payload = payload

        await asyncio.sleep(1)
''' 