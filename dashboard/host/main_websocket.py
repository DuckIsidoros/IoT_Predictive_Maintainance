from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path

clients: List[WebSocket] = []

DATA_DIR = Path(__file__).resolve().parents[1] / "reactdashboard" / "public" / "dashboard-data"


async def broadcast(payload: object) -> None:
    remove: list[WebSocket] = []
    for ws in list(clients):
        try:
            await ws.send_json(payload)
        except Exception:
            remove.append(ws)
    for ws in remove:
        try:
            clients.remove(ws)
        except ValueError:
            pass


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


def build_payload(snapshot: dict) -> dict:
    features = snapshot.get("features") or {}
    summary = snapshot.get("summary") or {}

    return {
        "timestamp": summary.get("timestamp", ""),
        "raw": summary.get("raw", {"ax": [], "ay": [], "az": []}),
        "features": {
            "rms": features.get("rms", {}),
            "band_power": features.get("band_power", {}),
            "band_level": features.get("band_level", "unknown"),
        },
        "fft": summary.get("fft", {"frequencies": [], "magnitudes": []}),
        "inference": summary.get("inference", {}),
        "status": summary.get("status", {}),
    }


async def poll_and_broadcast():
    last_payload = None
    while True:
        snapshot = await _read_snapshot_files()
        payload = build_payload(snapshot)

        if payload != last_payload:
            await broadcast({"type": "snapshot", "data": payload})
            last_payload = payload

        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(poll_and_broadcast())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


async def send_snapshot_to(websocket: WebSocket) -> None:
    snapshot = await _read_snapshot_files()
    payload = build_payload(snapshot)
    envelope = {"type": "snapshot", "data": payload}
    await websocket.send_json(envelope)


@app.websocket("/ws/dashboard")
async def dashboard_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        await send_snapshot_to(websocket)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        try:
            clients.remove(websocket)
        except ValueError:
            pass
    except Exception as e:
        print(f"dashboard ws error: {e}")
        try:
            clients.remove(websocket)
        except ValueError:
            pass