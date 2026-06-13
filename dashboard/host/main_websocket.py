from fastapi import FastAPI, WebSocket
from typing import List

app = FastAPI()


clients: List[WebSocket] = []

@app.websocket("/ws/dashboard")
async def dashboard_endpoint(websocket: WebSocket):
    await websocket.accept()    
    clients.append(websocket)       
    
    try:
        while True:
            await websocket.receive_text()  
    except:
        clients.remove(websocket)