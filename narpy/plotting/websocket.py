import websockets
import asyncio
from queue import Queue
import json
import time

class GraphWebsocket:
    def __init__(self, queue: Queue, host:str = 'localhost', port: int = 7235) -> None:
        self.port = port
        self.host = host
        self.x = []
        self.observations = []
        self.rewards = []
        self.queue = queue

    async def handler(self, websocket, path):
        while True:
            try:
                data = self.queue.get()
                if data is not None:
                    self.x.append(len(self.x) + 1)
                    self.observations.append(data[0])
                    self.rewards.append(data[1])
                    await websocket.send(json.dumps({
                        "x": self.x,
                        "observations": self.observations,
                        "rewards": self.rewards
                    }))
                    await asyncio.sleep(0)
                else:
                    break
            except websockets.exceptions.ConnectionClosed:
                break

    async def start(self):
        self.server = await websockets.serve(self.handler, self.host, self.port)
        print(f"Started ws server at {self.host}:{self.port}")
        await self.server.wait_closed()
        print("Closed ws server")
    
    def stop(self):
        self.queue.put(None)
        self.server.close()

        
    def run(self):
        asyncio.run(self.start())