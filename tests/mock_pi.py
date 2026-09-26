"""Small browser-test replacement for the Raspberry Pi bridge."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import websockets

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


async def handler(ws):
    state = {"status": "connected", "hardware": True, **dict.fromkeys(JOINTS, 0.0), "wheel_speed": [0, 0, 0]}
    await ws.send(json.dumps({"type": "hello", "joints": JOINTS, "state": state, "has_home": True,
                              "home_pose": dict.fromkeys(JOINTS, 0.0)}))
    jpeg = Path("screenshots/bus.jpg").read_bytes()

    async def updates():
        while True:
            await ws.send(json.dumps({"type": "state", "data": state}))
            await ws.send(jpeg)
            await asyncio.sleep(0.2)

    task = asyncio.create_task(updates())
    try:
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "set_home":
                await ws.send(json.dumps({"type": "notice", "message": "Mock home saved."}))
    finally:
        task.cancel()


async def main():
    async with websockets.serve(handler, "127.0.0.1", 18765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
