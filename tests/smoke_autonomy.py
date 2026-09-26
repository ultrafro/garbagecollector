"""Hardware-free end-to-end smoke test for camera -> YOLO -> motor commands."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets

from autonomy.server import Autonomy, JOINTS


async def main() -> None:
    jpeg = Path("screenshots/bus.jpg").read_bytes()
    received: list[dict] = []
    connected = asyncio.Event()
    send_frames = asyncio.Event()
    send_frames.set()

    async def fake_pi(ws):
        connected.set()
        home = dict.fromkeys(JOINTS, 0.0)
        await ws.send(json.dumps({"type": "hello", "home_pose": home}))

        async def frames():
            while True:
                if send_frames.is_set():
                    await ws.send(jpeg)
                await asyncio.sleep(0.15)

        sender = asyncio.create_task(frames())
        try:
            async for raw in ws:
                received.append(json.loads(raw))
        finally:
            sender.cancel()

    args = argparse.Namespace(
        pi="ws://127.0.0.1:9876", model="yolo11n.pt", labels="bus",
        confidence=0.25, stop_height=0.95, speed=0.06, turn_speed=8.0,
        patrol_speed=3.5, lost_timeout=0.7, control_hz=8.0,
        wrist_gain=0.025, wrist_range=0.8, turn_sign=1.0,
        wrist_sign=1.0, log_level="INFO",
    )
    async with websockets.serve(fake_pi, "127.0.0.1", 9876):
        robot = asyncio.create_task(Autonomy(args).run())
        try:
            await asyncio.wait_for(connected.wait(), 20)
            for _ in range(120):
                if any(m.get("type") == "drive" and m.get("x", 0) > 0 for m in received):
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("YOLO target never produced a forward drive command")

            send_frames.clear()
            await asyncio.sleep(args.lost_timeout + 0.5)
            recent = received[-20:]
            if not any(m.get("type") == "drive" and m.get("x") == 0 and abs(m.get("theta", 0)) == args.patrol_speed for m in recent):
                raise AssertionError("lost-frame patrol command was not emitted")
            if not any(m.get("type") == "joints" for m in received):
                raise AssertionError("home-relative wrist command was not emitted")
            print(json.dumps({"result": "PASS", "commands": len(received), "forward": True,
                              "lost_target_patrol": True, "wrist": True}, indent=2))
        finally:
            robot.cancel()
            await asyncio.gather(robot, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
