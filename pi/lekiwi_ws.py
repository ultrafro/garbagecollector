#!/usr/bin/env python3
"""Conservative WebSocket test bridge for a LeKiwi using rustypot."""
import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import websockets
from rustypot import Sts3215PyController

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MAX_BASE = 0.12
MAX_THETA = 12.0
JOINT_LIMITS = {name: (-3.2, 3.2) for name in JOINTS}
JOINT_LIMITS["gripper"] = (-1.2, 1.2)
HOME_POSE_FILE = Path.home() / "robopet" / "home_pose.json"


class Hardware:
    def __init__(self, device):
        self.bus = Sts3215PyController(serial_port=device, baudrate=1_000_000, timeout=0.5)
        self.arm_ids = [1, 2, 3, 4, 5, 6]
        self.wheel_ids = [7, 8, 9]
        self.bus.sync_read_present_position(self.arm_ids + self.wheel_ids)
        # STS3215 mode changes require torque to be off. Arm servos use
        # position mode (0); LeKiwi wheel servos use constant-speed mode (1).
        for motor_id in self.arm_ids + self.wheel_ids:
            self.bus.write_torque_enable(motor_id, False)
        for motor_id in self.arm_ids:
            self.bus.write_mode(motor_id, 0)
        for motor_id in self.wheel_ids:
            self.bus.write_mode(motor_id, 1)
        for motor_id in self.arm_ids + self.wheel_ids:
            self.bus.write_torque_enable(motor_id, True)

    def state(self):
        positions = self.bus.sync_read_present_position(self.arm_ids)
        speeds = self.bus.sync_read_present_speed(self.wheel_ids)
        return {**dict(zip(JOINTS, [float(v) for v in positions])), "wheel_speed": [float(v) for v in speeds]}

    def joints(self, values):
        targets = [max(JOINT_LIMITS[name][0], min(JOINT_LIMITS[name][1], float(values[name]))) for name in JOINTS]
        self.bus.sync_write_goal_position(self.arm_ids, targets)

    def drive(self, x, y, theta):
        phi = np.deg2rad(np.array([60.0, 180.0, 300.0]))
        body = np.array([x, y, np.deg2rad(theta)])
        matrix = np.column_stack((-np.sin(phi), np.cos(phi), np.full(3, 0.125)))
        speeds = (matrix @ body) / 0.05
        largest = float(np.max(np.abs(speeds)))
        if largest > 10:
            speeds *= 10 / largest
        self.bus.sync_write_goal_speed(self.wheel_ids, speeds.tolist())

    def stop(self):
        self.bus.sync_write_goal_speed(self.wheel_ids, [0.0, 0.0, 0.0])

    def close(self):
        self.stop()
        for motor_id in self.arm_ids + self.wheel_ids:
            self.bus.write_torque_enable(motor_id, False)


class Bridge:
    def __init__(self, args):
        self.args = args
        self.hw = None if args.mock else Hardware(os.environ.get("LEKIWI_PORT", args.device))
        self.clients = set()
        self.lock = asyncio.Lock()
        self.last_command = time.monotonic()
        self.state_data = {"status": "mock" if args.mock else "connected", "hardware": not args.mock}
        if args.mock:
            self.state_data.update(dict.fromkeys(JOINTS, 0.0))
        self.home_pose = self.load_home_pose()

    @staticmethod
    def load_home_pose():
        try:
            data = json.loads(HOME_POSE_FILE.read_text())
            return {name: float(data[name]) for name in JOINTS}
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    async def reply(self, message):
        if self.clients:
            payload = json.dumps(message)
            await asyncio.gather(*(client.send(payload) for client in list(self.clients)), return_exceptions=True)

    async def send_state(self):
        if self.hw:
            self.state_data.update(await asyncio.to_thread(self.hw.state))
        message = json.dumps({"type": "state", "data": self.state_data})
        await asyncio.gather(*(c.send(message) for c in list(self.clients)), return_exceptions=True)

    async def state_loop(self):
        while True:
            try:
                await self.send_state()
            except Exception as exc:
                logging.exception("state read failed")
                self.state_data = {"status": f"error: {exc}", "hardware": True}
            await asyncio.sleep(0.2)

    async def camera_loop(self):
        """Forward native MJPEG frames as binary WebSocket messages."""
        while True:
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "v4l2", "-input_format", "mjpeg",
                    "-video_size", self.args.camera_size,
                    "-i", self.args.camera_device,
                    "-an", "-c:v", "copy", "-f", "image2pipe", "pipe:1",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                buffer = bytearray()
                frame_number = 0
                assert process.stdout is not None
                while chunk := await process.stdout.read(65536):
                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                        if start < 0 or end < 0:
                            if len(buffer) > 2_000_000:
                                buffer.clear()
                            break
                        frame = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]
                        frame_number += 1
                        if frame_number % self.args.camera_divisor == 0 and self.clients:
                            await asyncio.gather(*(client.send(frame) for client in list(self.clients)), return_exceptions=True)
                error = (await process.stderr.read()).decode("utf-8", "replace") if process.stderr else ""
                raise RuntimeError(error.strip() or "camera process exited")
            except asyncio.CancelledError:
                if process and process.returncode is None:
                    process.terminate()
                raise
            except Exception:
                logging.exception("camera stream failed; retrying")
                if process and process.returncode is None:
                    process.terminate()
                await asyncio.sleep(2)

    async def watchdog_loop(self):
        while True:
            if self.hw and time.monotonic() - self.last_command > 0.7:
                await asyncio.to_thread(self.hw.stop)
            await asyncio.sleep(0.15)

    async def command(self, data):
        kind = data.get("type")
        if kind == "stop":
            if self.hw:
                await asyncio.to_thread(self.hw.stop)
            self.last_command = time.monotonic()
        elif kind == "joints" and data.get("enabled"):
            values = data.get("values", {})
            if not all(name in values for name in JOINTS):
                return
            if self.hw:
                await asyncio.to_thread(self.hw.joints, values)
            else:
                self.state_data.update({name: max(JOINT_LIMITS[name][0], min(JOINT_LIMITS[name][1], float(values[name]))) for name in JOINTS})
            self.last_command = time.monotonic()
        elif kind == "drive" and data.get("enabled"):
            x = max(-MAX_BASE, min(MAX_BASE, float(data.get("x", 0))))
            y = max(-MAX_BASE, min(MAX_BASE, float(data.get("y", 0))))
            theta = max(-MAX_THETA, min(MAX_THETA, float(data.get("theta", 0))))
            if self.hw:
                await asyncio.to_thread(self.hw.drive, x, y, theta)
            self.last_command = time.monotonic()
        elif kind == "set_home":
            measured = await asyncio.to_thread(self.hw.state) if self.hw else self.state_data
            self.home_pose = {name: float(measured[name]) for name in JOINTS}
            HOME_POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
            HOME_POSE_FILE.write_text(json.dumps(self.home_pose, indent=2) + "\n")
            await self.reply({"type": "notice", "message": "Home pose saved from current motor positions."})
        elif kind == "home":
            if not self.home_pose:
                await self.reply({"type": "error", "message": "No home pose has been saved yet."})
                return
            if self.hw:
                await asyncio.to_thread(self.hw.joints, self.home_pose)
            else:
                self.state_data.update(self.home_pose)
            await self.reply({"type": "notice", "message": "Moving arm to saved home pose."})

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.send(json.dumps({"type": "hello", "joints": JOINTS, "state": self.state_data, "has_home": self.home_pose is not None}))
            async for raw in websocket:
                try:
                    async with self.lock:
                        await self.command(json.loads(raw))
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            self.clients.discard(websocket)
            if self.hw:
                await asyncio.to_thread(self.hw.stop)


async def main(args):
    bridge = Bridge(args)
    try:
        async with websockets.serve(bridge.handler, args.host, args.port, ping_interval=20):
            await asyncio.gather(bridge.state_loop(), bridge.watchdog_loop(), bridge.camera_loop())
    finally:
        if bridge.hw:
            await asyncio.to_thread(bridge.hw.close)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="/dev/ttyACM0")
    parser.add_argument("--camera-device", default="/dev/video0")
    parser.add_argument("--camera-size", default="640x360")
    parser.add_argument("--camera-divisor", type=int, default=4, help="Send every Nth native camera frame")
    parser.add_argument("--mock", action="store_true")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(parser.parse_args()))
