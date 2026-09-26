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
    @staticmethod
    def decode_load(values):
        # STS3215 Present Load is a 10-bit magnitude plus bit 10 direction.
        # rustypot exposes the register integer; normalize it to signed load.
        return [float(-(int(v) & 0x3ff) if int(v) & 0x400 else int(v) & 0x3ff) for v in values]

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
        sample_start = time.monotonic()
        positions = self.bus.sync_read_present_position(self.arm_ids)
        speeds = self.bus.sync_read_present_speed(self.wheel_ids)
        result = {**dict(zip(JOINTS, [float(v) for v in positions])), "wheel_speed": [float(v) for v in speeds]}
        result['sample_monotonic'] = sample_start
        result['servo_goal'] = [float(v) for v in self.bus.sync_read_goal_position(self.arm_ids)]
        try:
            raw_load = [float(v) for v in self.bus.sync_read_present_load(self.arm_ids)]
            result["servo_load_raw"] = raw_load
            result["servo_load"] = self.decode_load(raw_load)
        except Exception:
            pass
        try:
            result["servo_current"] = [float(v) for v in self.bus.sync_read_present_current(self.arm_ids)]
        except Exception:
            pass
        return result

    def diagnostics(self):
        result = {}
        for name in ['cw_dead_zone', 'ccw_dead_zone', 'p_coefficient', 'i_coefficient', 'd_coefficient', 'offset', 'torque_enable', 'torque_limit', 'max_torque_limit', 'min_angle_limit', 'max_angle_limit']:
            try:
                result[name] = list(getattr(self.bus, 'sync_read_' + name)(self.arm_ids))
            except Exception as exc:
                result[name] = str(exc)
        return result

    def joints(self, values):
        targets = [max(JOINT_LIMITS[name][0], min(JOINT_LIMITS[name][1], float(values[name]))) for name in JOINTS]
        self.bus.sync_write_goal_position(self.arm_ids, targets)

    def arm_torque(self, enabled):
        for motor_id in self.arm_ids:
            self.bus.write_torque_enable(motor_id, bool(enabled))

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

    async def broadcast(self, payload):
        """Bound every send so a stale client cannot wedge camera/control service."""
        clients = list(self.clients)
        if not clients:
            return
        results = await asyncio.gather(
            *(asyncio.wait_for(client.send(payload), timeout=0.4) for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                # Dropping the client is not enough: leaving the socket open
                # makes the peer wait forever on a connection that will never
                # carry data again, because protocol pings still answer. Close
                # it so the controller sees the drop and reconnects.
                self.clients.discard(client)
                asyncio.create_task(client.close(code=1011, reason="send timed out"))

    async def reply(self, message):
        await self.broadcast(json.dumps(message))

    async def send_state(self):
        if self.hw:
            self.state_data.update(await asyncio.to_thread(self.hw.state))
        message = json.dumps({"type": "state", "data": self.state_data})
        await self.broadcast(message)

    async def state_loop(self):
        while True:
            try:
                await self.send_state()
            except Exception as exc:
                logging.exception("state read failed")
                self.state_data = {"status": f"error: {exc}", "hardware": True}
            await asyncio.sleep(0.2)

    async def configure_camera(self):
        # UVC exposure controls can reset when the stream is opened. Apply
        # after the first frame, and reapply periodically while streaming.
        settings = (f"gain={self.args.camera_gain},backlight_compensation=0,exposure_dynamic_framerate=0", "auto_exposure=3") if self.args.camera_exposure == 0 else (
            "auto_exposure=1", f"exposure_time_absolute={self.args.camera_exposure},gain={self.args.camera_gain},backlight_compensation=0,exposure_dynamic_framerate=0")
        for setting in settings:
            process = await asyncio.create_subprocess_exec(
                "v4l2-ctl", "-d", self.args.camera_device, "--set-ctrl=" + setting,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                _, error = await asyncio.wait_for(process.communicate(), 2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError("Camera settings command timed out")
            if process.returncode:
                raise RuntimeError(error.decode(errors="replace"))

    async def reapply_camera(self):
        """Refresh UVC controls without ever failing the capture.

        Controls can reset when the stream opens, so they are reapplied
        periodically -- but a settings hiccup must not cost the video feed.
        """
        try:
            await self.configure_camera()
        except Exception:
            logging.warning("camera settings refresh failed; stream continues", exc_info=True)

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
                exposure_applied = 0.
                reconfigure = None
                assert process.stdout is not None
                while chunk := await process.stdout.read(65536):
                    # Never await v4l2-ctl on the read path: while it runs
                    # nothing drains ffmpeg's stdout, so frames stall, and a
                    # timeout here used to tear down the whole capture.
                    if time.monotonic() - exposure_applied > 30 and (
                            reconfigure is None or reconfigure.done()):
                        reconfigure = asyncio.create_task(self.reapply_camera())
                        exposure_applied = time.monotonic()
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
                            await self.broadcast(frame)
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
        if kind == "servo_diagnostics":
            await self.reply({'type': 'servo_diagnostics', 'data': await asyncio.to_thread(self.hw.diagnostics) if self.hw else {}})
        elif kind == "camera_settings":
            exposure = int(data.get("exposure", 30))
            gain = int(data.get("gain", 0))
            if not (exposure == 0 or 1 <= exposure <= 300) or not 0 <= gain <= 60:
                raise ValueError("Camera settings outside bounded range")   # exposure 0 = the camera's auto exposure
            if (exposure, gain) == (self.args.camera_exposure, self.args.camera_gain):
                return
            self.args.camera_exposure = exposure
            self.args.camera_gain = gain
            await self.reapply_camera()
        elif kind == "stop":
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
        elif kind == "arm_torque":
            logging.info("arm_torque command: enabled=%s", bool(data.get("enabled")))
            if self.hw:
                await asyncio.to_thread(self.hw.arm_torque, bool(data.get("enabled")))
            await self.reply({"type": "notice", "message": f"Arm torque {'enabled' if data.get('enabled') else 'disabled'}."})
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
            await websocket.send(json.dumps({"type": "hello", "joints": JOINTS, "state": self.state_data,
                                             "has_home": self.home_pose is not None, "home_pose": self.home_pose}))
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
    parser.add_argument("--camera-exposure", type=int, default=30, help="Initial manual exposure; controlling server adjusts from live image brightness")
    parser.add_argument("--camera-gain", type=int, default=0, help="Initial UVC gain; server controls bounded adjustments")
    parser.add_argument("--camera-size", default="640x360")
    parser.add_argument("--camera-divisor", type=int, default=4, help="Send every Nth native camera frame")
    parser.add_argument("--mock", action="store_true")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(parser.parse_args()))
