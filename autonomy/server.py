#!/usr/bin/env python3
"""Run YOLO and trash homing on the controlling computer."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

import cv2
import numpy as np
import websockets
from ultralytics import YOLO

from autonomy.tracking import Detection, closest, track

DEFAULT_LABELS = "discarded packaging,piece of litter,plastic wrapper,snack wrapper,candy wrapper,food packaging,bottle,cup,can,plastic bag,paper cup,food container"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class Autonomy:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = YOLO(args.model)
        self.labels = {x.strip().lower() for x in args.labels.split(",") if x.strip()}
        if "world" in str(args.model).lower():
            self.model.set_classes(sorted(self.labels))
        self.home: dict[str, float] | None = None
        self.latest: tuple[Detection | None, int, int, float] = (None, 0, 0, 0.0)
        self.last_target_seen = time.monotonic()
        self.wrist_delta = 0.0
        self.mode = "starting"

    def infer(self, jpeg: bytes) -> None:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        result = self.model.predict(image, conf=self.args.confidence, verbose=False)[0]
        found = []
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            if label.lower() not in self.labels:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            found.append(Detection(label, float(box.conf.item()), x1, y1, x2, y2))
        h, w = image.shape[:2]
        target = closest(found)
        if target is not None:
            self.last_target_seen = time.monotonic()
        self.latest = (target, w, h, time.monotonic())

    async def receiver(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                await asyncio.to_thread(self.infer, raw)
                continue
            message = json.loads(raw)
            if message.get("type") == "hello":
                if message.get("home_pose"):
                    self.home = {name: float(message["home_pose"][name]) for name in JOINTS}
                else:
                    logging.warning("No saved home pose on Pi; wrist tracking is disabled")
            if message.get("type") == "error":
                logging.error("Pi: %s", message.get("message"))

    async def controller(self, ws) -> None:
        period = 1 / self.args.control_hz
        while True:
            detection, width, height, seen = self.latest
            # A stale frame is not a valid target. Briefly stop before scanning.
            age = time.monotonic() - seen
            searching = age >= self.args.lost_timeout or time.monotonic() - self.last_target_seen >= self.args.lost_timeout
            command = track(
                None if searching else detection, width, height,
                target_height_ratio=self.args.stop_height,
                max_forward=self.args.speed,
                max_theta=self.args.turn_speed,
                min_theta=self.args.min_turn_speed,
                wrist_gain=self.args.wrist_gain,
                patrol_theta=self.args.patrol_speed if searching else 0.0,
            )
            if command.mode != self.mode:
                self.mode = command.mode
                target = detection.label if detection and not searching else "none"
                logging.info("mode=%s target=%s", self.mode, target)
            await ws.send(json.dumps({
                "type": "drive", "enabled": True,
                "x": command.forward * self.args.drive_sign, "y": 0.0,
                "theta": command.theta * self.args.turn_sign,
            }))
            if self.home:
                if detection is None or searching:
                    self.wrist_delta *= 0.9
                self.wrist_delta = max(
                    -self.args.wrist_range,
                    min(self.args.wrist_range, self.wrist_delta + command.wrist_delta),
                )
                joints = dict(self.home)
                joints["wrist_flex"] += self.wrist_delta * self.args.wrist_sign
                await ws.send(json.dumps({"type": "joints", "enabled": True, "values": joints}))
            await asyncio.sleep(period)

    async def run(self) -> None:
        while True:
            try:
                logging.info("connecting to %s", self.args.pi)
                async with websockets.connect(self.args.pi, max_size=4_000_000) as ws:
                    receiver = asyncio.create_task(self.receiver(ws))
                    controller = asyncio.create_task(self.controller(ws))
                    done, pending = await asyncio.wait((receiver, controller), return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                    for task in done:
                        task.result()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("connection failed; retrying")
                await asyncio.sleep(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", default="ws://raspberrypi.local:8765")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated YOLO class names treated as trash")
    parser.add_argument("--confidence", type=float, default=0.004)
    parser.add_argument("--stop-height", type=float, default=0.50, help="Stop when target box occupies this fraction of image height")
    parser.add_argument("--speed", type=float, default=0.11)
    parser.add_argument("--turn-speed", type=float, default=12.0)
    parser.add_argument("--min-turn-speed", type=float, default=6.0)
    parser.add_argument("--patrol-speed", type=float, default=8.0,
                        help="Visible base scan rate in degrees/s; very low values may fall inside wheel deadband")
    parser.add_argument("--lost-timeout", type=float, default=0.7)
    parser.add_argument("--control-hz", type=float, default=12.0)
    parser.add_argument("--wrist-gain", type=float, default=0.025)
    parser.add_argument("--wrist-range", type=float, default=0.8)
    parser.add_argument("--turn-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--drive-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--wrist-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    logging.basicConfig(level=getattr(logging, options.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(Autonomy(options).run())
