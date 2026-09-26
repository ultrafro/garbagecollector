#!/usr/bin/env python3
"""Web control hub: continuous YOLO (or hybrid VLM targeting), manual drive, and autonomous trash homing."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import websockets
from aiohttp import WSMsgType, web
from ultralytics import YOLO

from autonomy.contact import DEFAULT_THRESHOLD, LoadContactDetector
from autonomy.server import DEFAULT_LABELS, JOINTS
from autonomy.tracking import Detection, associated, closest, track
from autonomy.wrist_ik import WristIK, clamp_direction, rotation, rotation_between


class ControlHub:
    targeter = None                  # HybridTargeter when --targeter vlm
    vlm = None                       # its latest snapshot
    vlm_client = None                # LlamaServerLocator when --targeter vlm, used for grasp checks
    auto_complete = False            # set when an auto approach finishes on its own
    auto_progress_t = 0.             # last time auto moved the base or wrist (stall watchdog)
    search_turn_deg = 0.             # rotation so far in the current search without finding a target
    search_tick = None
    logged_reason = None
    auto_cmd = None                  # last arm pose auto commanded (rate-limited toward the target pose)
    grip_reading = (-1.2, None)      # gripper (position, load) measured once the close settled
    dive_image_error = (None, None)  # target offset from image centre at the last dive correction
    grab_task = None

    def __init__(self, args):
        self.args = args
        self.wrist_ik = WristIK()
        self.model = YOLO(args.model)
        self.labels = {x.strip().lower() for x in args.labels.split(",") if x.strip()}
        if "world" in str(args.model).lower():
            self.model.set_classes(sorted(self.labels))
        # --targeter vlm: Qwen3-VL finds/verifies trash, CSRT tracks between answers,
        # and forward travel is only allowed while the targeter reports DRIVE.
        if args.targeter == "vlm":
            from autonomy.vlm_targeting import HybridTargeter, LlamaServerLocator, Settings, ThreadedLocator
            if args.vlm_backend == "llama":
                locator = self.vlm_client = LlamaServerLocator(args.vlm_url)
            else:
                from autonomy.vlm_targeting import QwenLocator
                locator = QwenLocator(args.vlm_model)
            self.targeter = HybridTargeter(ThreadedLocator(locator.locate, graspable=getattr(locator, 'graspable', None)),
                                           Settings(verify_max_age=args.vlm_verify_max_age))
        self.pi = None
        self.pi_lock = asyncio.Lock()
        self.browsers = set()
        self.state = {"status": "connecting", "hardware": False}
        self.home = None
        self.auto = False
        self.auto_distance_reached = False
        self.target = None
        self.candidate = None
        self.candidate_frames = 0
        self.frame_size = (0, 0)
        self.frame_seen = 0.0
        self.target_seen = time.monotonic()
        self.wrist_delta = 0.0
        self.mode = "manual"
        self.decision_sequence = 0
        self.grab_task = None
        self.grab_active = False
        self.grab_phase = "idle"
        self.grab_target_width = args.grab_target_width
        self.stop_height = args.stop_height
        self.auto_travel_speed = 3.5
        self.auto_arm_speed = 30.0       # scales the wrist-centering gain; 16.5 let the target slide down during approach
        self.grab_speed = 50.0           # dive speed multiplier = grab_speed/20 (was 100 -> 5x; the arm lagged the command)
        # The wrist camera's +Z ray is shallow at the home pose; 35° left
        # the trash too far away for the arm-only dive.  Keep the setting
        # adjustable, but default to a closer 55° approach.
        self.auto_wrist_stop_deg = 43.1
        # Keep the camera ray pitched down toward floor-level litter even
        # when the detected object's center is exactly on the image centerline.
        self.auto_wrist_min_deg = 20.0
        self.ground_offset_cm = -1.0
        self.grab_ik_tolerance = .001
        self.contact_threshold = args.grab_contact_load
        self.grab_debug = None
        self.grab_dive_log = []
        self._pending_frame = None
        self.latest_jpeg = (0.0, None)   # (arrival time, bytes) of the newest camera frame, for grasp checks
        self.grab_attempt = 0
        self._frame_ready = asyncio.Event()
        self.frame_lag_ms = 0.
        self.frames_dropped = 0
        self._dropped_since = 0
        self.state_seen = 0.0
        self.grab_message = "Awaiting grab capture"
        self.camera_exposure = 30
        self.camera_gain = 0
        self.camera_adjusted = 0.
        self.motion_dir = Path(__file__).resolve().parents[1] / "recordings" / "motions"
        self.motion_dir.mkdir(parents=True, exist_ok=True)
        self.motion_recording = False
        self.motion_name = None
        self.motion_started = 0.0
        self.motion_samples = []
        self.motion_task = None
        self.keypoints_path = self.motion_dir.parent / 'placement-keypoints.json'
        self.placement_keypoints = (json.loads(self.keypoints_path.read_text())
                                   if self.keypoints_path.exists() else [])

    async def goto_keypoint(self, point):
        """Move to one recorded pose for editing, including its gripper."""
        try:
            if self.state.get('status') != 'connected' or time.monotonic()-self.state_seen > 1.5:
                await self.grab_notice('Go to refused: fresh servo readings required.', error=True)
                return
            target = {n: float(point['joints'][n]) for n in JOINTS}
            initial = {n: float(self.state[n]) for n in JOINTS}
            if not all(np.isfinite(v) for v in [*target.values(), *initial.values()]):
                raise ValueError('Invalid joint values')
            await self.pi_send({'type':'stop'})
            # Seed goals before enabling torque so hand teaching cannot
            # leave an old servo goal that causes an immediate jump.
            await self.pi_send({'type':'joints','enabled':True,'values':initial})
            await self.pi_send({'type':'arm_torque','enabled':True})
            await self.grab_notice(f"Moving to keypoint: {point['name']}")
            duration = max(.5, 1.5*max(abs(target[n]-initial[n]) for n in JOINTS)/.8)
            ticks = max(2, int(np.ceil(duration/.02)))
            for tick in range(1, ticks+1):
                if self.state.get('status') != 'connected' or time.monotonic()-self.state_seen > 1.5:
                    raise ValueError('Servo feedback unavailable')
                u = tick/ticks
                blend = u*u*(3-2*u)
                await self.pi_send({'type':'joints','enabled':True,
                    'values':{n:initial[n]+blend*(target[n]-initial[n]) for n in JOINTS}})
                await asyncio.sleep(duration/ticks)
            await self.grab_notice(f"Pose commanded: {point['name']}. Adjust the arm, then Update from current pose.")
        except asyncio.CancelledError:
            if self.state.get('status') == 'connected' and time.monotonic()-self.state_seen <= 1.5:
                await self.pi_send({'type':'joints','enabled':True,'values':{n:float(self.state[n]) for n in JOINTS}})
            raise
        except (ValueError, KeyError) as exc:
            await self.grab_notice(f'Go to stopped: {exc}', error=True)
        finally:
            await self.pi_send({'type':'stop'})

    async def place_with_keypoints(self):
        """Replay a snapshot of the taught path after closing on the object."""
        points = json.loads(json.dumps(self.placement_keypoints))
        releases = [i for i,p in enumerate(points) if p['name'].strip().lower() == 'release']
        if len(releases) != 1 or releases[0] == 0:
            await self.grab_notice('Placement needs one release keypoint after the carry poses; holding grip.', error=True)
            return
        release_index = releases[0]
        for point in points:
            pose = point.get('joints', {})
            if not all(n in pose and np.isfinite(pose[n]) for n in JOINTS):
                await self.grab_notice('Invalid placement pose; holding grip.', error=True)
                return
        await self.pi_send({'type': 'stop'})
        previous_target = None
        for index, point in enumerate(points):
            name = point['name'].strip().lower()
            target = dict(point['joints'])
            if index < release_index:
                target['gripper'] = self.args.grab_gripper
            if self.state.get('status') != 'connected' or time.monotonic()-self.state_seen > 1.5:
                await self.grab_notice('Placement stopped: servo feedback unavailable.', error=True)
                return
            initial = dict(previous_target) if previous_target is not None else self._measured_arm_pose(target)
            if index < release_index:
                initial['gripper'] = self.args.grab_gripper
            self.grab_phase = 'placement-' + name
            # Carry moves run faster; releasing into the bin and lifting back out keep the taught pace.
            speed = 1. if name in self.args.place_slow_keypoints else self.args.place_speed
            await self.grab_notice(f'Placement {index+1}/{len(points)}: {name}' + (f' ({speed:g}x)' if speed != 1 else ''))
            # Smoothstep has a peak slope of 1.5: bound peak joint speed
            # to 0.8 rad/s (times the speed factor) even on the large swing over the bin.
            duration = max(.5/speed, 1.5*max(abs(target[n]-initial[n]) for n in JOINTS)/(.8*speed))
            ticks = max(2, int(np.ceil(duration/.02)))
            for tick in range(1, ticks+1):
                if self.state.get('status') != 'connected' or time.monotonic()-self.state_seen > 1.5:
                    await self.grab_notice('Placement stopped: servo feedback unavailable.', error=True)
                    return
                u = tick/ticks
                blend = u*u*(3-2*u)
                pose = {n: initial[n]+blend*(target[n]-initial[n]) for n in JOINTS}
                await self.pi_send({'type':'joints', 'enabled':True, 'values':pose})
                await asyncio.sleep(duration/ticks)
            # Continue immediately from the command endpoint; no settling
            # dwell or extra release hold between keypoints.
            previous_target = target
        await self.grab_notice('Placement complete: released in bin and returned to reset.')
        return True

    def motion_files(self):
        return sorted(p.stem for p in self.motion_dir.glob("*.json") if not p.stem.endswith(".original"))

    def motion_metadata(self):
        result = []
        for path in sorted(self.motion_dir.glob("*.json")):
            if path.stem.endswith(".original"):
                continue
            try:
                data = json.loads(path.read_text())
                result.append({"name": path.stem, "duration": float(data.get("duration", 0)),
                               "samples": len(data.get("samples", []))})
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                result.append({"name": path.stem, "duration": 0, "samples": 0})
        return result

    async def replay_motion(self, name, speed=1.0, bin_release=False):
        path = self.motion_dir / (Path(name).stem + ".json")
        if not path.exists():
            await self.grab_notice(f"Motion '{name}' was not found.", error=True)
            return
        try:
            data = json.loads(path.read_text())
            samples = data.get("samples", [])
            if not samples:
                raise ValueError("motion has no samples")
            self.auto = False
            self.mode = "manual"
            await self.pi_send({"type": "stop"})
            await self.pi_send({"type": "arm_torque", "enabled": True})
            await self.grab_notice(f"Replaying motion '{path.stem}' ({len(samples)} samples).")
            previous = float(samples[0].get("t", 0.0))
            released = False
            initial_grip = float(samples[0]['gripper'])
            for sample in samples:
                delay = max(0.01, min(0.5, (float(sample.get("t", previous)) - previous) / max(.1, speed)))
                await asyncio.sleep(delay)
                values = {name: float(sample[name]) for name in JOINTS if name in sample}
                release_now = bin_release and not released and values.get('gripper', initial_grip) > initial_grip + .1
                if bin_release:
                    values['gripper'] = self.args.grab_open if released or release_now else self.args.grab_gripper
                if len(values) == len(JOINTS):
                    await self.pi_send({"type": "joints", "enabled": True, "values": values})
                if release_now:
                    released = True
                    await self.grab_notice("At bin: opening gripper fully and holding for two seconds before return.")
                    await self._grab_pose(values, 2.0, 'bin-release', arrive='gripper')
                previous = float(sample.get("t", previous))
            await self.grab_notice(f"Finished motion '{path.stem}'.")
        except asyncio.CancelledError:
            await self.pi_send({"type": "stop"})
            await self.grab_notice("Motion replay stopped.")
            raise
        except Exception as exc:
            await self.grab_notice(f"Motion replay failed: {exc}", error=True)
        finally:
            self.motion_task = None

    async def adjust_camera(self, jpeg):
        now = time.monotonic()
        if now - self.camera_adjusted < .30:
            return
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return
        # Use most of the scene, ignoring isolated specular wrapper highlights.
        level = float(np.percentile(frame[::4, ::4], 70))
        clipped = float(np.mean(frame[::4, ::4] > 245))
        exposure, gain = self.camera_exposure, self.camera_gain
        if level < 85 or level > 155 or clipped > .20:
            factor = float(np.clip(np.sqrt(115 / max(level, 1)), .5, 2.))
            if clipped > .20:
                factor = min(factor, .7)
            if factor < 1 and gain > 0:
                gain = max(0, gain - 10)
            elif factor > 1 and exposure >= 300:
                gain = min(60, gain + 10)
            else:
                exposure = int(np.clip(round(exposure * factor), 1, 300))
        # Only speak up when something actually changed. Sending this every
        # second made the Pi re-run v4l2-ctl against the device ffmpeg is
        # streaming from, once a second, stalling and sometimes restarting the
        # capture -- the camera "keeps stopping".
        if (exposure, gain) != (self.camera_exposure, self.camera_gain):
            await self.pi_send({"type": "camera_settings", "exposure": exposure, "gain": gain})
            self.camera_exposure, self.camera_gain = exposure, gain
        self.camera_adjusted = now
        if (exposure, gain) != (30, 0):
            logging.debug("Camera exposure=%s gain=%s brightness=%.0f clipped=%.1f%%", exposure, gain, level, clipped*100)

    async def pi_send(self, message):
        if not self.pi:
            return
        try:
            async with self.pi_lock:
                await self.pi.send(json.dumps(message))
        except (websockets.exceptions.WebSocketException, OSError):
            # pi_loop owns reconnection. A send losing the race with a closing
            # link must not propagate: this runs from the browser handler's
            # finally block, where it surfaced as "Error handling request".
            pass

    async def broadcast_json(self, message):
        dead = []
        for ws in self.browsers:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        self.browsers.difference_update(dead)

    async def broadcast_bytes(self, frame):
        await asyncio.gather(*(ws.send_bytes(frame) for ws in list(self.browsers)), return_exceptions=True)

    def infer(self, jpeg):
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        h, w = image.shape[:2]
        result = self.model.predict(image, conf=self.args.confidence, verbose=False)[0]
        found = []
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            if label.lower() in self.labels:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detection = Detection(label, float(box.conf.item()), x1, y1, x2, y2)
                area_ratio = detection.area / (w * h)
                margin = self.args.target_border_margin
                # During the search/approach phase, reject any box clipped by
                # the image edge. Large rug/background regions commonly touch
                # the bottom edge and otherwise win ``closest()`` by area.
                # Once a grab is already locked, allow the object to leave the
                # frame as it fills the close-up view.
                fully_visible = self.grab_active or (x1 >= margin * w and y1 >= margin * h and
                                 x2 <= (1 - margin) * w and y2 <= (1 - margin) * h)
                if fully_visible and self.args.min_target_area <= area_ratio <= (.85 if self.grab_active else self.args.max_target_area):
                    found.append(detection)
        return found, w, h, result.plot()

    def infer_vlm(self, jpeg):
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        h, w = image.shape[:2]
        snap = self.targeter.step(image, time.monotonic())
        if snap.events:
            logging.info("targeter: %s (%s)", "; ".join(snap.events), snap.status)
        annotated = image.copy()
        color = {"DRIVE": (80, 200, 60), "HOLD": (0, 170, 255), "SEARCHING": (160, 160, 160)}[snap.status]
        rejected_t, rejected = self.targeter.rejected
        if rejected_t is not None and time.monotonic() - rejected_t < 2.:
            for box in rejected:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 60, 230), 2)
                cv2.putText(annotated, "NOT TRASH - ignored", (x1 + 4, max(22, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, .55, (60, 60, 230), 2)
        if snap.box:
            x1, y1, x2, y2 = (int(v) for v in snap.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, f"TARGET {snap.status}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        error = self.targeter.locator.error
        lines = [f"VLM {snap.status}: {snap.reason}",
                 f"VLM ERROR {error[:70]}" if error else
                 ("VLM checking a frame..." if self.targeter.locator.busy else "")
                 + (f"  verified {snap.verified_age:.1f}s ago" if snap.verified_age is not None else "")]
        for i, line in enumerate(lines):
            cv2.putText(annotated, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
            cv2.putText(annotated, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        return snap, w, h, annotated

    @staticmethod
    def overlap(a, b):
        # All configured vocabulary entries are aliases for the same actionable
        # class: trash. A wrapper may alternate between "bottle", "cup", and
        # "discarded packaging", so confirmation follows geometry, not wording.
        if not a or not b:
            return 0.0
        left, top, right, bottom = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = a.area + b.area - intersection
        return intersection / union if union else 0.0

    async def camera_worker(self):
        """Infer on the newest frame available, discarding any backlog.

        YOLO costs about 180 ms a frame (5.5 fps) while the Pi sends 7.7 fps.
        Inferring on every frame in arrival order means that shortfall piles up
        in the socket buffer and the delay grows without bound -- the view falls
        further behind the arm the longer it runs. Dropping whatever arrived
        during the last inference holds the lag at one inference instead.
        """
        while True:
            await self._frame_ready.wait()
            self._frame_ready.clear()
            pending, self._pending_frame = self._pending_frame, None
            if pending is None:
                continue
            raw, arrived = pending
            self.latest_jpeg = (arrived, raw)
            self.frames_dropped += self._dropped_since
            self._dropped_since = 0
            try:
                await self.adjust_camera(raw)
                output = await asyncio.to_thread(self.infer_vlm if self.targeter else self.infer, raw)
                if output:
                    found, w, h, annotated = output
                    now = time.monotonic()
                    if self.targeter:
                        # The hybrid targeter owns acquisition and tracking; ``found`` is its snapshot.
                        self.vlm = found
                        if found.box:
                            self.target = Detection("trash", 1.0, *found.box)
                            self.target_seen = now
                        elif not self.grab_active:
                            self.target = None
                    elif self.target:
                        # Once acquired, only update from the same spatial object.
                        # Other trash remains visible but cannot steal focus.
                        candidate = associated(self.target, found, w, h)
                        if self.grab_active and candidate:
                            # Prefer the containing wrapper over a nested logo
                            # or fragment; preserve its center during close-up.
                            cx, cy = candidate.center
                            containers = [d for d in found if d.x1 <= cx <= d.x2 and d.y1 <= cy <= d.y2
                                          and candidate.area <= d.area <= candidate.area * 2
                                          and d.label == candidate.label
                                          and d.confidence >= candidate.confidence]
                            candidate = max(containers, key=lambda d: d.area, default=candidate)
                        if candidate:
                            self.target = candidate
                            self.target_seen = now
                            self.candidate_frames = self.args.confirm_frames
                        elif now - self.target_seen >= self.args.lost_timeout and not self.grab_active:
                            self.target = None
                            self.candidate = None
                            self.candidate_frames = 0
                    else:
                        candidate = closest(found)
                        if self.overlap(candidate, self.candidate) >= self.args.confirm_iou:
                            self.candidate_frames += 1
                        else:
                            self.candidate_frames = 1 if candidate else 0
                        self.candidate = candidate
                        if candidate and self.candidate_frames >= self.args.confirm_frames:
                            self.target = candidate
                            self.target_seen = now
                    if self.target and not self.targeter:
                        cv2.rectangle(annotated, (int(self.target.x1), int(self.target.y1)),
                                      (int(self.target.x2), int(self.target.y2)), (0, 255, 255), 4)
                        cv2.putText(annotated, "LOCKED TARGET", (int(self.target.x1), max(22, int(self.target.y1) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    self.frame_size = (w, h)
                    self.frame_seen = time.monotonic()
                    if ok:
                        await self.broadcast_bytes(encoded.tobytes())
                        # Age of the frame the viewer is seeing, measured from
                        # its arrival to the moment it goes out.
                        self.frame_lag_ms = round((time.monotonic() - arrived) * 1000, 1)
                    await self.send_autonomy_status()
            except Exception:
                logging.exception("frame processing failed; dropping this frame")

    async def pi_receiver(self, ws):
        """Consume the Pi stream, treating prolonged silence as a dead link.

        The Pi drops a client from its broadcast set when a send exceeds its
        0.4 s bound, but leaves the socket open. Protocol pings keep answering,
        so the connection stays Established and simply never delivers anything
        again -- the camera goes dark with the status still reading "connected".
        State arrives every 0.2 s, so silence past the timeout means the Pi has
        stopped talking to us and the link has to be rebuilt.
        """
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.args.pi_idle_timeout)
            except asyncio.TimeoutError:
                raise ConnectionError(
                    f"no data from Pi for {self.args.pi_idle_timeout:.0f} s; reconnecting") from None
            if isinstance(raw, bytes):
                # Keep only the newest frame; see camera_worker.
                if self._pending_frame is not None:
                    self._dropped_since += 1
                self._pending_frame = (raw, time.monotonic())
                self._frame_ready.set()
                continue
            message = json.loads(raw)
            if message.get("type") == "hello":
                self.home = message.get("home_pose")
                self.state.update(message.get("state", {}))
            elif message.get("type") == "state":
                incoming = message.get("data", {})
                self.state.update(incoming)
                valid_feedback = (incoming.get('status') == 'connected'
                                  and all(isinstance(incoming.get(n), (int, float))
                                          and np.isfinite(incoming[n]) for n in JOINTS))
                self.state_seen = time.monotonic() if valid_feedback else 0.0
                if self.motion_recording and valid_feedback:
                    sample = {name: float(self.state[name]) for name in JOINTS}
                    sample["t"] = time.monotonic() - self.motion_started
                    self.motion_samples.append(sample)
            if message.get("type") in {"hello", "state"}:
                self._add_fk_telemetry()
                if message.get("type") == "state":
                    message["data"] = self.state
                elif message.get("type") == "hello":
                    message["state"] = self.state
            await self.broadcast_json(message)

    def _add_fk_telemetry(self):
        """Attach the same URDF-derived XYZ estimates used by grab diagnostics."""
        measured = {name: self.state.get(name) for name in JOINTS}
        goal_values = self.state.get("servo_goal")
        goal = dict(measured)
        if isinstance(goal_values, (list, tuple)) and len(goal_values) >= len(JOINTS):
            goal.update(dict(zip(JOINTS, goal_values)))
        try:
            self.state["estimated_position"] = self.wrist_ik.fk(measured)[:3, 3].tolist()
            self.state["commanded_position"] = self.wrist_ik.fk(goal)[:3, 3].tolist()
            self.state["position_error"] = float(np.linalg.norm(np.asarray(self.state["estimated_position"]) - np.asarray(self.state["commanded_position"])))
        except (KeyError, TypeError, ValueError):
            pass

    async def pi_loop(self):
        while True:
            try:
                async with websockets.connect(self.args.pi, max_size=4_000_000) as ws:
                    self.pi = ws
                    self.state["status"] = "connected"
                    await self.pi_receiver(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state["status"] = f"Pi disconnected: {exc}"
                await self.broadcast_json({"type": "state", "data": self.state})
                await asyncio.sleep(2)
            finally:
                self.pi = None

    async def send_autonomy_status(self, decision=None):
        self.decision_sequence += 1
        current_wrist_angle = 0.0
        if self.home and all(name in self.state for name in JOINTS):
            current_wrist_angle = self._ik_ground_angle_deg(self._measured_arm_pose(self.home))
        await self.broadcast_json({"type": "autonomy", "sequence": self.decision_sequence,
                                   "timestamp": time.time(), "enabled": self.auto, "mode": self.mode,
                                   "decision": decision, "candidate_frames": self.candidate_frames,
                                   "grab_active": self.grab_active, "grab_phase": self.grab_phase,
                                   "grab_attempt": self.grab_attempt,
                                   "grab_debug": self.grab_debug,
                                   "grab_message": self.grab_message,
                                   "grab_ik_tolerance": self.grab_ik_tolerance,
                                   "stop_height": self.stop_height,
                                   "auto_travel_speed": self.auto_travel_speed,
                                   "auto_arm_speed": self.auto_arm_speed,
                                   "current_wrist_angle_deg": round(current_wrist_angle, 1),
                                   "grab_speed": self.grab_speed,
                                   "auto_wrist_stop_deg": self.auto_wrist_stop_deg,
                                   "ground_offset_cm": self.ground_offset_cm,
                                   "motion_recording": self.motion_recording,
                                   "motion_name": self.motion_name,
                                   "motion_samples": len(self.motion_samples),
                                   "motions": self.motion_files(),
                                   "motion_metadata": self.motion_metadata(),
                                   "placement_keypoints": self.placement_keypoints,
                                   "contact_threshold": self.contact_threshold,
                                   "frame_lag_ms": self.frame_lag_ms,
                                   "frames_dropped": self.frames_dropped,
                                   "confirm_frames": self.args.confirm_frames,
                                   "targeter": self.args.targeter,
                                   "vlm": None if not self.vlm else {
                                       "status": self.vlm.status, "reason": self.vlm.reason,
                                       "verified_age_s": self.vlm.verified_age,
                                       "busy": self.targeter.locator.busy,
                                       "error": self.targeter.locator.error},
                                   "target": None if not self.target else {
                                       "label": self.target.label, "confidence": self.target.confidence,
                                       "area": self.target.area, "box": [self.target.x1, self.target.y1,
                                                                          self.target.x2, self.target.y2]}})

    async def grab_notice(self, message, error=False):
        self.grab_message = message
        logging.info("grab: %s", message)
        await self.broadcast_json({"type": "error" if error else "notice", "message": message})

    async def _grab_pose(self, pose, duration, phase, arrive=None, tolerance=.12, timeout=3.):
        """Command `pose` for at least `duration`, returning whether it landed.

        With `arrive`, keep commanding until that joint reaches its target
        instead of trusting a fixed hold time: crossing the full gripper range
        from closed takes well over the nominal hold, so a blind wait reports a
        healthy gripper as a calibration fault. Callers that expect to stop
        short -- closing on an object -- leave `arrive` unset.
        """
        self.grab_phase = phase
        await self.grab_notice(f"Grab: {phase} — commanding arm pose")
        start = time.monotonic()
        deadline = start + duration
        while True:
            await self.pi_send({"type": "joints", "enabled": True, "values": pose})
            await asyncio.sleep(.12)
            now = time.monotonic()
            if now < deadline:
                continue
            if arrive is None:
                return True
            if now - self.state_seen <= 1.5 and abs(float(self.state.get(arrive, pose[arrive])) - pose[arrive]) <= tolerance:
                return True
            if now - start >= timeout:
                return False

    def _measured_arm_pose(self, fallback):
        return {name: float(self.state.get(name, fallback[name])) for name in JOINTS}

    def _ik_ground_angle_deg(self, pose):
        """Angle of the IK wrist +Z ray from horizontal ground (0..90°)."""
        ray = np.asarray(self.wrist_ik.fk(pose)[:3, 2], dtype=float)
        return float(np.degrees(np.arcsin(np.clip(abs(ray[2]), 0., 1.))))

    async def _wait_lift_settled(self, pose, update_debug):
        """Wait for fresh FK feedback; stop on sustained lack of progress, not one lag sample."""
        commanded = self.wrist_ik.fk(pose)[:3, 3]
        last_feedback = self.state_seen
        best_error = float('inf')
        progress_at = time.monotonic()
        samples = settled = 0
        while True:
            now = time.monotonic()
            if now - self.state_seen > 1.5:
                await self.grab_notice("Lift stopped: stale telemetry; grip remains closed.", error=True)
                return False
            await self.pi_send({'type': 'joints', 'enabled': True, 'values': pose})
            if self.state_seen > last_feedback:
                last_feedback = self.state_seen
                measured = self._measured_arm_pose(pose)
                error = float(np.linalg.norm(self.wrist_ik.fk(measured)[:3, 3] - commanded))
                samples += 1
                update_debug(measured)
                self.grab_debug.update(lift_error_mm=error*1000, lift_feedback_samples=samples)
                await self.send_autonomy_status()
                # The SO-101 has several millimetres of backlash under a
                # closed-gripper load.  A 3 mm gate made every lift stop on
                # the first feedback sample even when the arm was still
                # physically rising, so allow the measured transient to
                # settle while retaining a hard stall detector below.
                settled = settled + 1 if error <= .008 else 0
                if settled >= 2:
                    return True
                if error < best_error - .001:
                    best_error, progress_at = error, now
                elif samples >= 5 and now - progress_at >= 2.:
                    hold = dict(measured, gripper=pose['gripper'])
                    await self.pi_send({'type': 'joints', 'enabled': True, 'values': hold})
                    reason = (self.grab_debug or {}).get('dive_stop_reason')
                    await self.grab_notice(
                        f"Lift: stopped rising with a {error*1000:.1f} mm gap; holding here with the grip closed"
                        + (f" (dive had stopped on {reason})." if reason else "."))
                    return False
            await asyncio.sleep(.12)

    def _write_dive_log(self, rows):
        """Persist the dive's per-step telemetry next to the probe logs."""
        if not rows:
            return "No dive samples recorded."
        folder = Path(__file__).resolve().parents[1] / "screenshots"
        folder.mkdir(exist_ok=True)
        path = folder / f"grab-dive-{time.time_ns()}.csv"
        # Union of keys, not just the first row's: a row gaining a field
        # mid-dive would otherwise make DictWriter raise and lose the log.
        fields = list(dict.fromkeys(key for row in rows for key in row))
        try:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, restval="")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            return f"Dive log could not be written: {exc}"
        return f"Dive log: {path.name} ({len(rows)} steps)."

    def _set_grab_debug(self, start, commanded, measured, forward, progress=0., width=0., deviation=0.):
        initial = self.wrist_ik.fk(start)
        actual = self.wrist_ik.fk(measured)
        delta = actual[:3, 3] - initial[:3, 3]
        along = float(np.dot(delta, forward))
        commanded_position = self.wrist_ik.fk(commanded)[:3, 3]
        # Carry the contact readout forward: this rebuilds grab_debug from
        # scratch on every call, and the lift phase calls it too, which
        # otherwise blanks the load telemetry exactly when it is being read.
        carried = {k: v for k, v in (getattr(self, "grab_debug", None) or {}).items()
                   if k.startswith("contact_") or k == "dive_stop_reason"}
        self.grab_debug = {
            "start_joints": dict(start), "commanded_joints": dict(commanded),
            "measured_joints": dict(measured), "forward": list(map(float, forward)),
            "start_position": initial[:3, 3].tolist(),
            "commanded_position": self.wrist_ik.fk(commanded)[:3, 3].tolist(),
            "measured_position": actual[:3, 3].tolist(),
            "position_error_m": float(np.linalg.norm(commanded_position - actual[:3, 3])),
            "commanded_along_ray_m": float(np.dot(commanded_position - initial[:3, 3], forward)),
            "along_ray_m": along, "cross_track_m": float(np.linalg.norm(delta - along * np.asarray(forward))),
            "progress": round(float(progress), 4), "target_width": round(float(width), 3),
            "deviation": round(float(deviation), 4), "ray_axis": "gripper_frame_link +Z", **carried}

    async def grab_routine(self):
        """Bounded, explicit dive/grip/lift sequence; STOP cancels it."""
        self.grab_active = True
        self.mode = "grab"
        try:
            # Grab never owns the base; cancel any stale manual/auto wheel
            # command immediately, including while waiting for vision/home.
            await self.pi_send({"type": "stop"})
            if (self.state.get('status') != 'connected' or
                    time.monotonic() - self.state_seen > 1.5 or any(n not in self.state for n in JOINTS)):
                await self.grab_notice("Grab refused: fresh measured joint positions are required.", error=True)
                return
            # A grab is deliberately a camera-guided action.  Do not start a
            # blind dive when the detector has not produced a recent target;
            # this used to make the arm move into the rug/background after a
            # stale or false lock disappeared.
            # A momentary tracker drop should not refuse the grab: give the target a moment to come back.
            waited = time.monotonic()
            while ((self.target is None or time.monotonic() - self.target_seen > max(.7, self.args.grab_track_hold))
                   and time.monotonic() - waited < 2.):
                await asyncio.sleep(.1)
            if (self.target is None or
                    time.monotonic() - self.target_seen > max(.7, self.args.grab_track_hold)):
                await self.grab_notice(
                    "Grab refused: no fresh locked target in view; center the trash and try again.",
                    error=True)
                return
            capture_pose = {name: float(self.state[name]) for name in JOINTS}
            opened = dict(capture_pose, gripper=self.args.grab_open)
            if not await self._grab_pose(opened, .8, "open-gripper", arrive='gripper'):
                measured = self.state.get('gripper')
                await self.grab_notice(
                    f"Grab: gripper opened to {measured:.3f} rather than {self.args.grab_open:.3f} rad; diving anyway."
                    if measured is not None else "Grab: no gripper feedback while opening; diving anyway.")
            capture_pose = self._measured_arm_pose(opened)
            capture_pose['gripper'] = self.args.grab_open
            # Primary contact signal: servo load on every arm joint. Which
            # joint carries a contact depends on the pose, so all are watched.
            # See autonomy/contact.py for how the thresholds were measured.
            contact_detector = LoadContactDetector(self.contact_threshold)
            home = dict(capture_pose)
            start_transform, forward = self.wrist_ik.capture(capture_pose)
            start_xyz = start_transform[:3, 3].copy()
            ground_z = self.args.grab_ground_z + self.ground_offset_cm / 100.0
            # Follow the captured forward approach vector down to the ground:
            # advance the gripper along `forward` (which points forward and
            # down) until its height reaches the predicted ground plane.
            # Orientation floats so the arm
            # can reach; while the target is visible the vector is steered toward
            # it, and when it drops out of view the last vector is held.
            forward = forward / np.linalg.norm(forward)
            ray = forward.copy()
            self._set_grab_debug(capture_pose, capture_pose, capture_pose, forward)
            self.grab_phase = "approach-dive"
            await self.grab_notice(
                f"Grab: approaching along the captured forward vector {forward.round(3).tolist()} "
                f"from {start_xyz.round(3).tolist()} m toward the ground plane at z={ground_z*1000:.0f} mm. "
                f"Steers the IK arm to keep the target centered; stops only at the predicted ground plane. "
                f"Dive speed {self.grab_speed/20.0:.2f}x. STOP cancels.")
            progress = 0.0
            contact = False
            dive = dict(capture_pose)
            reaim = 0.
            dive_started = time.monotonic()
            dive_log = self.grab_dive_log = []
            last_feedback = self.state_seen
            last_vision_correction = 0.0
            self.dive_image_error = (None, None)
            # The sideways sign between image x and the pan direction is not calibrated: if a correction makes the
            # target drift further off-centre on the same side twice in a row, flip it for the rest of the dive.
            yaw_sign, last_ex, last_yaw, worse = 1.0, None, 0.0, 0
            stop_reason = "reached the ground plane"
            def gripper_z():
                return self.wrist_ik.fk(dive)[2, 3]
            while gripper_z() > ground_z:
                # Command timing must not be gated by the slower telemetry
                # stream. Reuse the latest fresh measurement between packets.
                if time.monotonic() - self.state_seen > 1.5:
                    raise ValueError("Fresh joint feedback lost during dive")
                last_feedback = self.state_seen
                measured = self._measured_arm_pose(dive)
                errors = {name: abs(dive[name] - measured[name]) for name in JOINTS if name != "gripper"}
                worst = max(errors, key=errors.get)
                width_ratio = ((self.target.x2 - self.target.x1) / self.frame_size[0]) if self.target and self.frame_size[0] and time.monotonic() - self.frame_seen < .7 else 0.
                self._set_grab_debug(capture_pose, dive, measured, ray, progress, width_ratio, errors[worst])
                self.grab_debug.update({"worst_joint": worst, "joint_errors": errors,
                                        "reaim_deg": round(float(np.degrees(reaim)), 2),
                                        "current_ray": list(map(float, ray))})
                touched = contact_detector.update(self.state.get('servo_load'), progress * 1000,
                                                  threshold=self.contact_threshold)
                self.grab_debug.update(contact_detector.debug())
                currents = self.state.get('servo_current') or []
                dive_log.append({"t_s": round(time.monotonic() - dive_started, 2),
                                 "mm": round(progress * 1000, 1),
                                 "measured_mm": round(self.grab_debug["along_ray_m"] * 1000, 1),
                                 "xyz_error_mm": round(self.grab_debug["position_error_m"] * 1000, 2),
                                 "width": round(width_ratio, 4),
                                 "worst_joint": worst, "deviation": round(errors[worst], 4),
                                 "watched": contact_detector.watched,
                                 "threshold": contact_detector.threshold,
                                 "armed": contact_detector.armed,
                                 "run": contact_detector.run,
                                 "fired": bool(touched),
                                 # Where the target sits in the image during the dive (x, y in -1..1; 0 = centre).
                                 "image_x": self.dive_image_error[0],
                                 "image_y": self.dive_image_error[1],
                                 "yaw_sign": yaw_sign,
                                 "target_fresh": bool(self.target and time.monotonic() - self.target_seen < .7),
                                 **{f"load_{n}": v for n, v in contact_detector.loads.items()},
                                 **{f"current_{n}": (float(currents[i]) if i < len(currents) else None)
                                    for i, n in enumerate(JOINTS)}})
                await self.send_autonomy_status()

                # Steer the approach vector toward the target while it is in
                # view; hold the last vector when it is not. Bounded so the
                # correction cannot walk the approach away from the capture.
                # During the dive, keep the geometric lock briefly through a
                # missed detector frame; the next frame can then correct the
                # ray instead of abandoning camera feedback mid-grab.
                if (self.target and self.target_seen > last_vision_correction
                        and time.monotonic() - self.target_seen < .7 and all(self.frame_size)):
                    # Visual servo: on each new camera frame, turn the camera (and the dive path with it) toward
                    # the target by a fraction of its angular offset from the image centre.
                    last_vision_correction = self.target_seen
                    cx, cy = self.target.center
                    ex, ey = 2*cx/self.frame_size[0]-1, 2*cy/self.frame_size[1]-1
                    self.dive_image_error = (round(float(ex), 3), round(float(ey), 3))
                    self.grab_debug["image_error"] = list(self.dive_image_error)
                    gain = self.args.grab_servo_gain
                    if last_yaw and last_ex is not None and ex * last_ex > 0:
                        worse = worse + 1 if abs(ex) > abs(last_ex) + .02 else 0
                        if worse >= 2:
                            yaw_sign, worse = -yaw_sign, 0
                            await self.grab_notice(f"Dive: sideways correction was moving the target away; flipped its sign "
                                                   f"(now {yaw_sign:+.0f}; set --grab-pan-sign to {-self.args.grab_pan_sign:+.0f} "
                                                   f"if this keeps happening).")
                    yaw = 0. if abs(ex) < .03 else -yaw_sign*self.args.grab_pan_sign*ex*np.radians(self.args.camera_hfov/2)*gain
                    last_ex, last_yaw = ex, yaw
                    pitch = 0. if abs(ey) < .03 else self.args.wrist_sign*ey*np.radians(self.args.camera_vfov/2)*gain
                    steered = rotation([0., 0., 1.], yaw) @ ray
                    horiz = np.array([steered[0], steered[1], 0.])
                    if np.linalg.norm(horiz) > 1e-6:
                        side = np.cross([0., 0., 1.], horiz/np.linalg.norm(horiz))
                        steered = rotation(side, pitch) @ steered
                    # Sideways, bending the path swings the base pan and the camera with it; up/down, the IK
                    # below also pitches the camera to match the path (the wrist is the only free pointing joint).
                    ray, reaim = clamp_direction(steered, forward, self.args.grab_max_reaim)
                next_distance = progress + self.args.grab_approach_step
                # Local increments avoid swinging the entire accumulated path
                # sideways whenever the visual direction changes.
                target = self.wrist_ik.fk(dive)[:3, 3] + ray*self.args.grab_approach_step
                hit_ground = target[2] <= ground_z
                if hit_ground:                                  # this step crosses the floor
                    target[2] = ground_z
                # Preserve continuity of the command through encoder lag.
                # Camera feedback adjusts the ray; do not reset the command
                # backwards to a lagging measured joint at every packet.
                ik_seed = dive.copy()
                ik_seed['gripper'] = dive['gripper']
                def solve(elevation_weight):
                    return self.wrist_ik.solve_position(ik_seed, target,
                        position_tolerance=.0005, orient_weight=.01,
                        orientation=self.wrist_ik.fk(dive)[:3, :3],
                        elevation=float(ray[2] / np.linalg.norm(ray)), elevation_weight=elevation_weight)
                try:
                    try:
                        next_pose = solve(self.args.grab_elevation_weight)
                    except ValueError:
                        # Some poses cannot also pitch the camera to the path: keep going on position alone.
                        next_pose = solve(0.)
                except ValueError as exc:
                    stop_reason = f"IK could not continue after {progress*1000:.0f} mm: {exc}"
                    self.grab_debug['dive_stop_reason'] = stop_reason
                    await self.grab_notice(f"Grab stopped: {stop_reason}. No grip; treating it as a miss.", error=True)
                    return "missed"                          # the retry loop backs up and approaches again
                # IK can distribute a small Cartesian step into a large jump
                # on one servo near a kinematic boundary. Slew-limit every
                # joint so the wrist moves continuously and the camera keeps
                # the target in view instead of losing it on a jerk.
                speed_factor = self.grab_speed / 20.0
                max_step = self.args.grab_joint_slew
                largest = max(abs(float(next_pose[name]) - float(dive[name])) for name in JOINTS if name != "gripper")
                slew_factor = 1.0
                if largest > max_step:
                    factor = max_step / largest
                    slew_factor = factor
                    for name in JOINTS:
                        if name != "gripper":
                            next_pose[name] = dive[name] + (next_pose[name] - dive[name]) * factor
                self.grab_debug["joint_slew_factor"] = round(float(slew_factor), 3)
                # Stream an eased segment at 50 Hz instead of a single step
                # every encoder packet. Keep the command continuous through
                # backlash; measured joints still drive contact and diagnostics.
                segment_start = dict(dive)
                segment_duration = max(.03, self.args.grab_approach_period / speed_factor)
                ticks = max(2, int(segment_duration / .02))
                self.grab_debug.update(dive_speed_multiplier=speed_factor,
                                       dive_segment_seconds=segment_duration,
                                       dive_requested_mm_s=self.args.grab_approach_step * 1000 / segment_duration)
                for tick in range(1, ticks + 1):
                    if time.monotonic() - self.state_seen > 1.5:
                        raise ValueError("Fresh joint feedback lost during dive")
                    u = tick / ticks
                    blend = u*u*(3 - 2*u)
                    dive = {name: segment_start[name] + blend*(next_pose[name]-segment_start[name]) for name in JOINTS}
                    await self.pi_send({"type": "joints", "enabled": True, "values": dive})
                    await asyncio.sleep(segment_duration / ticks)
                progress += self.args.grab_approach_step * slew_factor
                if hit_ground and self.wrist_ik.fk(dive)[2, 3] <= ground_z + .0005:
                    stop_reason = f"reached grab height plane at z={ground_z*1000:.0f} mm"
                    break

            await self.pi_send({"type": "stop"})
            self.grab_debug['dive_stop_reason'] = stop_reason
            await self.grab_notice(f"Grab: dive stopped — {stop_reason}. {self._write_dive_log(dive_log)}")
            self.grab_dive_log = []
            # The dive command runs ahead of the arm (it has lagged 15-65 mm at the end). Hold the bottom pose
            # until the arm has actually stopped, close fully while holding still, and only then lift.
            settle_s, settle_error = await self._settle(dive, "dive-settle", lambda m: {n: m[n] for n in JOINTS if n != 'gripper'})
            closed = self._measured_arm_pose(dive)
            closed["gripper"] = self.args.grab_gripper
            grip_s, _ = await self._settle(closed, "grip", lambda m: {'gripper': m['gripper']},
                                           minimum=self.args.grab_grip_time)
            load = self.state.get('servo_load') or []
            self.grip_reading = (float(self.state.get('gripper', closed['gripper'])),
                                 float(load[JOINTS.index('gripper')]) if len(load) > JOINTS.index('gripper') else None)
            await asyncio.sleep(self.args.grab_hold_time)
            await self.grab_notice(f"Grab: arm settled at the bottom in {settle_s:.2f}s "
                                   f"({settle_error*1000:.0f} mm from the dive target); gripper closed in {grip_s:.2f}s "
                                   f"to {float(self.state.get('gripper', closed['gripper'])):+.2f}. Lifting.")
            if self.vlm_client and self.args.grab_attempts > 1:
                held = await self._verify_grasp(closed)
                if held is False:
                    released = self._measured_arm_pose(closed)
                    released['gripper'] = self.args.grab_open
                    await self._grab_pose(released, .5, "release-miss", arrive='gripper')
                    await self._grab_pose(dict(capture_pose, gripper=self.args.grab_open), 1.2, "return-to-aim")
                    return "missed"
            if self.placement_keypoints:
                return "placed" if await self.place_with_keypoints() else None
            lift_start = self.wrist_ik.fk(self._measured_arm_pose(closed))
            lift = self._measured_arm_pose(closed)
            lift['gripper'] = closed['gripper']
            for step in range(1, 16):
                try:
                    lift = self.wrist_ik.solve(lift, lift_start, 0., desired_position=lift_start[:3, 3] + np.array([0., 0., step*.002]))
                except ValueError:
                    break
                self.grab_phase = 'lift-and-settle'
                if not await self._wait_lift_settled(lift, lambda measured: self._set_grab_debug(capture_pose, lift, measured, forward, progress)):
                    break
            await asyncio.sleep(.4)
            measured_lift = self.wrist_ik.fk(self._measured_arm_pose(lift))[:3, 3] - lift_start[:3, 3]
            visible = self.target is not None and time.monotonic() - self.target_seen < .7
            await self.grab_notice(f"Lift finished: measured rise {measured_lift[2]*1000:.1f} mm; target {'still visible (retention candidate, not proof)' if visible else 'not visible; grasp unconfirmed'}.")
            putaway = self.motion_dir / "put-away-1.json"
            if putaway.exists():
                self.grab_phase = "put-away"
                await self.grab_notice("Grip sequence complete; grasp unconfirmed. Running put-away-1 at 2x speed.")
                await self.replay_motion("put-away-1", speed=2.0, bin_release=True)
                await self.grab_notice("Put-away motion finished; release commanded over the bin.")
                return "placed"
            else:
                # Fallback when no put-away demonstration has been recorded.
                self.grab_phase = "return-home"
                home_pose = dict(self.home) if self.home else dict(home)
                home_pose['gripper'] = self.args.grab_gripper
                await self._grab_pose(home_pose, self.args.grab_grip_time, "return-home")
                release_pose = dict(home_pose)
                release_pose['gripper'] = self.args.grab_open
                await self._grab_pose(release_pose, self.args.grab_grip_time, "release-object", arrive='gripper')
                await self.grab_notice("Returned home, lifted, and released the object.")
                return "placed"
        except asyncio.CancelledError:
            await self.pi_send({"type": "stop"})
            raise
        finally:
            # STOP cancels mid-dive, and those are the runs worth reading.
            # Writing only on a clean exit threw that telemetry away.
            if self.grab_dive_log:
                await self.grab_notice(f"Grab ended early. {self._write_dive_log(self.grab_dive_log)}")
                self.grab_dive_log = []
            await self.pi_send({"type": "stop"})
            self.grab_active = False
            self.grab_phase = "idle"
            self.mode = "search" if self.auto else "manual"
            await self.send_autonomy_status()

    async def _fresh_frame(self, timeout=2.):
        """A camera frame taken after now (the arm has stopped), or None."""
        since = time.monotonic()
        while self.latest_jpeg[0] <= since + .05:
            if time.monotonic() - since > timeout:
                return None
            await asyncio.sleep(.03)
        return cv2.imdecode(np.frombuffer(self.latest_jpeg[1], np.uint8), cv2.IMREAD_COLOR)

    async def _verify_grasp(self, closed):
        """Lift straight up and decide whether the gripper holds the trash, by majority of three votes:

        - parallax: the trash box at half height vs. the top. Held trash rises with the camera and keeps its size;
          trash left on the floor shrinks sharply (the camera's height above it roughly doubles) or leaves view.
        - bottom strip: "is there an object at the bottom of the image, close to the camera?" (the jaws are often
          out of view at this camera angle, so a held object only pokes in from the bottom edge).
        - held_or_empty: the original whole-frame question.
        Snapshots and votes are saved under recordings/grasp-checks/ for labelling.
        Returns True/False, or None when the check could not run (the caller then proceeds unverified).
        """
        from autonomy.vlm_targeting import BOTTOM_PROMPT, GRASP_PROMPT, bottom_strip, grasp_answer
        start = self.wrist_ik.fk(self._measured_arm_pose(closed))
        lift = self._measured_arm_pose(closed)
        lift['gripper'] = closed['gripper']
        path = [dict(lift)]                                # straight up (camera angle kept), 5 mm per IK step
        rise = max(0., self.args.grab_verify_height - float(start[2, 3]))
        for step in range(1, int(np.ceil(rise / .005)) + 1):
            try:
                lift = self.wrist_ik.solve(lift, start, 0., desired_position=start[:3, 3] + np.array([0., 0., min(rise, step * .005)]))
            except ValueError:
                break
            lift['gripper'] = closed['gripper']
            path.append(dict(lift))
        half = len(path) // 2
        seconds_per_step = .005 / self.args.grab_lift_speed
        still = lambda m: {n: m[n] for n in JOINTS if n != 'gripper'}
        await self._stream_path(path[:half + 1], half * seconds_per_step, "verify-lift")
        await self._settle(path[half], "verify-lift", still, timeout=1.)
        low_image = await self._fresh_frame()
        await self._stream_path(path[half:], (len(path) - 1 - half) * seconds_per_step, "verify-lift")
        await self._settle(lift, "verify-lift", still, timeout=1.5)
        image = await self._fresh_frame()
        if image is None or low_image is None:
            await self.grab_notice("Grasp check: no fresh camera frame; continuing unverified.", error=True)
            return None

        def largest_box(frame):
            boxes = self.vlm_client.locate(frame)['boxes']
            return max((b['box'] for b in boxes), key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)

        try:
            low_box, high_box, held_answer, bottom_answer = await asyncio.gather(
                asyncio.to_thread(largest_box, low_image), asyncio.to_thread(largest_box, image),
                asyncio.to_thread(self.vlm_client.ask, [image], GRASP_PROMPT, 8),
                asyncio.to_thread(self.vlm_client.ask, [bottom_strip(image)], BOTTOM_PROMPT, 8))
        except Exception as exc:
            await self.grab_notice(f"Grasp check unavailable ({type(exc).__name__}); continuing unverified.", error=True)
            return None
        area = lambda b: (b[2]-b[0])*(b[3]-b[1]) if b else 0.
        ratio = area(high_box) / area(low_box) if low_box and high_box else None
        if low_box and not high_box:
            parallax = False                               # it was there lower down and did not come up with us
        elif ratio is None:
            parallax = None                                # nothing seen at half height: no evidence either way
        else:
            parallax = ratio >= self.args.grasp_parallax_min
        votes = {'parallax': parallax,
                 'bottom_strip': grasp_answer(bottom_answer[0], 'YES', 'NO'),
                 'held_or_empty': grasp_answer(held_answer[0], 'HELD', 'EMPTY')}
        cast = [v for v in votes.values() if v is not None]
        held = bool(cast) and sum(cast) * 2 > len(cast)
        if cast and sum(cast) * 2 == len(cast):            # a tie: parallax is the most direct evidence
            held = bool(parallax) if parallax is not None else False
        # A grasp counts if the camera confirms it OR the jaws did not fully close: something between them (e.g. a
        # lemon hanging below the frame) is physical evidence. A full close casts no vote; the camera decides then.
        grip_position, grip_load = self.grip_reading
        jaws_held = grip_position >= self.args.grasp_empty_gripper + self.args.grasp_gripper_margin
        votes['gripper'] = True if jaws_held else None
        if jaws_held:
            held = True
        height = self.wrist_ik.fk(self._measured_arm_pose(lift))[2, 3] * 1000
        shown = ', '.join(f"{k}={'-' if v is None else 'held' if v else 'empty'}" for k, v in votes.items())
        await self.grab_notice(f"Grasp check at {height:.0f} mm: {'HELD' if held else 'MISSED'} ({shown}"
                               + (f"; box size ratio {ratio:.2f}" if ratio is not None else "")
                               + f"; gripper stopped at {grip_position:+.2f}"
                               + (f", load {grip_load:.0f}" if grip_load is not None else "") + ").")
        try:
            folder = self.motion_dir.parent / "grasp-checks" / time.strftime("%Y%m%d-%H%M%S")
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / "half.jpg"), low_image)
            cv2.imwrite(str(folder / "top.jpg"), image)
            (folder / "check.json").write_text(json.dumps({
                "held": held, "votes": votes, "box_ratio": ratio, "half_box": low_box, "top_box": high_box,
                "answers": {"held_or_empty": held_answer[0], "bottom_strip": bottom_answer[0]},
                "height_mm": round(height), "gripper_position": grip_position, "gripper_load": grip_load,
                "actual_outcome": None}, indent=1))
        except OSError:
            logging.exception("could not save grasp check snapshots")
        return held

    async def _stream_path(self, path, duration, phase):
        """Follow a list of poses smoothly (ease in/out along the whole path) instead of jumping to the end."""
        self.grab_phase = phase
        if len(path) < 2 or duration <= 0:
            await self.pi_send({"type": "joints", "enabled": True, "values": path[-1]})
            return
        ticks = max(2, int(np.ceil(duration / .02)))
        for tick in range(1, ticks + 1):
            u = tick / ticks
            x = u * u * (3 - 2 * u) * (len(path) - 1)        # smoothstep position along the path
            i = min(int(x), len(path) - 2)
            f = x - i
            pose = {n: path[i][n] + f * (path[i + 1][n] - path[i][n]) for n in JOINTS}
            await self.pi_send({"type": "joints", "enabled": True, "values": pose})
            await asyncio.sleep(duration / ticks)

    async def _settle(self, pose, phase, watched, minimum=0., timeout=2.5, still=.01, samples=3):
        """Keep commanding `pose` until the watched joints stop moving across `samples` fresh servo readings.

        Returns (seconds taken, remaining gripper position error in metres vs. `pose`).
        """
        self.grab_phase = phase
        start = time.monotonic()
        readings, last_seen = [], None
        while True:
            await self.pi_send({"type": "joints", "enabled": True, "values": pose})
            await asyncio.sleep(.05)
            now = time.monotonic()
            if self.state_seen != last_seen and all(n in self.state for n in JOINTS):
                last_seen = self.state_seen
                readings = (readings + [watched({n: float(self.state[n]) for n in JOINTS})])[-samples:]
            stopped = (len(readings) == samples and
                       max(max(r[k] for r in readings) - min(r[k] for r in readings) for k in readings[0]) < still)
            if (stopped and now - start >= minimum) or now - start >= timeout:
                measured = self._measured_arm_pose(pose)
                error = float(np.linalg.norm(self.wrist_ik.fk(measured)[:3, 3] - self.wrist_ik.fk(pose)[:3, 3]))
                return now - start, error

    async def _back_up(self, metres, speed=.05):
        end = time.monotonic() + metres / speed
        while time.monotonic() < end:
            await self.pi_send({"type": "drive", "enabled": True, "x": -speed * self.args.drive_sign, "y": 0, "theta": 0})
            await asyncio.sleep(.1)
        await self.pi_send({"type": "stop"})

    def _log_auto_tick(self, command, decision, joints):
        """Append one row per auto tick to recordings/auto-log/auto-YYYYMMDD.csv (for tuning the approach)."""
        if getattr(self, "motion_dir", None) is None:     # hubs built without __init__ (tests)
            return
        folder = self.motion_dir.parent / "auto-log"
        path = folder / time.strftime("auto-%Y%m%d.csv")
        fields = ["time", "mode", "vlm", "x_error", "y_error", "forward_m_s", "turn_deg_s", "wrist_step_rad",
                  "wrist_delta", "wrist_cmd", "wrist_measured", "wrist_pitch_deg", "arm_settled", "distance_reached"]
        row = [time.strftime("%H:%M:%S") + f"{time.time() % 1:.2f}"[1:], command.mode,
               self.vlm.status if self.vlm else "", decision.get("x_error", ""), decision.get("y_error", ""),
               decision.get("forward_m_s", ""), decision.get("turn_deg_s", ""), round(command.wrist_delta, 5),
               round(self.wrist_delta, 4), round(joints.get("wrist_flex", 0.), 4),
               round(float(self.state.get("wrist_flex", float("nan"))), 4), decision.get("wrist_pitch_deg", ""),
               decision.get("arm_settled", ""), self.auto_distance_reached]
        try:
            folder.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", newline="") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(fields)
                writer.writerow(row)
        except OSError:
            logging.exception("auto log write failed")

    def _slew_auto_joints(self, joints, period):
        """Limit how fast auto moves the arm: large jumps (e.g. back to home when auto starts) become smooth moves."""
        if self.auto_cmd is None:
            self.auto_cmd = self._measured_arm_pose(joints) if all(n in self.state for n in JOINTS) else dict(joints)
        step = self.args.auto_joint_speed * period
        self.auto_cmd = {n: self.auto_cmd[n] + max(-step, min(step, joints[n] - self.auto_cmd[n])) for n in JOINTS}
        return dict(self.auto_cmd)

    def _start_auto(self):
        """Begin an auto run from the home pose (see the auto_mode handler for why the wrist offset resets)."""
        self.auto_cmd = None             # slew from wherever the arm is now
        self.auto_progress_t = time.monotonic()
        self.wrist_delta = 0.0
        self.auto_distance_reached = False
        self.auto_complete = False
        self.auto = True
        self.mode = "search"

    def _target_fresh(self):
        return self.target is not None and time.monotonic() - self.target_seen < .5

    async def _path_clear(self):
        """Ask the VLM about the newest camera frame; None if it could not be asked."""
        if not self.vlm_client or self.latest_jpeg[1] is None or time.monotonic() - self.latest_jpeg[0] > 1.:
            return None
        image = cv2.imdecode(np.frombuffer(self.latest_jpeg[1], np.uint8), cv2.IMREAD_COLOR)
        try:
            return await asyncio.to_thread(self.vlm_client.path_clear, image)
        except Exception:
            return None

    async def _wander(self):
        """Nothing found in a full turn: face a random new direction and drive a short random distance.

        The robot has no obstacle sensor, so the VLM is asked whether the floor ahead is clear before and every
        few seconds during the drive; anything but a clear answer stops it. Finding trash also stops it.
        """
        heading = random.uniform(60., 300.)
        distance = random.uniform(self.args.wander_min, self.args.wander_max)
        await self.grab_notice(f"No trash found in a full turn; turning {heading:.0f}° and driving up to "
                               f"{distance*100:.0f} cm to search somewhere else.")
        self.mode = "wander"
        turn = self.args.turn_speed
        end = time.monotonic() + heading / turn
        while self.auto and time.monotonic() < end and not self._target_fresh():
            await self.pi_send({"type": "drive", "enabled": True, "x": 0, "y": 0, "theta": turn * self.args.turn_sign})
            await asyncio.sleep(.1)
        await self.pi_send({"type": "stop"})
        await asyncio.sleep(.4)                            # let a frame of the new view arrive
        travelled, checked = 0., None
        while self.auto and travelled < distance and not self._target_fresh():
            if checked is None or time.monotonic() - checked >= self.args.wander_check_s:
                await self.pi_send({"type": "stop"})
                clear = await self._path_clear()
                checked = time.monotonic()
                if not clear:
                    await self.grab_notice(f"Wander stopped after {travelled*100:.0f} cm: path ahead "
                                           f"{'blocked' if clear is False else 'could not be checked'}.")
                    break
            await self.pi_send({"type": "drive", "enabled": True, "x": self.args.wander_speed * self.args.drive_sign,
                                "y": 0, "theta": 0})
            await asyncio.sleep(.1)
            travelled += self.args.wander_speed * .1
        await self.pi_send({"type": "stop"})
        if self._target_fresh():
            await self.grab_notice("Spotted trash while wandering; approaching it.")
        self.mode = "search"

    async def _escape_stall(self, reason):
        """Auto is stuck on a target it cannot approach: back up, turn away, and carry on searching."""
        await self.grab_notice(f"Auto stalled for {self.args.auto_stall_timeout:.0f}s ({reason}); "
                               f"backing up {self.args.grab_scan_backup*100:.0f} cm and turning away.")
        await self._back_up(self.args.grab_scan_backup)
        turn = self.args.turn_speed
        end = time.monotonic() + self.args.stall_turn_deg / turn
        while self.auto and time.monotonic() < end:
            await self.pi_send({"type": "drive", "enabled": True, "x": 0, "y": 0, "theta": turn * self.args.turn_sign})
            await asyncio.sleep(.1)
        await self.pi_send({"type": "stop"})
        self.auto_progress_t = time.monotonic()

    async def _reapproach(self, timeout):
        """Run the normal auto approach from the home pose; True if it finished centered on a target."""
        self._start_auto()
        deadline = time.monotonic() + timeout
        while self.auto:
            if time.monotonic() > deadline:
                self.auto = False
                self.mode = "manual"
                await self.pi_send({"type": "stop"})
                return False
            await asyncio.sleep(.1)
        return self.auto_complete

    async def _give_up(self, why, resume_auto):
        """End a grab sequence. In an auto run, skip this object for a while and keep searching instead of stopping."""
        if resume_auto and self.targeter:
            if self.latest_jpeg[1] is not None:
                self.targeter.give_up(cv2.imdecode(np.frombuffer(self.latest_jpeg[1], np.uint8), cv2.IMREAD_COLOR))
            await self.grab_notice(f"{why} Skipping this object for {self.targeter.s.reject_memory_s:.0f}s; auto keeps searching.")
            self._start_auto()
        else:
            await self.grab_notice(f"{why} Stopping.", error=True)

    async def grab_with_retries(self, resume_auto=False):
        """Grab; if the VLM says the gripper came up empty, back up, re-approach, and try again.

        After `grab_attempts` misses, back up further and scan for the next target (auto search/approach),
        which starts a fresh set of attempts, up to `grab_scan_cycles` times.
        """
        attempts = self.args.grab_attempts if self.vlm_client else 1
        try:
            for cycle in range(1, self.args.grab_scan_cycles + 1):
                for attempt in range(1, attempts + 1):
                    self.grab_attempt = attempt
                    result = await self.grab_routine()
                    if result == "placed" and resume_auto:
                        await self.grab_notice(f"Trash placed; backing up {self.args.place_backup*100:.0f} cm, "
                                               "then resuming auto mode to find the next target.")
                        await self._back_up(self.args.place_backup)
                        self._start_auto()
                    if result != "missed":
                        return
                    if attempt == attempts:
                        break
                    await self.grab_notice(f"Missed (attempt {attempt}/{attempts}). Backing up "
                                           f"{self.args.grab_retry_backup*100:.0f} cm and re-approaching.")
                    await self._back_up(self.args.grab_retry_backup)
                    if not await self._reapproach(self.args.grab_retry_timeout):
                        await self._give_up("Retry stopped: could not re-approach the target.", resume_auto)
                        return
                    await self.grab_notice(f"Re-approach complete; starting attempt {attempt + 1}/{attempts}.")
                    await asyncio.sleep(.4)
                if cycle == self.args.grab_scan_cycles:
                    await self._give_up(f"Grasp missed {attempts}x in each of {cycle} scan cycles.", resume_auto)
                    return
                await self.grab_notice(f"Missed {attempts}x. Backing up {self.args.grab_scan_backup*100:.0f} cm and scanning "
                                       f"for the next target (scan {cycle}/{self.args.grab_scan_cycles - 1}).")
                await self._back_up(self.args.grab_scan_backup)
                if not await self._reapproach(self.args.grab_scan_timeout):
                    await self._give_up("Scan found no target to approach.", resume_auto)
                    return
                await self.grab_notice("Scan locked a target and approached it; attempts reset.")
                await asyncio.sleep(.4)
        except asyncio.CancelledError:
            self.auto = False
            await self.pi_send({"type": "stop"})
            raise
        finally:
            self.grab_attempt = 0

    async def start_grab(self, resume_auto=False):
        """Start a grab (with retries). `resume_auto`: go back to auto search after the trash is placed."""
        if self.grab_task and not self.grab_task.done():
            await self.grab_notice("Grab already in progress.", error=True)
            return
        self.auto = False
        await self.grab_notice("Grab requested — checking target alignment and distance…")
        self.grab_task = asyncio.create_task(self.grab_with_retries(resume_auto))

    async def auto_loop(self):
        period = 1 / self.args.control_hz
        while True:
            if not self.auto:
                await asyncio.sleep(period)
                continue
            now = time.monotonic()
            stale = now - self.frame_seen >= self.args.lost_timeout
            searching = stale or now - self.target_seen >= self.args.lost_timeout
            holding = searching and not stale and now - self.target_seen < self.args.search_grace
            w, h = self.frame_size
            command = track(None if searching else self.target, w, h,
                            target_height_ratio=self.stop_height,
                            max_forward=self.args.speed * (self.auto_travel_speed / 10.0),
                            max_theta=self.args.turn_speed, min_theta=self.args.min_turn_speed,
                            wrist_gain=self.args.wrist_gain * (self.auto_arm_speed / 10.0),
                            # Wrist pitch is the sole approach-distance stop;
                            # target apparent size does not terminate approach.
                            ignore_distance=True,
                            steer_band=self.args.steer_band, vertical_band=self.args.vertical_band,
                            # Just lost the target: hold still so the VLM can re-find it in the same view.
                            patrol_theta=self.args.patrol_speed if searching and not holding else 0.0)
            self.mode = command.mode
            actual_turn = command.theta * self.args.turn_sign
            actual_forward = command.forward * self.args.drive_sign
            target_age = now - self.target_seen
            frame_age = now - self.frame_seen
            decision = {"frame_age_s": round(frame_age, 3), "target_age_s": round(target_age, 3),
                        "forward_m_s": round(actual_forward, 4), "turn_deg_s": round(actual_turn, 3),
                        "wrist_step_rad": round(command.wrist_delta, 5)}
            if searching:
                decision["reason"] = ("camera frame is stale; scanning for a target" if stale else
                                      f"target lost {target_age:.1f}s ago; holding still for the VLM to re-find it"
                                      if holding else f"no trash target seen for {target_age:.1f}s; scanning")
            elif self.target:
                cx, cy = self.target.center
                decision.update({"x_error": round((cx - w / 2) / (w / 2), 3),
                                 "y_error": round((cy - h / 2) / (h / 2), 3),
                                 "box_height_ratio": round((self.target.y2 - self.target.y1) / h, 3),
                                 "target_height_ratio": self.stop_height})
                decision["reason"] = {
                    "align": "target is off the center line; rotating before moving forward",
                    "steer": "target slightly off center; steering and slowing while approaching",
                    "approach": "target is centered and appears farther than the stop distance; approaching",
                    "arrived": "target is centered and at the requested apparent distance; holding",
                    "too_close": "target appears too close; holding position",
                }.get(command.mode, command.mode)
            else:
                decision["reason"] = "waiting through the lost-target delay; base stopped"
            if self.home:
                if searching and not holding:
                    self.wrist_delta *= .97      # drift back toward home over a couple of seconds, not snap
                self.wrist_delta = max(-self.args.wrist_range,
                                       min(self.args.wrist_range, self.wrist_delta + command.wrist_delta))
                joints = {name: float(self.home[name]) for name in JOINTS}
                joints["wrist_flex"] += self.wrist_delta * self.args.wrist_sign
                feedback_valid = (self.state.get('status') == 'connected'
                                  and now-self.state_seen <= 1.5
                                  and all(n in self.state for n in JOINTS))
                ik_wrist_angle = self._ik_ground_angle_deg(self._measured_arm_pose(joints)) if feedback_valid else 0.
                if not feedback_valid:
                    actual_forward = actual_turn = 0.
                    decision.update(forward_m_s=0., turn_deg_s=0., reason='Waiting for fresh measured arm pose')
                # Only trust the measured wrist angle once the arm is actually at the commanded pose;
                # otherwise a pose left over from before auto started can latch "arrived" immediately.
                arm_settled = feedback_valid and abs(float(self.state['wrist_flex']) - joints['wrist_flex']) <= .08
                decision['arm_settled'] = arm_settled
                if not searching and arm_settled and ik_wrist_angle >= self.auto_wrist_stop_deg:
                    self.auto_distance_reached = True
                if self.auto_distance_reached:
                    actual_forward = 0.0
                    decision['forward_m_s'] = 0.0
                if (not searching and feedback_valid and self.auto_distance_reached
                        and command.mode == "approach"):
                    actual_forward = 0.0
                    actual_turn = 0.0
                    decision["forward_m_s"] = 0.0
                    decision["turn_deg_s"] = 0.0
                    vertically_centered = abs(decision.get('y_error', 1.)) <= .06
                    decision['vertically_centered'] = vertically_centered
                    reason = (f'Target centered in both axes at {ik_wrist_angle:.1f}°; auto complete'
                              if vertically_centered else
                              f'Wrist threshold reached at {ik_wrist_angle:.1f}°; base stopped, centering target vertically')
                    command = command.__class__(forward=0.0, theta=command.theta,
                                                wrist_delta=command.wrist_delta, mode="wrist-angle")
                    decision["reason"] = reason
                    # The approach phase is complete. Leave the arm at this
                    # pose, stop the base, and return control to manual mode;
                    # with --auto-grab the grab starts right after this tick.
                    if vertically_centered:
                        self.auto = False
                        self.auto_complete = True
                        self.mode = "manual"
                        # Hold the measured pose that produced the centered
                        # image instead of issuing one more wrist correction.
                        joints = self._measured_arm_pose(joints)
                        await self.pi_send({"type": "stop"})
                    else:
                        self.mode = 'center-vertical'
                decision["wrist_pitch_deg"] = round(ik_wrist_angle, 1)
                decision["wrist_stop_deg"] = self.auto_wrist_stop_deg
            if self.targeter and not searching and self.vlm and self.vlm.overfills:
                # Wrist tilting cannot un-clip something bigger than the view: creep backwards until it fits.
                actual_forward, actual_turn = -self.args.overfill_backup_speed * self.args.drive_sign, 0.0
                decision.update(forward_m_s=round(actual_forward, 4), turn_deg_s=0.,
                                reason=f"target fills the view ({self.vlm.reason}); backing up")
            elif self.targeter and not searching and self.vlm and self.vlm.status != "DRIVE" and actual_forward:
                # Turning and wrist recentering continue; forward travel waits for a fresh, in-view VLM check.
                actual_forward = 0.0
                decision["forward_m_s"] = 0.0
                decision["reason"] = f"VLM hold ({self.vlm.reason}); " + decision["reason"]
            # Driving forward always moves the target down the image; tilt the wrist down in proportion to the
            # actual forward speed (after any HOLD zeroed it) so the error correction only handles what is left.
            feedforward = self.args.wrist_feedforward * abs(actual_forward) * period
            if self.home and feedforward and not searching:
                self.wrist_delta = min(self.args.wrist_range, self.wrist_delta + feedforward)
                joints["wrist_flex"] += feedforward * self.args.wrist_sign
            joints = self._slew_auto_joints(joints, period)
            self._log_auto_tick(command, decision, joints)
            await asyncio.gather(
                self.pi_send({"type": "joints", "enabled": True, "values": joints}),
                self.pi_send({"type": "drive", "enabled": True, "x": actual_forward,
                              "y": 0, "theta": actual_turn}))
            await self.send_autonomy_status(decision)
            reason = decision.get("reason", "")
            if reason.split(";")[0][:60] != self.logged_reason:
                self.logged_reason = reason.split(";")[0][:60]
                logging.info("auto: %s (fwd %.3f, turn %.1f)", reason, actual_forward, actual_turn)
            # A full turn without finding anything: go somewhere else and search again.
            tick = time.monotonic()
            if searching and self.auto and self.args.wander_after_deg > 0:
                if self.search_tick is not None:
                    self.search_turn_deg += abs(actual_turn) * (tick - self.search_tick)
                self.search_tick = tick
                if self.search_turn_deg >= self.args.wander_after_deg:
                    self.search_turn_deg, self.search_tick = 0., None
                    await self._wander()
                    self.auto_progress_t = time.monotonic()
                    continue
            else:
                self.search_turn_deg, self.search_tick = 0., None
            # Stall watchdog: a target is held but neither the base nor the wrist is moving.
            moving = searching or not self.auto or abs(actual_forward) > 1e-4 or abs(actual_turn) > 1e-3 \
                or abs(command.wrist_delta) > 1e-4
            if moving:
                self.auto_progress_t = time.monotonic()
            elif time.monotonic() - self.auto_progress_t > self.args.auto_stall_timeout:
                await self._escape_stall(decision.get("reason", ""))
                continue
            # Approach finished on its own: grab now. Skip when a grab retry is already running --
            # its own re-approaches finish the same way and it grabs next itself.
            if (self.auto_complete and not self.auto and self.args.auto_grab
                    and not (self.grab_task and not self.grab_task.done())):
                self.auto_complete = False
                await self.grab_notice("Auto approach complete; starting grab.")
                await self.start_grab(resume_auto=True)
            await asyncio.sleep(period)

    async def browser_message(self, data):
        kind = data.get("type")
        if (kind in {'auto_mode', 'test_grab', 'joints', 'home', 'drive'}
                and self.motion_task and not self.motion_task.done()):
            await self.grab_notice('Stop the current pose/replay move before issuing another motion.', error=True)
            return
        if kind.startswith('placement_'):
            if self.auto or self.grab_active or self.motion_recording or (self.motion_task and not self.motion_task.done()):
                await self.grab_notice('Stop automatic motion/recording before editing placement keypoints.', error=True)
                return
            if kind == 'placement_goto':
                point = next((p for p in self.placement_keypoints if p['id']==data.get('id')), None)
                if point is None: return
                self.motion_task = asyncio.create_task(self.goto_keypoint(json.loads(json.dumps(point))))
                return
            if kind in {'placement_capture', 'placement_update'}:
                if self.state.get('status') != 'connected' or time.monotonic()-self.state_seen > 1.5:
                    await self.grab_notice('Cannot capture: fresh servo measurements required.', error=True)
                    return
                pose = {n: float(self.state[n]) for n in JOINTS}
                if not all(np.isfinite(v) for v in pose.values()):
                    return
                if kind == 'placement_update':
                    point = next((p for p in self.placement_keypoints if p['id']==data.get('id')), None)
                    if point is None: return
                    point.update(joints=pose, xyz=self.wrist_ik.fk(pose)[:3, 3].tolist())
                    await self.grab_notice(f"Updated keypoint: {point['name']}")
                else:
                    self.placement_keypoints.append({'id': uuid.uuid4().hex,
                        'name': str(data.get('name') or 'Keypoint')[:80], 'joints': pose,
                        'xyz': self.wrist_ik.fk(pose)[:3, 3].tolist()})
            elif kind in {'placement_delete', 'placement_move'}:
                index = next((i for i,p in enumerate(self.placement_keypoints) if p['id']==data.get('id')), None)
                if index is None: return
                if kind == 'placement_delete': self.placement_keypoints.pop(index)
                else:
                    dest = max(0, min(len(self.placement_keypoints)-1, index + (1 if data.get('direction')=='down' else -1)))
                    self.placement_keypoints.insert(dest, self.placement_keypoints.pop(index))
            elif kind == 'placement_teach':
                await self.pi_send({'type':'stop'})
                if not data.get('enabled'):
                    if self.state.get('status') != 'connected' or time.monotonic()-self.state_seen > 1.5:
                        await self.grab_notice('Fresh servo readings required to hold the current pose.', error=True)
                        return
                    await self.pi_send({'type':'joints','enabled':True,'values':{n:float(self.state[n]) for n in JOINTS}})
                await self.pi_send({'type':'arm_torque','enabled':not bool(data.get('enabled'))})
                await self.grab_notice('Arm released for hand positioning.' if data.get('enabled') else 'Holding current arm pose.')
                return
            else: return
            self.keypoints_path.write_text(json.dumps(self.placement_keypoints, indent=2))
            await self.send_autonomy_status()
            return
        if kind == "start_motion_record":
            name = "".join(ch for ch in str(data.get("name", "put-away")) if ch.isalnum() or ch in "-_ ").strip()[:64]
            if not name:
                await self.grab_notice("Give the demonstration a name first.", error=True)
                return
            if self.motion_task and not self.motion_task.done():
                self.motion_task.cancel()
            self.auto = False
            await self.pi_send({"type": "stop"})
            await self.pi_send({"type": "arm_torque", "enabled": False})
            self.motion_recording = True
            self.motion_name = name
            self.motion_started = time.monotonic()
            self.motion_samples = []
            await self.grab_notice(f"Recording '{name}'. Move the arm through the put-away motion, then press Stop recording.")
            await self.send_autonomy_status()
            return
        if kind == "stop_motion_record":
            if not self.motion_recording:
                await self.grab_notice("No motion recording is active.", error=True)
                return
            self.motion_recording = False
            self.mode = "manual"
            if all(name in self.state for name in JOINTS):
                await self.pi_send({"type": "joints", "enabled": True,
                                    "values": {name: float(self.state[name]) for name in JOINTS}})
            await self.pi_send({"type": "arm_torque", "enabled": True})
            name = self.motion_name or "put-away"
            path = self.motion_dir / (name + ".json")
            payload = {"name": name, "created": time.time(), "joints": JOINTS,
                       "duration": self.motion_samples[-1]["t"] if self.motion_samples else 0,
                       "samples": self.motion_samples}
            path.write_text(json.dumps(payload, indent=2) + "\n")
            await self.grab_notice(f"Saved '{name}' with {len(self.motion_samples)} measured poses.")
            await self.send_autonomy_status()
            return
        if kind == "replay_motion":
            if self.motion_recording:
                await self.grab_notice("Stop recording before replaying a motion.", error=True)
                return
            if self.motion_task and not self.motion_task.done():
                self.motion_task.cancel()
            self.motion_task = asyncio.create_task(self.replay_motion(str(data.get("name", ""))))
            return
        if kind == "delete_motion":
            name = Path(str(data.get("name", ""))).stem
            path = self.motion_dir / (name + ".json")
            if not name or not path.exists() or path.parent != self.motion_dir:
                await self.grab_notice("That recording does not exist.", error=True)
                return
            path.unlink()
            await self.grab_notice(f"Deleted recording '{name}'.")
            await self.send_autonomy_status()
            return
        if kind in {"auto_travel_speed", "auto_arm_speed", "auto_wrist_stop", "ground_offset", "stop_height", "grab_speed"}:
            try:
                value = float(data.get("value"))
                if not np.isfinite(value):
                    raise ValueError
                if kind == "auto_travel_speed":
                    if not .1 <= value <= 100.: raise ValueError
                    self.auto_travel_speed = value
                    message = f"Auto travel speed set to {value:.1f}."
                elif kind == "auto_arm_speed":
                    if not .1 <= value <= 100.: raise ValueError
                    self.auto_arm_speed = value
                    message = f"Auto arm speed set to {value:.1f}."
                elif kind == "auto_wrist_stop":
                    if not 0. <= value <= 90.: raise ValueError
                    self.auto_wrist_stop_deg = value
                    message = f"Auto wrist pitch stop set to {value:.1f}°."
                elif kind == "stop_height":
                    if not .20 <= value <= .95: raise ValueError
                    self.stop_height = value
                    message = f"Auto target close threshold set to {value:.0%}."
                elif kind == "grab_speed":
                    if not .1 <= value <= 100.: raise ValueError
                    self.grab_speed = value
                    message = f"Grab dive speed set to {value:.1f}."
                else:
                    if not -100. <= value <= 100.: raise ValueError
                    self.ground_offset_cm = value
                    message = f"Ground offset set to {value:.1f} cm."
                await self.grab_notice(message)
                await self.send_autonomy_status()
            except (TypeError, ValueError):
                await self.grab_notice("Invalid auto setting.", error=True)
            return
        if kind == "grab_ik_tolerance":
            try:
                value = float(data.get("value"))
                if not np.isfinite(value) or not .001 <= value <= .010:
                    raise ValueError("Expected 1–10 mm")
                self.grab_ik_tolerance = value
                await self.grab_notice(f"Allowed IK position error: {value*1000:.1f} mm. Higher allows more ray drift; joint and orientation limits remain enforced.")
                await self.send_autonomy_status()
            except (TypeError, ValueError):
                await self.grab_notice("Invalid IK tolerance: use 1–10 mm.", error=True)
            return
        if kind == "contact_threshold":
            try:
                value = float(data.get("value"))
                if not np.isfinite(value) or not 8. <= value <= 400.:
                    raise ValueError("Expected 8-400 load units")
                self.contact_threshold = value
                await self.grab_notice(f"Contact load threshold: {value:.0f} units on elbow_flex. Takes effect immediately, including mid-dive.")
                await self.send_autonomy_status()
            except (TypeError, ValueError):
                await self.grab_notice("Invalid contact threshold: use 8-400 load units.", error=True)
            return
        if kind == "test_grab":
            await self.start_grab()
            return
        if kind == "stop_height":
            try:
                value = float(data.get("value"))
                if not np.isfinite(value) or not .20 <= value <= .95:
                    raise ValueError
                self.stop_height = value
                await self.grab_notice(f"Auto approach stop height set to {value:.0%} of camera height; higher stops closer.")
                await self.send_autonomy_status()
            except (TypeError, ValueError):
                await self.grab_notice("Invalid auto stop height: use 20–95%.", error=True)
            return
        if kind == "grab_threshold":
            try:
                self.grab_target_width = max(.10, min(.95, float(data.get("value"))))
                await self.grab_notice(f"Grab threshold set to {self.grab_target_width:.0%} of camera width.")
            except (TypeError, ValueError):
                await self.grab_notice("Invalid grab threshold.", error=True)
            return
        if kind in ("stop", "auto_mode") and not self.grab_active and self.grab_task and not self.grab_task.done():
            # A grab retry is between attempts (backing up / re-approaching): the user's STOP or auto toggle ends it.
            self.grab_task.cancel()
        if self.grab_active:
            if kind == "stop":
                if self.grab_task and not self.grab_task.done():
                    self.grab_task.cancel()
                await self.pi_send({"type": "stop"})
            return
        if kind == "stop" and self.motion_task and not self.motion_task.done():
            self.motion_task.cancel()
            await self.pi_send({"type": "stop"})
            return
        if kind == "auto_mode":
            self.auto_distance_reached = False
            self.auto = bool(data.get("enabled"))
            if self.auto:
                # Start every run from the home pose; a wrist offset left from the previous run
                # would begin already tilted to the stop angle and report "arrived" at once.
                self._start_auto()
            self.mode = "search" if self.auto else "manual"
            if not self.auto:
                await self.pi_send({"type": "stop"})
            await self.send_autonomy_status()
            return
        if self.auto and kind in {"drive", "joints", "home", "set_home"}:
            return
        await self.pi_send(data)

    async def websocket(self, request):
        ws = web.WebSocketResponse(max_msg_size=4_000_000)
        await ws.prepare(request)
        self.browsers.add(ws)
        await ws.send_json({"type": "hello", "joints": JOINTS, "state": self.state,
                            "has_home": self.home is not None})
        await self.send_autonomy_status()
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await self.browser_message(json.loads(message.data))
        finally:
            self.browsers.discard(ws)
            if not self.auto:
                await self.pi_send({"type": "stop"})
        return ws


async def start(args):
    hub = ControlHub(args)
    app = web.Application()
    app.router.add_get("/ws", hub.websocket)
    web_root = Path(__file__).resolve().parents[1] / "web"
    app.router.add_get("/", lambda request: web.FileResponse(web_root / "index.html"))
    app.router.add_get("/diagnostics", lambda request: web.FileResponse(web_root / "diagnostics.html"))
    app.router.add_get("/load-probe", lambda request: web.FileResponse(web_root / "load-probe.html"))
    app.router.add_static("/probe-data", Path(__file__).resolve().parents[1] / "screenshots")
    app.router.add_static("/", web_root)
    async def startup(app):
        app["pi_task"] = asyncio.create_task(hub.pi_loop())
        app["auto_task"] = asyncio.create_task(hub.auto_loop())
        app["camera_task"] = asyncio.create_task(hub.camera_worker())
    app.on_startup.append(startup)
    async def cleanup(app):
        await hub.pi_send({"type": "stop"})
        if hub.grab_task and not hub.grab_task.done():
            hub.grab_task.cancel()
        for key in ("pi_task", "auto_task", "camera_task"):
            app[key].cancel()
        await asyncio.gather(app["pi_task"], app["auto_task"], app["camera_task"], return_exceptions=True)
    app.on_cleanup.append(cleanup)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    logging.info("control page: http://%s:%s", args.host, args.port)
    await asyncio.Event().wait()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pi", default="ws://raspberrypi.local:8765")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--targeter", choices=("yolo", "vlm"), default="yolo",
                        help="vlm: Qwen3-VL finds/verifies trash, CSRT tracks between answers")
    parser.add_argument("--vlm-backend", choices=("llama", "hf"), default="llama",
                        help="llama: llama-server at --vlm-url (scripts/start-vlm-server.ps1); "
                             "hf: in-process transformers (needs .venv-targeting, ~7x slower on Windows)")
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8091")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct", help="hf backend only")
    parser.add_argument("--grab-attempts", type=int, default=3,
                        help="With --targeter vlm: lift slightly, ask the VLM whether the grasp held, and retry up to this many times")
    parser.add_argument("--grab-servo-gain", type=float, default=.5,
                        help="During the dive, fraction of the target's angular offset corrected per camera frame")
    parser.add_argument("--grab-elevation-weight", type=float, default=.5,
                        help="How strongly the dive pitches the camera to point along the (steered) path; 0 = off")
    parser.add_argument("--camera-hfov", type=float, default=90., help="Wrist camera horizontal field of view (deg)")
    parser.add_argument("--camera-vfov", type=float, default=60., help="Wrist camera vertical field of view (deg)")
    parser.add_argument("--grab-lift-speed", type=float, default=.06,
                        help="Average m/s for the smooth lift to the grasp-check height")
    parser.add_argument("--grasp-empty-gripper", type=float, default=-.80,
                        help="Gripper position (rad) where an empty close stops (measured -0.78 to -0.81)")
    parser.add_argument("--grasp-gripper-margin", type=float, default=.08,
                        help="Jaws stopping this much more open than an empty close count as holding something "
                             "(empty closes stop within about 0.03 rad of each other)")
    parser.add_argument("--grasp-parallax-min", type=float, default=.6,
                        help="Top/half-height box area ratio at or above which the trash rose with the gripper (held)")
    parser.add_argument("--grab-verify-height", type=float, default=.12,
                        help="Gripper height above the floor plane (m) for the grasp check")
    parser.add_argument("--grab-retry-backup", type=float, default=.08, help="Metres to back up before re-approaching")
    parser.add_argument("--grab-retry-timeout", type=float, default=30., help="Seconds allowed for each re-approach")
    parser.add_argument("--auto-grab", action=argparse.BooleanOptionalAction, default=True,
                        help="Start a grab as soon as an auto approach completes (--no-auto-grab to only approach)")
    parser.add_argument("--place-speed", type=float, default=2.,
                        help="Speed factor for moves between placement keypoints (except --place-slow-keypoints)")
    parser.add_argument("--place-slow-keypoints", type=lambda text: {n.strip().lower() for n in text.split(',') if n.strip()},
                        default={"release", "out"}, help="Keypoints moved to at the taught 1x speed, comma separated")
    parser.add_argument("--auto-stall-timeout", type=float, default=6.,
                        help="Seconds auto may hold a target without moving base or wrist before backing off")
    parser.add_argument("--stall-turn-deg", type=float, default=45., help="Degrees to turn away after a stall")
    parser.add_argument("--overfill-backup-speed", type=float, default=.03,
                        help="m/s to creep backwards while the target fills the view")
    parser.add_argument("--place-backup", type=float, default=.10,
                        help="Metres to back up after an auto grab places trash, before auto resumes")
    parser.add_argument("--wander-after-deg", type=float, default=360.,
                        help="Search rotation without a target before driving somewhere else (0 disables)")
    parser.add_argument("--wander-min", type=float, default=.3, help="Shortest wander drive (m)")
    parser.add_argument("--wander-max", type=float, default=.8, help="Longest wander drive (m)")
    parser.add_argument("--wander-speed", type=float, default=.06, help="Wander drive speed (m/s)")
    parser.add_argument("--wander-check-s", type=float, default=1.5, help="Seconds between path-clear checks while wandering")
    parser.add_argument("--steer-band", type=float, default=.25,
                        help="Keep driving (steering, slowing) while the target is within this horizontal error; 0 = stop and turn")
    parser.add_argument("--wrist-feedforward", type=float, default=2.,
                        help="Wrist tilt (rad) per metre driven toward the target, so pitch keeps up during the approach")
    parser.add_argument("--search-grace", type=float, default=2.5,
                        help="Seconds to hold still (wrist and base) after losing a target before searching")
    parser.add_argument("--auto-joint-speed", type=float, default=.8,
                        help="Max rad/s per joint for arm moves commanded by auto (smooths the return to home)")
    parser.add_argument("--vertical-band", type=float, default=.35,
                        help="Slow the approach as the target sits off-centre vertically; 0 = off")
    parser.add_argument("--grab-scan-cycles", type=int, default=3,
                        help="Sets of attempts; after each set misses, back up and scan for the next target")
    parser.add_argument("--grab-scan-backup", type=float, default=.15, help="Metres to back up before scanning")
    parser.add_argument("--grab-scan-timeout", type=float, default=60., help="Seconds allowed to scan and approach")
    parser.add_argument("--grab-hold-time", type=float, default=.25,
                        help="Seconds to keep still after the gripper has stopped closing, before lifting")
    parser.add_argument("--vlm-verify-max-age", type=float, default=2.,
                        help="Seconds a VLM verification stays valid for forward travel")
    parser.add_argument("--confidence", type=float, default=.004,
                        help="Candidate threshold; multi-frame confirmation limits low-confidence false positives")
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--confirm-iou", type=float, default=.35)
    parser.add_argument("--min-target-area", type=float, default=.002)
    parser.add_argument("--max-target-area", type=float, default=.35,
                        help="Reject giant background regions such as rugs/floors")
    parser.add_argument("--target-border-margin", type=float, default=.015,
                        help="Reject partial detections clipped against a camera edge")
    parser.add_argument("--stop-height", type=float, default=.60,
                        help="Auto approach stop distance as target box height ratio; higher stops closer")
    parser.add_argument("--speed", type=float, default=.16,
                        help="Auto base approach speed in m/s")
    parser.add_argument("--turn-speed", type=float, default=12.)
    parser.add_argument("--min-turn-speed", type=float, default=6.,
                        help="Minimum correction outside center deadband; clears wheel-servo deadband")
    parser.add_argument("--patrol-speed", type=float, default=8.)
    parser.add_argument("--lost-timeout", type=float, default=.7)
    parser.add_argument("--control-hz", type=float, default=12.)
    parser.add_argument("--wrist-gain", type=float, default=.025)
    parser.add_argument("--wrist-range", type=float, default=1.0,
                        help="Maximum auto wrist tracking offset in radians; bounded by the URDF wrist-flex limit")
    parser.add_argument("--turn-sign", type=float, choices=(-1., 1.), default=-1.,
                        help="LeKiwi camera/base convention: invert image error into base rotation")
    parser.add_argument("--drive-sign", type=float, choices=(-1., 1.), default=-1.,
                        help="LeKiwi wheel convention: physical forward is negative body x")
    parser.add_argument("--wrist-sign", type=float, choices=(-1., 1.), default=1.)
    parser.add_argument("--grab-center-tolerance", type=float, default=.10)
    parser.add_argument("--grab-align-time", type=float, default=1.8)
    parser.add_argument("--grab-align-speed", type=float, default=8.)
    parser.add_argument("--grab-min-height", type=float, default=.18)
    parser.add_argument("--grab-forward", type=float, default=.04,
                        help="Bounded body-forward speed during the IK dive")
    parser.add_argument("--grab-probe-time", type=float, default=.55)
    parser.add_argument("--grab-dive-time", type=float, default=5.0,
                        help="Maximum IK dive time while increasing apparent target width")
    parser.add_argument("--grab-grip-time", type=float, default=.55)
    parser.add_argument("--grab-lift-time", type=float, default=.65)
    parser.add_argument("--grab-return-time", type=float, default=.8)
    parser.add_argument("--grab-load-delta", type=float, default=80.)
    # On this LeKiwi calibration, more-negative shoulder and more-positive
    # elbow extend the wrist toward the floor/forward; the lift reverses it.
    parser.add_argument("--grab-shoulder", type=float, default=-.18)
    parser.add_argument("--grab-elbow", type=float, default=.25)
    parser.add_argument("--grab-wrist", type=float, default=.02)
    parser.add_argument("--grab-lift-shoulder", type=float, default=.12)
    parser.add_argument("--grab-lift-elbow", type=float, default=-.17)
    parser.add_argument("--grab-lift-wrist", type=float, default=-.02)
    parser.add_argument("--grab-gripper", type=float, default=-1.2, help="Gripper close endpoint; -1.2 is the fully-closed limit of the Pi gripper clamp, so it clamps hard on whatever it reached")
    parser.add_argument("--grab-open", type=float, default=1.2, help="Configured open endpoint within Pi gripper clamp; not an uncalibrated mechanical-limit seek")
    parser.add_argument("--grab-pan-sign", type=float, choices=(-1., 1.), default=1.,
                        help="Image-x to dive-yaw sign; the dive's self-check flipped -1 in every dive on 2026-09-26")
    parser.add_argument("--grab-max-travel", type=float, default=.25,
                        help="Maximum dive travel in metres; reaching it closes the gripper rather than failing")
    parser.add_argument("--grab-approach-step", type=float, default=.0015,
                        help="Metres advanced along the approach vector per step (was .002; larger = faster dive)")
    parser.add_argument("--grab-approach-period", type=float, default=.10,
                        help="Seconds between approach steps (was .12; smaller = faster dive)")
    parser.add_argument("--grab-joint-slew", type=float, default=.015,
                        help="Maximum radians each non-gripper joint may change per IK approach step")
    parser.add_argument("--grab-ground-z", type=float, default=-0.057,
                        help="Ground plane height in the URDF base frame (m). Robot sits on a 2.5in pedestal; the standoff was reduced 10%% from 0.0635 to 0.057 m so the approach stops a touch higher. The dive stops here or on contact.")
    parser.add_argument("--grab-max-reaim", type=float, default=float(np.deg2rad(20)),
                        help="Radians the dive ray may be steered away from the captured ray by target centring")
    parser.add_argument("--grab-track-hold", type=float, default=1.5,
                        help="Seconds to retain camera target geometry through missed detector frames during a grab")
    parser.add_argument("--pi-idle-timeout", type=float, default=5.,
                        help="Reconnect if the Pi sends nothing for this long; state normally arrives every .2 s")
    parser.add_argument("--grab-contact-load", type=float, default=DEFAULT_THRESHOLD, help="Starting wrist_flex load threshold that stops the dive; the control page slider overrides it live.")
    parser.add_argument("--grab-target-width", type=float, default=.95,
                        help="Stop IK dive once the locked target occupies this camera-width fraction")
    parser.add_argument("--grab-ik-step", type=float, default=.04)
    parser.add_argument("--grab-ik-max", type=float, default=2.0)
    parser.add_argument("--grab-pan-gain", type=float, default=.28)
    parser.add_argument("--grab-pan-limit", type=float, default=.35)
    parser.add_argument("--grab-vertical-gain", type=float, default=.10)
    parser.add_argument("--grab-wrist-vertical-gain", type=float, default=.05)
    parser.add_argument("--grab-wrist-back", type=float, default=.12)
    parser.add_argument("--grab-retry-time", type=float, default=.45)
    parser.add_argument("--grab-link1", type=float, default=.16)
    parser.add_argument("--grab-link2", type=float, default=.16)
    parser.add_argument("--grab-link3", type=float, default=.10)
    parser.add_argument("--grab-cartesian-forward", type=float, default=.18)
    parser.add_argument("--grab-cartesian-down", type=float, default=.12)
    parser.add_argument("--grab-joint-delta-limit", type=float, default=.45)
    parser.add_argument("--grab-verify-motion", type=float, default=.06,
                        help="Minimum normalized image motion after lift to call the grasp retained")
    parser.add_argument("--grab-verify-width", type=float, default=.05)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(start(parse_args()))
