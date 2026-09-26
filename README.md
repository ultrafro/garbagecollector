# RoboPet LeKiwi control panel

This folder contains a local browser UI and a Raspberry Pi WebSocket bridge for the LeKiwi.

The bridge controls six arm joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) and the three-wheel base. Arm sliders command immediately. Base direction buttons command while held and stop on release; the Pi watchdog also stops stale base commands automatically.

The control page also has an explicit `test grab` button and a configurable grab-width slider (default 95%). It only runs when a confirmed trash target is close: the hub centers it with arm IK, snapshots the measured six-joint pose, then advances along the captured forward approach vector (which points forward and down) until the gripper reaches the ground plane (`--grab-ground-z`, default -0.057 m — the 2.5in pedestal standoff reduced 10%), wrist_flex loads, or the arm runs out of reach. It steers the vector toward the target while it is visible and holds the last vector when it is not; orientation floats so the wrist can tilt to reach. The base is held stopped throughout grab. The dive stops when a 3-sample moving average of the wrist_flex servo load magnitude holds at or above a threshold (default 64), adjustable live from the control page slider (see `autonomy/contact.py`). The average smooths isolated movement-torque spikes; the threshold was calibrated from a real ground-contact dive. That load check is the only thing that ends the dive early; lag, joint tracking error, an exhausted ray and the travel limit all simply end the dive where it is and close the gripper rather than abandoning the grab. Then the gripper fully closes (`--grab-gripper`), the arm lifts, and it returns to the saved home pose holding the grip closed. The hub checks that the target moves with the wrist; if it does not, it retracts and retries the grip once before returning home. The Pi reports optional STS3215 `present_load`/`present_current` telemetry; when firmware does not return those registers, image motion and position error are used for verification.

The same WebSocket carries JSON motor telemetry and binary MJPEG camera frames from `/dev/video0`.

The UI can save the current measured arm pose as home and return to it later. The pose is persisted on the Pi at `~/robopet/home_pose.json`.

On connection, the six arm sliders initialize from the measured motor positions.

Arm positions are radians, matching the `rustypot` API. The UI and bridge clamp targets to conservative ranges.

## Run the local page

```powershell
python -m http.server 8080 --directory .\robopet\web
```

Open <http://localhost:8080>. The page connects to `ws://raspberrypi.local:8765` by default.

## Pi deployment

Copy `pi/lekiwi_ws.py` to the Pi alongside the installed LeRobot package and run:

```bash
python3 lekiwi_ws.py --host 0.0.0.0 --port 8765
```

The Pi needs Python packages `websockets`, `numpy`, and `rustypot`. Set `LEKIWI_PORT` if the motor bus is not `/dev/ttyACM0`.

For a local page, run `.\run-local.ps1` from PowerShell and open `http://localhost:8080`.

This is a test console for a real robot. Keep the arm clear, start with low speed, and have a physical power disconnect available.

## Computer-side trash homing

The autonomy loop runs on the controlling computer; the Pi remains a camera and
motor WebSocket bridge. First use the web console to save a safe home pose. Then,
on the computer:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-server.txt
.\run-local.ps1
```

Open <http://localhost:8080>. The computer-side hub runs YOLO continuously, so
annotated boxes remain visible during manual driving. Use the joystick for
forward/strafe, hold the rotation buttons to turn, and use the sliders for the
arm. `Start auto mode` transfers base and arm ownership to trash homing. Its
rolling decision trace reports target confidence, centering and distance errors,
exact motion commands, and the reason for every behavior.

The default YOLO-World model is downloaded on first use and uses open-vocabulary
prompts for wrappers, discarded packaging, litter, bottles, cups, cans, bags,
and food containers. Low-confidence candidates must overlap for three
successive frames before they become motion targets. Override the vocabulary with
`--labels ...`; a custom trash-trained checkpoint remains the best production
option once representative camera images have been collected.

Candidates covering more than 35% of the image are rejected by default so rugs,
floors, and other giant background regions cannot become targets. Tune this with
`--max-target-area` if necessary. Candidates clipped against the frame edge are
also rejected because they cannot be centered reliably.

The largest qualifying bounding box is selected as the closest object. The robot
turns until it is horizontally centered, then approaches until the box is 50% of
the image height (`--stop-height`). The wrist tracks vertical error relative to
the saved home pose. If frames/targets are lost, forward motion stops and the base
slowly scans. This LeKiwi uses an inverted image-to-base turn convention by
default; use `--turn-sign 1` only if another build rotates oppositely. Use
`--drive-sign 1` only if physical forward is positive body-x on another build,
and `--wrist-sign -1` if its vertical axis is reversed. The default scan rate is 8 degrees/s because lower rotation commands
can fall inside the wheel-servo deadband; tune it with `--patrol-speed`. Stop the
autonomy process before using manual controls.

Performance defaults are a 0.004 confidence floor with three-frame confirmation,
0.11 m/s maximum approach speed, 12 degrees/s maximum turning, and a 12 Hz
control loop. Keep a clear path and physical power cutoff available.
