# RoboPet LeKiwi control panel

This folder contains a local browser UI and a Raspberry Pi WebSocket bridge for the LeKiwi.

The bridge controls six arm joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) and the three-wheel base. Arm sliders command immediately. Base direction buttons command while held and stop on release; the Pi watchdog also stops stale base commands automatically.

The same WebSocket carries JSON motor telemetry and binary MJPEG camera frames from `/dev/video0`.

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
