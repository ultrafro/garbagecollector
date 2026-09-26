"""Record raw camera frames and telemetry during a bounded base-only sweep.

Default is stationary. --move uses two small opposing turns, no forward/grab.
--distance performs a forward traverse; use only after checking the entire path.
Always stops on completion, stale camera/telemetry, disconnect, or exception.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np
import websockets


async def capture(args):
    folder = Path('recordings') / ('targeting-' + time.strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True, exist_ok=False)
    events = []
    frames = []
    latest_frame = 0.0
    latest_state = 0.0
    hardware = False
    measured = {}
    encoder_distance = 0.0
    commanded_distance = 0.0
    traversing = False
    stopped_at = None
    wheel_matrix = np.column_stack((-np.sin(np.deg2rad([60.,180.,300.])), np.cos(np.deg2rad([60.,180.,300.])), np.full(3,.125)))
    start = time.monotonic()

    async with websockets.connect(args.hub, max_size=4_000_000) as hub, websockets.connect(args.pi, max_size=4_000_000) as pi:
        async def send(message):
            events.append({'t': time.monotonic()-start, 'command': message})
            await hub.send(json.dumps(message))

        async def receive_raw():
            nonlocal latest_frame, latest_state, hardware, measured, encoder_distance
            async for raw in pi:
                now = time.monotonic()
                if isinstance(raw, bytes):
                    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    latest_frame = now
                    name = f'raw-{len(frames):05d}.jpg'
                    (folder/name).write_bytes(raw)
                    frames.append({'t': now-start, 'file': name})
                else:
                    message = json.loads(raw)
                    events.append({'t': now-start, 'pi': message})
                    if message.get('type') == 'state':
                        state = message.get('data', {})
                        if traversing and latest_state and 'wheel_speed' in state:
                            body = np.linalg.solve(wheel_matrix, np.asarray(state['wheel_speed'])*.05)
                            encoder_distance += max(0., -float(body[0]))*min(.4, now-latest_state)
                        latest_state = now
                        hardware = bool(message.get('data', {}).get('hardware'))
                        measured = message.get('data', {})

        async def receive_hub():
            annotated_index = 0
            async for raw in hub:
                if isinstance(raw, bytes):
                    (folder/f'annotated-{annotated_index:05d}.jpg').write_bytes(raw)
                    annotated_index += 1
                else:
                    events.append({'t': time.monotonic()-start, 'hub': json.loads(raw)})

        tasks = [asyncio.create_task(receive_raw()), asyncio.create_task(receive_hub())]
        try:
            await send({'type': 'auto_mode', 'enabled': False})
            await send({'type': 'stop'})
            deadline = time.monotonic()+15
            while not (latest_frame and latest_state):
                if time.monotonic() > deadline:
                    raise RuntimeError('No fresh camera/telemetry; no movement attempted')
                await asyncio.sleep(.1)
            print(json.dumps({'folder': str(folder), 'hardware': hardware}), flush=True)
            if (args.move or args.wrist_offset or args.distance) and not hardware:
                raise RuntimeError('Hardware not confirmed; refusing motion')
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
            initial_pose = {n: float(measured[n]) for n in joint_names}
            motion_start = time.monotonic()
            previous_tick = motion_start
            previous_speed = 0.
            while time.monotonic()-motion_start < args.seconds:
                now = time.monotonic()
                elapsed = now-motion_start
                commanded_distance += previous_speed * (now-previous_tick)
                previous_tick = now
                if now-latest_frame > 1.0 or now-latest_state > 1.0 or any(t.done() for t in tasks):
                    raise RuntimeError('Stream stale or disconnected; stopping')
                # 1.5 s clockwise, pause, 3 s anticlockwise, pause, 1.5 s clockwise.
                # Nominal +/-9 degrees, ending at original heading. No translation.
                theta = 0.0
                forward = 0.0
                if args.move:
                    if 2 <= elapsed < 3.5 or 9 <= elapsed < 10.5:
                        theta = 6.0
                    elif 5 <= elapsed < 8:
                        theta = -6.0
                if args.distance and elapsed >= 2:
                    if stopped_at is None and max(commanded_distance, encoder_distance) >= args.distance:
                        stopped_at = now
                        print(json.dumps({'traverse_stop': True, 'commanded_m': commanded_distance, 'wheel_estimate_m': encoder_distance}), flush=True)
                    if stopped_at is None:
                        traversing = True
                        forward = args.speed*min(1., (elapsed-2)/1.)
                    elif now-stopped_at >= 2:
                        break
                previous_speed = forward
                await send({'type': 'drive', 'enabled': True, 'x': -forward, 'y': 0, 'theta': theta})
                if args.wrist_offset:
                    pose = dict(initial_pose)
                    pose['wrist_flex'] += args.wrist_offset*min(1., elapsed/2.)
                    await send({'type': 'joints', 'enabled': True, 'values': pose})
                await asyncio.sleep(.10)
        finally:
            try:
                try:
                    await send({'type': 'stop'})
                finally:
                    await pi.send(json.dumps({'type': 'stop'}))
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                (folder/'capture.json').write_text(json.dumps({'move': args.move, 'wrist_offset': args.wrist_offset, 'requested_distance_m': args.distance, 'commanded_distance_m': commanded_distance, 'wheel_estimate_m': encoder_distance, 'distance_warning': 'Wheel-based estimate, not externally measured distance; rug slip possible', 'frames': frames, 'events': events}, indent=2))
                if len(frames) >= 2:
                    first = cv2.imread(str(folder/frames[0]['file']))
                    fps = (len(frames)-1)/max(.01, frames[-1]['t']-frames[0]['t'])
                    writer = cv2.VideoWriter(str(folder/'raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (first.shape[1], first.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Video writer failed; JPEGs preserved')
                    for frame in frames:
                        writer.write(cv2.imread(str(folder/frame['file'])))
                    writer.release()
                print(json.dumps({'saved': str(folder), 'frames': len(frames)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub', default='ws://127.0.0.1:8080/ws')
    parser.add_argument('--pi', default='ws://192.168.0.119:8765')
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--distance', type=float, default=0., help='Forward pass in metres (0-0.5); requires verified clear path')
    parser.add_argument('--speed', type=float, default=.04, help='Forward speed, limited to 0.02-0.06 m/s')
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--wrist-offset', type=float, default=0., help='Small wrist-only adjustment, ramped over 2 seconds; left at adjusted pose')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error('Use 1-30 seconds')
    if not -.25 <= args.wrist_offset <= .25:
        parser.error('Wrist adjustment is limited to +/-0.25 rad')
    if not 0 <= args.distance <= .5 or not .02 <= args.speed <= .06:
        parser.error('Distance must be 0-0.5 m and speed 0.02-0.06 m/s')
    if args.distance and (args.move or args.wrist_offset):
        parser.error('Forward traversal cannot be combined with turns or wrist movements')
    asyncio.run(capture(args))
