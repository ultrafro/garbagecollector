"""Small, bounded live tracking experiment; default is read-only planning."""
import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
import numpy as np
import websockets
from autonomy.wrist_ik import WristIK


async def run(execute=False, tracking_stops=True, radius_mm=5., points=24, step_seconds=.96):
    if not 0 < radius_mm <= 10 or not 24 <= points <= 360 or not .08 <= step_seconds <= 1.:
        raise ValueError('Probe parameters outside bounded range')
    ik = WristIK()
    names = ik.names + ['gripper']
    async with websockets.connect('ws://127.0.0.1:8080/ws') as hub, websockets.connect('ws://192.168.0.119:8765') as pi:
        async def state():
            while True:
                raw = await asyncio.wait_for(pi.recv(), 1.5)
                if isinstance(raw, str):
                    message = json.loads(raw)
                    if message.get('type') == 'state':
                        data = message['data']
                        if not data.get('hardware') or any(n not in data for n in names):
                            raise RuntimeError('Invalid hardware telemetry')
                        return {n: float(data[n]) for n in names}
        while True:
            raw = await asyncio.wait_for(hub.recv(), 3)
            if isinstance(raw, str):
                m = json.loads(raw)
                if m.get('type') == 'autonomy':
                    if m['enabled'] or m['grab_active']:
                        raise RuntimeError('Robot is not idle')
                    break
        initial = await state()
        transform = ik.fk(initial)
        origin = transform[:3, 3]
        pan = initial['shoulder_pan']
        radial = np.array([np.cos(pan), -np.sin(pan), 0.])
        radius = radius_mm / 1000
        poses, targets = [], []
        seed = initial
        for theta in np.linspace(0, 2*np.pi, points + 1):
            target = origin + radius*np.sin(theta)*radial + np.array([0, 0, radius*(1-np.cos(theta))])
            seed = ik.solve(seed, transform, 0., desired_position=target)
            poses.append(seed)
            targets.append(target)
        print(json.dumps({'planned': f'{radius_mm:g} mm radius circle, vertical radial plane, entirely above starting height', 'start_joints': initial, 'start_xyz': origin.tolist(), 'points': len(poses), 'step_seconds': step_seconds, 'execute': execute, 'tracking_stops': tracking_stops}), flush=True)
        if not execute:
            return
        rows = []
        current = initial
        latest_at = time.monotonic()
        sequence = 0
        abort = None
        async def receive():
            nonlocal current, latest_at, sequence
            while True:
                current = await state()
                latest_at = time.monotonic()
                sequence += 1
        async def drain_hub():
            nonlocal abort
            async for raw in hub:
                if isinstance(raw, str):
                    m = json.loads(raw)
                    if m.get('type') == 'autonomy' and (m.get('enabled') or m.get('grab_active')):
                        abort = 'Another robot mode started'
        reader = asyncio.create_task(receive())
        hub_reader = asyncio.create_task(drain_hub())
        started = time.monotonic()
        outcome = 'completed'
        try:
            await pi.send(json.dumps({'type': 'stop'}))
            for index, (pose, target) in enumerate(zip(poses, targets)):
                if abort or reader.done() or time.monotonic()-latest_at > 1.:
                    raise RuntimeError(abort or 'Feedback unavailable')
                command_xyz = ik.fk(pose)[:3, 3]
                await pi.send(json.dumps({'type': 'joints', 'enabled': True, 'values': pose}))
                commanded_at = time.monotonic()
                last_sequence = sequence
                while time.monotonic() - commanded_at < step_seconds:
                    await asyncio.sleep(min(.04, max(.001, step_seconds - (time.monotonic() - commanded_at))))
                    if reader.done() or time.monotonic()-latest_at > 1.:
                        raise RuntimeError('Stale telemetry')
                    if sequence == last_sequence:
                        continue
                    last_sequence = sequence
                    measured_xyz = ik.fk(current)[:3, 3]
                    error = float(np.linalg.norm(command_xyz-measured_xyz))
                    joint_error = max(abs(pose[n]-current[n]) for n in ik.names)
                    rows.append({'step': index, 'elapsed_s': time.monotonic()-started,
                                 'after_command_s': latest_at-commanded_at, 'error_mm': error*1000,
                                 **{f'command_{axis}_mm': float(command_xyz[i]*1000) for i, axis in enumerate('xyz')},
                                 **{f'measured_{axis}_mm': float(measured_xyz[i]*1000) for i, axis in enumerate('xyz')},
                                 **{f'command_{n}': pose[n] for n in names},
                                 **{f'measured_{n}': current[n] for n in names}})
                    if tracking_stops and (error > .015 or joint_error > .12):
                        raise RuntimeError(f'Tracking bound exceeded: {error*1000:.1f} mm / {joint_error:.3f} rad')
                if rows and (index % max(1, points//24) == 0 or index == points):
                    print(f'point {index}/{points}: latest XYZ error {rows[-1]["error_mm"]:.2f} mm', flush=True)
        except Exception as exc:
            outcome = str(exc)
            print('STOP:', outcome, flush=True)
        finally:
            if time.monotonic()-latest_at < 1.:
                await pi.send(json.dumps({'type': 'joints', 'enabled': True, 'values': dict(current, gripper=initial['gripper'])}))
            await pi.send(json.dumps({'type': 'stop'}))
            reader.cancel()
            hub_reader.cancel()
            await asyncio.gather(reader, hub_reader, return_exceptions=True)
            folder = Path('screenshots')
            folder.mkdir(exist_ok=True)
            if rows:
                filename = f'circle-probe-{radius_mm:g}mm-{points}steps-{time.time_ns()}.csv'
                with (folder/filename).open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
                print(f'Readings saved: {folder/filename}', flush=True)
                settled = list({row['step']: row for row in rows}.values())
                print(json.dumps({'outcome': outcome, 'points_observed': len(settled), 'samples': len(rows),
                    'endpoint_error_mm_mean': float(np.mean([r['error_mm'] for r in settled])),
                    'endpoint_error_mm_max': max(r['error_mm'] for r in settled),
                    'all_samples_max_mm': max(r['error_mm'] for r in rows)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--no-tracking-stops', action='store_true', help='Record tracking error without aborting; preflight limits and telemetry checks remain')
    parser.add_argument('--radius-mm', type=float, default=5.)
    parser.add_argument('--points', type=int, default=24)
    parser.add_argument('--step-seconds', type=float, default=.96)
    args = parser.parse_args()
    asyncio.run(run(args.execute, not args.no_tracking_stops, args.radius_mm, args.points, args.step_seconds))
