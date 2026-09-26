"""Diagnostic: small elbow reversals with fixed goals and long observation holds."""
import asyncio
import csv
import json
import time
from pathlib import Path
import websockets
from autonomy.wrist_ik import WristIK


async def run():
    names = WristIK().names + ['gripper']
    async with websockets.connect('ws://127.0.0.1:8080/ws') as hub, websockets.connect('ws://192.168.0.119:8765') as ws:
        while True:
            raw = await hub.recv()
            if isinstance(raw, str):
                m = json.loads(raw)
                if m.get('type') == 'autonomy':
                    assert not m['enabled'] and not m['grab_active'], 'Robot active'
                    break
        async def receive():
            while True:
                raw = await asyncio.wait_for(ws.recv(), 1.5)
                if isinstance(raw, str):
                    m = json.loads(raw)
                    if m.get('type') == 'state':
                        return m['data']
        current = await receive()
        initial = {n: current[n] for n in names}
        plan = [0., -.04, -.08, -.04, 0., .04, 0.]
        assert all(-1.69 <= initial['elbow_flex']+d <= 1.69 for d in plan)
        rows = []
        try:
            await ws.send(json.dumps({'type': 'stop'}))
            for offset in plan:
                pose = dict(initial, elbow_flex=initial['elbow_flex']+offset)
                print(f'Elbow step {offset:+.3f} rad; holding 4 seconds', flush=True)
                await ws.send(json.dumps({'type': 'joints', 'enabled': True, 'values': pose}))
                started = time.monotonic()
                while time.monotonic()-started < 4:
                    current = await receive()
                    if abs(current['elbow_flex']-pose['elbow_flex']) > .15:
                        raise RuntimeError('Unexpected elbow displacement')
                    rows.append({'offset': offset, 'time_s': time.monotonic()-started,
                        'command': pose['elbow_flex'], 'read_goal': current['servo_goal'][2],
                        'measured': current['elbow_flex'], 'load': current.get('servo_load', [0]*6)[2],
                        'current': current.get('servo_current', [0]*6)[2],
                        'sample_monotonic': current.get('sample_monotonic')})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            await ws.send(json.dumps({'type': 'joints', 'enabled': True, 'values': {n: current[n] for n in names}}))
            await ws.send(json.dumps({'type': 'stop'}))
            if rows:
                path = Path('screenshots') / f'elbow-probe-{time.time_ns()}.csv'
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
                print(str(path), flush=True)


if __name__ == '__main__':
    asyncio.run(run())
