"""Capture a bounded live approach and cancel before grip/put-away."""
import asyncio
import json
import time
import sys
from pathlib import Path
import websockets

async def main():
    folder = Path('recordings') / ('grab-observe-' + time.strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True)
    rows = []
    async with websockets.connect('ws://127.0.0.1:8080/ws', max_size=4000000) as ws:
        start = time.monotonic()
        last_frame = -1
        try:
            await ws.send(json.dumps({'type': 'test_grab'}))
            while time.monotonic() - start < (100 if '--full' in sys.argv else 12):
                raw = await asyncio.wait_for(ws.recv(), 3)
                elapsed = time.monotonic() - start
                if isinstance(raw, bytes):
                    if elapsed-last_frame >= .3:
                        (folder / f'{elapsed:06.2f}.jpg').write_bytes(raw)
                        last_frame = elapsed
                    continue
                message = json.loads(raw)
                rows.append({'t': elapsed, 'message': message})
                if message.get('type') in ('notice', 'error'):
                    print(round(elapsed, 2), message.get('message'), flush=True)
                    if 'Closing gripper' in message.get('message', '') and '--full' not in sys.argv:
                        break
                    if 'Put-away motion finished' in message.get('message', '') or message.get('type') == 'error':
                        break
        finally:
            await ws.send(json.dumps({'type': 'stop'}))
            await asyncio.sleep(.3)
            (folder / 'telemetry.json').write_text(json.dumps(rows))
            print('Captured:', folder, flush=True)

asyncio.run(main())
