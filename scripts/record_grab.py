"""Trigger a test grab (or just observe) and record everything: hub (annotated) frames, raw Pi frames, hub JSON, Pi telemetry.

Ends when the grab finishes or fails, or after --timeout (then sends stop). Frames are written as they arrive.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets

DONE = ('Put-away motion finished', 'Placement complete', 'Grab ended', 'Grab refused', 'Grab aborted',
        'Grasp missed on all', 'Retry stopped')


async def main(args):
    folder = Path('recordings') / ('grab-' + time.strftime('%Y%m%d-%H%M%S'))
    (folder/'hub').mkdir(parents=True)
    (folder/'raw').mkdir()
    events, hub_frames, raw_frames = [], [], []
    start = time.monotonic()
    finished = asyncio.Event()
    seen_active = False

    async with websockets.connect(args.hub, max_size=4_000_000) as hub, websockets.connect(args.pi, max_size=4_000_000) as pi:
        async def read_hub():
            nonlocal seen_active
            async for raw in hub:
                t = time.monotonic()-start
                if isinstance(raw, bytes):
                    name = f'hub/{len(hub_frames):05d}.jpg'
                    (folder/name).write_bytes(raw)
                    hub_frames.append({'t': t, 'file': name})
                    continue
                message = json.loads(raw)
                events.append({'t': t, 'hub': message})
                if message.get('type') in ('notice', 'error'):
                    print(f"{t:6.2f}s {message['type']}: {message.get('message')}", flush=True)
                    if message['type'] == 'error' or any(k in message.get('message', '') for k in DONE):
                        finished.set()
                if message.get('type') == 'autonomy':
                    seen_active |= bool(message.get('grab_active'))
                    # Between retry attempts grab_active drops while grab_attempt stays set.
                    if seen_active and not message.get('grab_active') and not message.get('grab_attempt'):
                        finished.set()

        async def read_pi():
            async for raw in pi:
                t = time.monotonic()-start
                if isinstance(raw, bytes):
                    name = f'raw/{len(raw_frames):05d}.jpg'
                    (folder/name).write_bytes(raw)
                    raw_frames.append({'t': t, 'file': name})
                else:
                    events.append({'t': t, 'pi': json.loads(raw)})

        tasks = [asyncio.create_task(read_hub()), asyncio.create_task(read_pi())]
        clean = False
        try:
            if args.observe:                                  # passive: record whatever the robot is doing
                events.append({'t': 0., 'sent': 'observe'})
                await asyncio.sleep(args.observe)
                clean = True
                return
            await asyncio.sleep(args.pre)                     # a little footage before the grab starts
            print(f'{time.monotonic()-start:6.2f}s sending test_grab', flush=True)
            events.append({'t': time.monotonic()-start, 'sent': 'test_grab'})
            await hub.send(json.dumps({'type': 'test_grab'}))
            await asyncio.wait_for(finished.wait(), args.timeout)
            await asyncio.sleep(args.post)
            clean = True
        finally:
            if not clean:
                print('timeout/exception: sending stop', flush=True)
                await hub.send(json.dumps({'type': 'stop'}))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            (folder/'capture.json').write_text(json.dumps({'hub_frames': hub_frames, 'raw_frames': raw_frames, 'events': events}))
            print(json.dumps({'saved': str(folder), 'hub_frames': len(hub_frames), 'raw_frames': len(raw_frames),
                              'seconds': round(time.monotonic()-start, 1)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub', default='ws://127.0.0.1:8080/ws')
    parser.add_argument('--pi', default='ws://192.168.0.119:8765')
    parser.add_argument('--pre', type=float, default=2.)
    parser.add_argument('--post', type=float, default=2.)
    parser.add_argument('--timeout', type=float, default=240.)
    parser.add_argument('--observe', type=float, default=0., help='Only record for this many seconds; send nothing')
    asyncio.run(main(parser.parse_args()))
