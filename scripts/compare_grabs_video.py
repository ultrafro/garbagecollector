"""Side-by-side wrist-camera comparison of recorded grabs (scripts/record_grab.py), aligned to the grab command.

Each panel shows the raw wrist camera, the grab phase, commanded vs measured dive travel (from the dive CSV),
and the gripper position. The dive/grip window is repeated in slow motion.
"""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, size=.5, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, org, FONT, size, (0, 0, 0), thick+3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, size, color, thick, cv2.LINE_AA)


def load(folder, label):
    capture = json.loads((folder/'capture.json').read_text())
    events = capture['events']
    sent = next(e['t'] for e in events if e.get('sent'))
    notices = [(e['t']-sent, e['hub']['message']) for e in events if e.get('hub', {}).get('type') in ('notice', 'error')]
    dive_start = next(t for t, m in notices if m.startswith('Grab: approaching'))
    dive_end, dive_csv = next((t, re.search(r'(grab-dive-\d+\.csv)', m).group(1)) for t, m in notices if 'Dive log:' in m)
    with open(Path('screenshots')/dive_csv) as f:
        dive = [(float(r['t_s']), float(r['mm']), float(r['measured_mm'])) for r in csv.DictReader(f)]
    # CSV time starts at the dive; scale it onto the notice window so it lines up with the video.
    span = dive[-1][0] or 1.
    dive = [(dive_start+(t/span)*(dive_end-dive_start), c, m) for t, c, m in dive]
    states = [(e['t']-sent, e['pi']['data']) for e in events if e.get('pi', {}).get('type') == 'state']
    phases = []
    for t, m in notices:
        name = ('open gripper' if 'open-gripper' in m else 'DIVE' if m.startswith('Grab: approaching') else
                'GRIP (jaws closing)' if 'Grab: grip' in m else m.split(': ', 1)[-1] if m.startswith('Placement ') else None)
        if name:
            phases.append((t, name))
    frames = [(f['t']-sent, folder/f['file']) for f in capture['raw_frames']]
    return {'label': label, 'frames': frames, 'dive': dive, 'states': states, 'phases': phases}


def panel(run, t, width):
    frame = min(run['frames'], key=lambda f: abs(f[0]-t))
    image = cv2.imread(str(frame[1]))
    image = cv2.resize(image, (width, int(width*image.shape[0]/image.shape[1])))
    good = run['label'].startswith('HIT')
    text(image, run['label'], (10, 30), .8, (80, 220, 80) if good else (80, 80, 255), 2)
    phase = next((p for tt, p in reversed(run['phases']) if tt <= t), 'aiming (before grab)')
    text(image, phase, (10, 58), .65, (0, 220, 255), 2)
    past = [d for d in run['dive'] if d[0] <= t]
    if past:
        _, commanded, measured = past[-1]
        text(image, f'dive travel: commanded {commanded:5.0f} mm   measured {measured:5.0f} mm', (10, image.shape[0]-40), .55)
        text(image, f'arm lag {commanded-measured:4.0f} mm', (10, image.shape[0]-16), .6,
             (80, 80, 255) if commanded-measured > 30 else (230, 230, 230), 2)
    state = next((s for tt, s in reversed(run['states']) if tt <= t), None)
    if state:
        text(image, f"gripper {state['gripper']:+.2f}", (image.shape[1]-170, 30), .6, (230, 230, 230), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='folder=LABEL, e.g. recordings/grab-x="HIT"')
    parser.add_argument('--out', type=Path, default=Path('screenshots/grab-compare.mp4'))
    parser.add_argument('--slow', type=float, default=4., help='slow-motion factor for the dive/grip window')
    args = parser.parse_args()
    runs = [load(Path(spec.split('=')[0]), spec.split('=', 1)[1]) for spec in args.runs]
    width = 640
    fps = 15.
    start = -1.0
    end = min(r['frames'][-1][0] for r in runs)
    slow_from = min(r['dive'][0][0] for r in runs)-.3
    slow_to = max(r['dive'][-1][0] for r in runs)+1.2

    raw = args.out.with_name(args.out.stem+'-mp4v.mp4')
    writer = None

    def write_segment(t_from, t_to, step, banner):
        nonlocal writer
        t = t_from
        while t <= t_to:
            tiles = [panel(r, t, width) for r in runs]
            row = np.concatenate(tiles, axis=1)
            bar = np.full((40, row.shape[1], 3), 24, np.uint8)
            text(bar, f'{banner}   t = {t:+.2f} s from grab command', (10, 27), .65, (255, 255, 255), 2)
            out = np.concatenate([bar, row], axis=0)
            if writer is None:
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out.shape[1], out.shape[0]))
            writer.write(out)
            t += step

    write_segment(start, end, 1/fps, 'REAL TIME')
    write_segment(slow_from, slow_to, 1/(fps*args.slow), f'SLOW MOTION {args.slow:.0f}x: dive + grip')
    writer.release()
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.out)], check=True)
    raw.unlink()
    print('video:', args.out)


if __name__ == '__main__':
    main()
