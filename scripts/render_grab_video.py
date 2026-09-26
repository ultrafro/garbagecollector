"""Render a recorded grab (scripts/record_grab.py) as a video with telemetry: phase, notices, gripper, joints."""
import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 1.5
PANEL_W = 460
PLOT_H = 260
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
PHASE_COLORS = [(230, 200, 60), (80, 200, 60), (0, 170, 255), (220, 120, 220), (90, 160, 255), (200, 200, 90), (160, 160, 160)]


def text(img, s, org, size=.5, color=(235, 235, 235), thick=1):
    # OpenCV fonts are ASCII-only; server notices use em dashes and ellipses.
    s = s.replace('—', '-').replace('–', '-').replace('…', '...').encode('ascii', 'replace').decode()
    cv2.putText(img, s, org, FONT, size, (0, 0, 0), thick+2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, size, color, thick, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--source', choices=('hub', 'raw'), default='hub', help='hub = annotated feed, raw = Pi camera')
    args = parser.parse_args()
    capture = json.loads((args.folder/'capture.json').read_text())
    frames = capture[f'{args.source}_frames']
    events = capture['events']
    states = [(e['t'], e['pi']['data']) for e in events if e.get('pi', {}).get('type') == 'state']
    notices = [(e['t'], e['hub']['type'], e['hub']['message']) for e in events if e.get('hub', {}).get('type') in ('notice', 'error')]
    phases = [(e['t'], e['hub'].get('grab_phase')) for e in events if e.get('hub', {}).get('type') == 'autonomy']
    sent = next((e['t'] for e in events if e.get('sent')), 0.)
    t0, t1 = frames[0]['t'], frames[-1]['t']

    # Phase segments (collapse repeats); placement notices refine the put-away phase.
    segments = []
    for t, phase in phases:
        if not segments or segments[-1][1] != phase:
            segments.append([t, phase])
    for t, kind, message in notices:
        if message.startswith('Placement ') and '/' in message.split(':')[0]:
            segments.append([t, 'place ' + message.split(': ', 1)[-1]])
    segments.sort()
    names = list(dict.fromkeys(p for _, p in segments))
    color_of = {p: PHASE_COLORS[i % len(PHASE_COLORS)] for i, p in enumerate(names)}

    h0, w0 = cv2.imread(str(args.folder/frames[0]['file'])).shape[:2]
    vw, vh = int(w0*SCALE), int(h0*SCALE)
    W, H = vw+PANEL_W, vh+PLOT_H
    left, right = 60, W-20

    def tx(t):
        return int(left+(t-t0)/(t1-t0)*(right-left))

    # Static plot strip: phase bar, gripper position, gripper load, wrist/shoulder/elbow positions.
    base = np.full((PLOT_H, W, 3), 26, np.uint8)
    for i, (t, phase) in enumerate(segments):
        end = segments[i+1][0] if i+1 < len(segments) else t1
        if phase:
            cv2.rectangle(base, (tx(max(t, t0)), 8), (tx(min(end, t1)), 26), color_of[phase], -1)
    x = left
    for name in names:
        if name and x < W-120:
            cv2.rectangle(base, (x, 32), (x+10, 42), color_of[name], -1)
            text(base, str(name), (x+14, 42), .36, (200, 200, 200))
            x += 24+len(str(name))*7

    def plot(series, y0, y1, lo, hi, color, label, label_y):
        pts = [(tx(t), int(y1-(v-lo)/(hi-lo)*(y1-y0))) for t, v in series if t0 <= t <= t1]
        if len(pts) > 1:
            cv2.polylines(base, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
        text(base, label, (left+4, label_y), .4, color)

    ts = [t for t, _ in states]
    cv2.rectangle(base, (left, 52), (right, 142), (45, 45, 45), 1)
    plot([(t, s['gripper']) for t, s in states], 56, 138, -1.3, 1.3, (80, 220, 255), 'gripper position (rad; open 1.2, closed limit -1.2)', 66)
    plot([(t, s['servo_load'][5]) for t, s in states if s.get('servo_load')], 56, 138, -600, 600, (120, 120, 255), 'gripper load', 82)
    cv2.line(base, (left, 97), (right, 97), (60, 60, 60), 1)
    cv2.rectangle(base, (left, 150), (right, 232), (45, 45, 45), 1)
    for name, color in (('shoulder_lift', (80, 200, 60)), ('elbow_flex', (230, 200, 60)), ('wrist_flex', (220, 120, 220))):
        plot([(t, s[name]) for t, s in states], 154, 228, -2., 2., color, name, 164+16*['shoulder_lift', 'elbow_flex', 'wrist_flex'].index(name))
    for t, kind, _ in notices:
        cv2.line(base, (tx(t), 50), (tx(t), 234), (0, 0, 255) if kind == 'error' else (110, 110, 110), 1)
    for s in range(int(t0)+1, int(t1)+1):
        text(base, f'{s}s', (tx(s)-8, 252), .36, (150, 150, 150))
    text(base, 'phase', (8, 22), .4, (200, 200, 200))
    text(base, 'grip', (8, 100), .4, (200, 200, 200))
    text(base, 'arm', (8, 195), .4, (200, 200, 200))

    raw_video = args.folder/'grab-overlay-mp4v.mp4'
    fps = (len(frames)-1)/(t1-t0)
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        t = frame['t']
        canvas = np.full((H, W, 3), 20, np.uint8)
        canvas[:vh, :vw] = cv2.resize(cv2.imread(str(args.folder/frame['file'])), (vw, vh))
        text(canvas, f't = {t-sent:+.2f}s from grab command', (10, vh-14), .6, (0, 255, 255), 2)

        px = vw+16
        cv2.rectangle(canvas, (vw, 0), (W, vh), (32, 32, 32), -1)
        phase = next((p for tt, p in reversed(segments) if tt <= t), None) or 'idle'
        cv2.rectangle(canvas, (px, 14), (W-16, 58), color_of.get(phase, (120, 120, 120)), -1)
        cv2.putText(canvas, str(phase).upper()[:26], (px+10, 45), FONT, .8, (20, 20, 20), 2, cv2.LINE_AA)
        state = next((s for tt, s in reversed(states) if tt <= t), None)
        y = 86
        if state:
            for name in JOINTS:
                goal = state.get('servo_goal', {})
                goal = goal.get(name) if isinstance(goal, dict) else None
                text(canvas, f'{name:14s} {state[name]:+6.3f}' + (f'  goal {goal:+6.3f}' if isinstance(goal, (int, float)) else ''),
                     (px, y), .45, (210, 210, 210))
                y += 20
            load = state.get('servo_load') or [0]*6
            text(canvas, 'load  ' + ' '.join(f'{v:+5.0f}' for v in load), (px, y+4), .42, (160, 160, 255))
            y += 30
        recent = [(tt, k, m) for tt, k, m in notices if tt <= t][-3:]
        text(canvas, 'latest notices:', (px, y), .45, (230, 200, 60))
        y += 20
        for tt, kind, message in reversed(recent):
            for i, line in enumerate(textwrap.wrap(f'{tt-sent:+.1f}s {message}', 50)[:4]):
                text(canvas, line, (px, y), .4, (255, 120, 120) if kind == 'error' else (220, 220, 220))
                y += 17
            y += 6
            if y > vh-20:
                break

        strip = base.copy()
        cv2.line(strip, (tx(t), 6), (tx(t), 236), (255, 255, 255), 2)
        canvas[vh:, :] = strip
        writer.write(canvas)
    writer.release()
    final = Path('screenshots')/f'{args.folder.name}.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(raw_video), '-c:v', 'libx264',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(final)], check=True)
    raw_video.unlink()
    print('video:', final)


if __name__ == '__main__':
    main()
