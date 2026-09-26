"""Replay a recording through the hybrid VLM + tracker targeter and render an explanatory overlay video.

Offline only; no robot connection. VLM answers are cached in <folder>/hybrid/vlm-cache.json and released
after their measured latency, so the replay reflects what the live loop would have seen.
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy.vlm_targeting import HybridTargeter, Settings, SimulatedLocator, iou

STATUS_COLORS = {'DRIVE': (80, 200, 60), 'HOLD': (0, 170, 255), 'SEARCHING': (150, 150, 150)}
PRESENCE_COLORS = {'visible': (80, 200, 60), 'partial': (0, 170, 255), 'absent': (90, 90, 90)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 1.5
PANEL_W = 440
TIMELINE_H = 170
MODEL_NAME = 'Qwen3-VL'
VERIFY_MAX_AGE = 3.
TAG = ''


def text(img, s, org, size=.5, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, org, FONT, size, (0, 0, 0), thick+2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, size, color, thick, cv2.LINE_AA)


def run_replay(folder, frames, settings, model_id, four_bit, llama_url=None):
    out = folder/'hybrid'
    out.mkdir(exist_ok=True)
    cache_path = out/f'vlm-cache-{TAG}.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    locator_model = None

    def locate(image):
        nonlocal locator_model
        if locator_model is None:
            from autonomy.vlm_targeting import LlamaServerLocator, QwenLocator
            locator_model = LlamaServerLocator(llama_url) if llama_url else QwenLocator(model_id, four_bit)
            locator_model.locate(image)                      # load + warm-up, not timed
        start = time.perf_counter()
        answer = locator_model.locate(image)
        answer['latency_s'] = time.perf_counter()-start
        return answer

    locator = SimulatedLocator(locate, cache)
    targeter = HybridTargeter(locator, settings)
    snaps, requests = [], {}
    for frame in frames:
        image = cv2.imread(str(folder/frame['file']))
        snap = targeter.step(image, frame['t'])
        if snap.vlm_submitted:
            requests[snap.index] = {'index': snap.index, 't': snap.t, 'latency_s': cache[str(snap.index)]['latency_s']}
        if snap.vlm_answer is not None:
            requests[snap.vlm_answer['index']].update(applied_index=snap.index, applied_t=snap.t,
                                                      boxes=snap.vlm_answer['boxes'], chosen=snap.vlm_answer.get('chosen'),
                                                      agreement_iou=snap.vlm_answer.get('agreement_iou'))
        snaps.append(snap)
        if snap.events or snap.vlm_submitted:
            print(f'{snap.t:6.2f}s #{snap.index:3d} {snap.status:9s} {snap.reason:40s} {"; ".join(snap.events)}'
                  f'{" [VLM request]" if snap.vlm_submitted else ""}', flush=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    return snaps, list(requests.values())


def presence_at(index, labels):
    for start, end, status in labels.get('presence_ranges', []):
        if start <= index <= end:
            return status
    return None


def draw_box(img, box, color, thick, scale=SCALE):
    x1, y1, x2, y2 = (int(v*scale) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    return x1, y1, x2, y2


def thumbnail(folder, frames, index, box, width, color):
    image = cv2.imread(str(folder/frames[index]['file']))
    if box is not None:
        draw_box(image, box, color, 3, 1.)
    return cv2.resize(image, (width, int(width*image.shape[0]/image.shape[1])), interpolation=cv2.INTER_AREA)


def render(folder, frames, snaps, requests, labels, video_path):
    h0, w0 = cv2.imread(str(folder/frames[0]['file'])).shape[:2]
    vw, vh = int(w0*SCALE), int(h0*SCALE)
    W, H = vw+PANEL_W, vh+TIMELINE_H
    t0, t1 = frames[0]['t'], frames[-1]['t']
    fps = (len(frames)-1)/(t1-t0)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    left, right = 150, W-20

    def tx(t):
        return int(left+(t-t0)/(t1-t0)*(right-left))

    # Static timeline background.
    base = np.full((TIMELINE_H, W, 3), 28, np.uint8)
    rows = {'wrapper in view*': 22, 'targeting status': 62, 'VLM requests': 102}
    for name, y in rows.items():
        text(base, name, (10, y+14), .45, (200, 200, 200))
    for i, (snap, frame) in enumerate(zip(snaps, frames)):
        xa = tx(frame['t'])
        xb = tx(frames[i+1]['t']) if i+1 < len(frames) else right
        presence = presence_at(i, labels)
        if presence:
            cv2.rectangle(base, (xa, 22), (xb, 42), PRESENCE_COLORS[presence], -1)
        cv2.rectangle(base, (xa, 62), (xb, 82), STATUS_COLORS[snap.status], -1)
    for req in requests:
        xa, xb = tx(req['t']), tx(req.get('applied_t', t1))
        found = bool(req.get('boxes'))
        color = (230, 200, 60) if found else (130, 130, 130)
        cv2.rectangle(base, (xa+1, 104), (max(xa+2, xb-2), 120), color, -1)
        cv2.line(base, (xa, 100), (xa, 124), (255, 255, 255), 1)
    for s in range(int(t0)+1, int(t1)+1):
        cv2.line(base, (tx(s), 128), (tx(s), 134), (160, 160, 160), 1)
        text(base, f'{s}s', (tx(s)-8, 150), .38, (160, 160, 160))
    text(base, '* approx, hand-marked', (10, 166), .36, (130, 130, 130))
    legend = [('DRIVE', STATUS_COLORS['DRIVE']), ('HOLD', STATUS_COLORS['HOLD']), ('SEARCH', STATUS_COLORS['SEARCHING']),
              ('VLM found trash', (230, 200, 60)), ('VLM: nothing', (130, 130, 130))]
    x = left+30
    for name, color in legend:
        cv2.rectangle(base, (x, 156), (x+12, 166), color, -1)
        text(base, name, (x+16, 166), .38, (200, 200, 200))
        x += 30+len(name)*8

    last_answer = None
    for i, (snap, frame) in enumerate(zip(snaps, frames)):
        canvas = np.full((H, W, 3), 20, np.uint8)
        view = cv2.resize(cv2.imread(str(folder/frame['file'])), (vw, vh), interpolation=cv2.INTER_LINEAR)
        color = STATUS_COLORS[snap.status]

        truth = labels['frames'].get(frame['file'])
        if truth and truth['box']:
            draw_box(view, truth['box'], (255, 255, 255), 1)
            text(view, 'hand reference', (int(truth['box'][0]*SCALE), int(truth['box'][3]*SCALE)+16), .42)
        if snap.box is not None:
            x1, y1, x2, y2 = draw_box(view, snap.box, color, 3)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.drawMarker(view, (cx, cy), color, cv2.MARKER_CROSS, 22, 2)
            tag = f'TARGET  {snap.status}'
            if truth and truth['box']:
                tag += f'  IoU vs ref {iou(snap.box, truth["box"]):.2f}'
            text(view, tag, (x1, max(18, y1-8)), .55, color, 2)
        if snap.vlm_answer is not None:
            last_answer = (snap.vlm_answer, i)
        if last_answer and i-last_answer[1] < 8:
            ans = last_answer[0]
            msg = (f"VLM answer arrived: saw frame #{ans['index']} {ans['latency_s']:.1f}s ago -> "
                   + (f"{len(ans['boxes'])} object(s), tracker re-seeded + caught up" if ans['boxes'] else 'no trash'))
            cv2.rectangle(view, (0, vh-34), (vw, vh), (0, 0, 0), -1)
            text(view, msg, (10, vh-11), .55, (230, 200, 60) if ans['boxes'] else (200, 200, 200), 1)
        canvas[:vh, :vw] = view

        # Side panel.
        px = vw+16
        cv2.rectangle(canvas, (vw, 0), (W, vh), (32, 32, 32), -1)
        cv2.rectangle(canvas, (px, 14), (W-16, 64), color, -1)
        label = {'DRIVE': 'DRIVE OK', 'HOLD': 'HOLD BASE', 'SEARCHING': 'SEARCHING'}[snap.status]
        cv2.putText(canvas, label, (px+12, 51), FONT, 1.05, (20, 20, 20), 3, cv2.LINE_AA)
        text(canvas, snap.reason[:52], (px, 88), .47)
        lines = [f't = {frame["t"]-t0:5.2f}s    frame #{i}',
                 f'last VLM verification: ' + (f'{snap.verified_age:.1f}s ago' if snap.verified_age is not None else '-'),
                 f'appearance match: ' + (f'{snap.appearance:.2f}' if snap.appearance is not None else '-'),
                 f'tracker {snap.track_ms:.0f} ms' + (f'   catch-up {snap.catchup_ms:.0f} ms' if snap.catchup_ms else '')]
        for k, line in enumerate(lines):
            text(canvas, line, (px, 118+k*24), .47, (200, 200, 200))

        thumb_w = (PANEL_W-16*3)//2
        flight = next((r for r in requests if r['t'] <= frame['t'] < r.get('applied_t', 1e9)), None)
        ty = 230
        text(canvas, 'VLM is looking at:', (px, ty), .45, (230, 200, 60))
        if flight:
            thumb = thumbnail(folder, frames, flight['index'], None, thumb_w, (0, 0, 0))
            canvas[ty+8:ty+8+thumb.shape[0], px:px+thumb_w] = thumb
            text(canvas, f"frame #{flight['index']}, {frame['t']-flight['t']:.1f}s in", (px, ty+26+thumb.shape[0]), .42)
        text(canvas, 'Last VLM answer:', (px+thumb_w+16, ty), .45, (230, 200, 60))
        if last_answer:
            ans = last_answer[0]
            thumb = thumbnail(folder, frames, ans['index'], ans.get('chosen'), thumb_w, (230, 200, 60))
            canvas[ty+8:ty+8+thumb.shape[0], px+thumb_w+16:px+2*thumb_w+16] = thumb
            summary = f"#{ans['index']}: " + (f"{ans['boxes'][0]['label']}" if ans['boxes'] else 'no trash')
            text(canvas, summary[:30], (px+thumb_w+16, ty+26+thumb.shape[0]), .42)
            text(canvas, f"latency {ans['latency_s']:.1f}s", (px+thumb_w+16, ty+46+thumb.shape[0]), .42)
        notes = [f'{MODEL_NAME} finds + verifies trash;', 'CSRT follows it between answers.',
                 'Late answers re-seed the tracker on the',
                 'frame the VLM saw, then replay to now.',
                 f'Base may drive only when VLM-verified <{VERIFY_MAX_AGE:.0f}s ago', 'and the target is fully in view.']
        for k, line in enumerate(notes):
            text(canvas, line, (px, vh-128+k*20), .42, (150, 150, 150))

        strip = base.copy()
        cx = tx(frame['t'])
        cv2.line(strip, (cx, 14), (cx, 128), (255, 255, 255), 2)
        canvas[vh:, :] = strip
        writer.write(canvas)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=Path('recordings/targeting-20260926-104256'))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--fp16', dest='four_bit', action='store_false', help='Disable 4-bit quantization')
    parser.add_argument('--llama-url', help='Use a running llama-server (Qwen3-VL GGUF) instead of transformers')
    parser.add_argument('--verify-max-age', type=float, default=None, help='Default: 2 s with llama.cpp, 6 s with transformers')
    args = parser.parse_args()
    capture = json.loads((args.folder/'capture.json').read_text())
    frames = capture['frames']
    labels_path = args.folder/'evaluation-labels.json'
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {'frames': {}}
    global MODEL_NAME, VERIFY_MAX_AGE
    global TAG
    TAG = 'llamacpp-Qwen3-VL-4B-Q4_K_M' if args.llama_url else f"{args.model.split('/')[-1]}{'-4bit' if args.four_bit else ''}"
    MODEL_NAME = ('Qwen3-VL-4B Q4_K_M (llama.cpp)' if args.llama_url else
                  args.model.split('/')[-1].replace('-Instruct', '')+(' (4-bit)' if args.four_bit else ' (fp16)'))
    settings = Settings(verify_max_age=args.verify_max_age or (2. if args.llama_url else 6.))
    VERIFY_MAX_AGE = settings.verify_max_age
    snaps, requests = run_replay(args.folder, frames, settings, args.model, args.four_bit, args.llama_url)

    scored = [(f['file'], s, labels['frames'][f['file']]) for s, f in zip(snaps, frames) if f['file'] in labels['frames']]
    summary = {
        'model': MODEL_NAME,
        'settings': asdict(settings),
        'frames': len(snaps),
        'status_counts': {k: sum(s.status == k for s in snaps) for k in STATUS_COLORS},
        'vlm_requests': len(requests),
        'mean_vlm_latency_s': float(np.mean([r['latency_s'] for r in requests])),
        'max_catchup_ms': max(s.catchup_ms for s in snaps),
        'mean_track_ms': float(np.mean([s.track_ms for s in snaps if s.track_ms])),
        'drive_while_absent': sum(s.status == 'DRIVE' and presence_at(s.index, labels) == 'absent' for s in snaps),
        'labeled_frames': [{'file': f, 'truth': t['status'], 'status': s.status,
                            'iou': round(iou(s.box, t['box']), 2) if t['box'] and s.box else None} for f, s, t in scored],
    }
    out = args.folder/'hybrid'
    tag = TAG
    (out/f'replay-{tag}.json').write_text(json.dumps({'summary': summary, 'requests': requests,
                                               'snapshots': [asdict(s) for s in snaps]}, indent=1, default=str))
    print(json.dumps(summary, indent=1))

    raw_video = out/f'overlay-{tag}-mp4v.mp4'
    render(args.folder, frames, snaps, requests, labels, raw_video)
    final = args.out or Path('screenshots')/f'hybrid-targeting-{tag}.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(raw_video), '-c:v', 'libx264',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(final)], check=True)
    print('video:', final)


if __name__ == '__main__':
    main()
