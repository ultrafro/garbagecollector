"""Compare framewise YOLO with an explicitly human-seeded appearance tracker.

This is an offline experiment, not automatic trash recognition or a motor controller.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy.server import DEFAULT_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--box', type=int, nargs=4, required=True, metavar=('X', 'Y', 'W', 'H'))
    args = parser.parse_args()
    capture = json.loads((args.folder/'capture.json').read_text())
    frames = capture['frames']
    first = cv2.imread(str(args.folder/frames[0]['file']))
    tracker = cv2.TrackerCSRT_create()
    tracker.init(first, tuple(args.box))
    model = YOLO('yolov8s-worldv2.pt')
    model.set_classes(sorted(DEFAULT_LABELS.split(',')))
    fps = (len(frames)-1)/(frames[-1]['t']-frames[0]['t'])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(args.folder/'targeting-comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*2, h))
    if not writer.isOpened():
        raise RuntimeError('Cannot write comparison video')
    rows = []
    sheet = []
    trusted = True
    invalidated_at = None
    try:
        for index, frame in enumerate(frames):
            image = cv2.imread(str(args.folder/frame['file']))
            start = time.perf_counter()
            ok, box = (True, args.box) if index == 0 else tracker.update(image)
            track_ms = (time.perf_counter()-start)*1000
            result = model.predict(image, conf=.004, verbose=False)[0]
            annotated = result.plot()
            tracked = image.copy()
            if ok:
                x, y, bw, bh = (int(v) for v in box)
                cv2.rectangle(tracked, (x,y), (x+bw,y+bh), (0,255,255), 2)
                cv2.drawMarker(tracked, (x+bw//2,y+bh//2), (0,255,255), cv2.MARKER_CROSS, 15, 2)
                # Experimental safety gate: no automatic reacquisition from rug texture.
                if x < .015*w or y < .015*h or x+bw > .985*w or y+bh > .985*h:
                    trusted = False
            else:
                trusted = False
            if not trusted and invalidated_at is None:
                invalidated_at = frame['t']
            if not trusted:
                cv2.putText(tracked, 'EDGE GATE: TARGET LOST - DO NOT DRIVE', (8,24), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2)
            cv2.putText(annotated, 'Framewise YOLO (current vocabulary)', (8,h-12), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(tracked, 'Human-seeded CSRT (NOT automatic recognition)', (8,h-12), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            combined = np.concatenate([annotated, tracked],axis=1)
            writer.write(combined)
            if index in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]:
                sheet.append(combined)
            rows.append({'file':frame['file'], 't':frame['t'], 'tracker_ok':bool(ok), 'edge_gated_valid':trusted, 'tracker_box':list(box), 'tracker_ms':track_ms,
                         'detections':[{'label':result.names[int(b.cls.item())], 'confidence':float(b.conf.item()), 'box':b.xyxy[0].tolist()} for b in result.boxes]})
    finally:
        writer.release()
    cv2.imwrite(str(args.folder/'tracking-contact-sheet.jpg'), np.concatenate(sheet,axis=0))
    report = {'seed_box_xywh':args.box, 'seed_source':'human visual inspection, not automatic detection', 'frames':len(rows),
              'tracker_reported_success':sum(r['tracker_ok'] for r in rows), 'mean_tracker_ms':float(np.mean([r['tracker_ms'] for r in rows[1:]])),
              'edge_gate_invalidated_at_s': invalidated_at,
              'warning':'Tracker success is not an accuracy metric; inspect overlay for drift. One scene only.', 'rows':rows}
    (args.folder/'tracking.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__ == '__main__':
    main()
