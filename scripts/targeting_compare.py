"""Offline, same-frame comparison of vocabularies/resolution. Never moves hardware."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy.server import DEFAULT_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--yoloe', action='store_true')
    args = parser.parse_args()
    model = YOLO('yolov8s-worldv2.pt')
    profiles = [
        ('baseline', sorted(DEFAULT_LABELS.split(',')), 640),
        ('focused', ['plastic wrapper', 'snack wrapper', 'crumpled food packaging', 'plastic bottle', 'aluminum can', 'paper cup', 'rug', 'floor'], 640),
        ('focused960', ['plastic wrapper', 'snack wrapper', 'crumpled food packaging', 'plastic bottle', 'aluminum can', 'paper cup', 'rug', 'floor'], 960),
        ('packaging', ['discarded packaging', 'piece of litter', 'plastic wrapper'], 640),
    ]
    if args.yoloe:
        model = YOLOE('yoloe-11s-seg-pf.pt')
        profiles = [('yoloe', None, 640)]
    files = sorted(args.folder.glob('raw-*.jpg'))[::args.stride]
    report = {}
    for name, labels, size in profiles:
        if labels is not None:
            model.set_classes(labels)
        results = []
        times = []
        thumbnails = []
        for index, file in enumerate(files):
            image = cv2.imread(str(file))
            start = time.perf_counter()
            result = model.predict(image, conf=.004, imgsz=size, verbose=False)[0]
            times.append(time.perf_counter()-start)
            boxes = [{'label': result.names[int(b.cls.item())], 'confidence': float(b.conf.item()), 'xyxy': b.xyxy[0].tolist()} for b in result.boxes]
            results.append({'file': file.name, 'boxes': boxes})
            if index in [0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]:
                plotted = result.plot()
                cv2.putText(plotted, f'{name} {file.name}', (10, plotted.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
                thumbnails.append(plotted)
        if thumbnails:
            cv2.imwrite(str(args.folder/f'comparison-{name}.jpg'), np.concatenate(thumbnails, axis=0))
        report[name] = {'labels': labels, 'imgsz': size, 'mean_inference_ms': float(np.mean(times[1:])*1000), 'frames': results}
        print(name, 'frames', len(results), 'labels', dict(Counter(b['label'] for r in results for b in r['boxes'])), 'mean_ms', round(report[name]['mean_inference_ms']), flush=True)
        (args.folder/('comparison-yoloe.json' if args.yoloe else 'comparison.json')).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
