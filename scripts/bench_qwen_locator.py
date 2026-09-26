"""Accuracy/latency of autonomy.vlm_targeting.QwenLocator on the hand-labelled frames. Offline only."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy.vlm_targeting import LlamaServerLocator, QwenLocator, iou

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
parser.add_argument('--fp16', dest='four_bit', action='store_false', help='Disable 4-bit quantization')
parser.add_argument('--llama-url', help='Use a running llama-server instead of transformers')
parser.add_argument('--folder', type=Path, default=Path('recordings/targeting-20260926-104256'))
args = parser.parse_args()
labels = json.loads((args.folder/'evaluation-labels.json').read_text())['frames']
locator = LlamaServerLocator(args.llama_url) if args.llama_url else QwenLocator(args.model, args.four_bit)
locator.locate(cv2.imread(str(args.folder/'raw-00000.jpg')))
hits = fps = 0
times = []
for name, truth in labels.items():
    start = time.perf_counter()
    answer = locator.locate(cv2.imread(str(args.folder/name)))
    ms = (time.perf_counter()-start)*1000
    times.append(ms)
    best = max((iou(b['box'], truth['box']) for b in answer['boxes']), default=0.) if truth['box'] else None
    hits += best is not None and best >= .5
    fps += truth['box'] is None and bool(answer['boxes'])
    print(f"{name} {truth['status']:8s} {ms:6.0f}ms tokens={answer['tokens']:2d} iou={best if best is None else round(best, 2)} raw={answer['raw']!r}")
print(json.dumps({'model': args.llama_url or args.model, 'four_bit': args.four_bit, 'hits_of_6': hits, 'empty_false_positives_of_6': fps,
                  'mean_ms': round(sum(times)/len(times))}))
