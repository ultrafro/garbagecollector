"""Score different VLM grasp-check strategies on recorded grabs with known outcomes. Offline; needs llama-server.

Frames: from the gripper closing until the gripper is ~170 mm up, labelled by the run's outcome.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy.vlm_targeting import GRASP_PROMPT, PROMPT, LlamaServerLocator, parse_boxes
from autonomy.wrist_ik import WristIK

CAMERA = 'These photos come from a camera mounted on a robot gripper; the two gripper jaws are at the bottom of the image. '


def word(answer, yes, no):
    """First of the two keywords that appears in the answer (last one wins for reasoning answers)."""
    found = [(m.start(), m.group(0).upper()) for m in re.finditer(rf'\b({yes}|{no})\b', answer, re.I)]
    return found[-1][1] == yes.upper() if found else None


def crop_jaws(image):
    h, w = image.shape[:2]
    return image[h//2:, w//6:5*w//6]


def crop_box(image, box, pad=.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    return image[int(max(0, y1-pad*bh)):int(min(h, y2+pad*bh)), int(max(0, x1-pad*bw)):int(min(w, x2+pad*bw))]


def bottom_strip(image, fraction):
    h = image.shape[0]
    return image[int(h*(1-fraction)):, :]


def strategies(loc):
    """Each returns True (held), False (empty) or None (unparseable), given (before, after, before_box)."""
    return {
        'held_or_empty': lambda b, a, box: word(loc.ask([a], GRASP_PROMPT, 8)[0], 'HELD', 'EMPTY'),
        'is_gripper_empty': lambda b, a, box: (lambda r: None if r is None else not r)(word(loc.ask([a],
            CAMERA + 'The gripper just closed and lifted. Is the gripper empty, with nothing pinched between its jaws? '
            'Answer YES or NO.', 8)[0], 'YES', 'NO')),
        'anything_between_jaws': lambda b, a, box: word(loc.ask([a],
            CAMERA + 'Is there any object (plastic, wrapper, paper, film) pinched between the two gripper jaws, moving with the gripper? '
            'Things lying on the floor do not count. Answer YES or NO.', 8)[0], 'YES', 'NO'),
        'before_after_same_object_in_hand': lambda b, a, box: word(loc.ask([b, a],
            CAMERA + 'Photo 1: before the grasp, a piece of trash lies on the floor. Photo 2: after the gripper closed and lifted. '
            'Is the same piece of trash from photo 1 now in the gripper in photo 2? Answer YES or NO.', 8)[0], 'YES', 'NO'),
        'target_crop_then_after': lambda b, a, box: word(loc.ask([crop_box(b, box), a],
            'Photo 1 shows a piece of trash. Photo 2 is from a camera on a robot gripper whose jaws are at the bottom of the image, '
            'taken after the gripper closed and lifted. Is the object from photo 1 held by the gripper in photo 2? Answer YES or NO.', 8)[0],
            'YES', 'NO'),
        'jaw_crop_only': lambda b, a, box: word(loc.ask([crop_jaws(a)],
            'Close-up of the area between a robot gripper\'s two jaws. Is some object (plastic, wrapper, film, paper) held in it? '
            'Answer YES or NO.', 8)[0], 'YES', 'NO'),
        'describe_then_decide': lambda b, a, box: word(loc.ask([a],
            CAMERA + 'The gripper just closed on a piece of trash and lifted. In one sentence, describe what is between the jaws and '
            'what is on the floor. Then on a new line write HELD if the trash is in the gripper, or EMPTY if it was left behind.', 70)[0],
            'HELD', 'EMPTY'),
        'bottom30_anything': lambda b, a, box: word(loc.ask([bottom_strip(a, .3)],
            'This is the bottom strip of a photo from a camera on a robot gripper. Is there any object (plastic, wrapper, '
            'paper, peel, film) in this strip, close to the camera, rather than only floor or rug? Answer YES or NO.', 8)[0], 'YES', 'NO'),
        'bottom45_anything': lambda b, a, box: word(loc.ask([bottom_strip(a, .45)],
            'This is the bottom part of a photo from a camera on a robot gripper. Is there any object (plastic, wrapper, '
            'paper, peel, film) in it, close to the camera, rather than only floor or rug? Answer YES or NO.', 8)[0], 'YES', 'NO'),
        'full_bottom_edge': lambda b, a, box: word(loc.ask([a],
            CAMERA + 'The gripper just closed and lifted. Is there an object sticking into the image from the bottom edge, '
            'very close to the camera and held by the gripper (not lying on the floor further away)? Answer YES or NO.', 8)[0],
            'YES', 'NO'),
        'box_held_object': lambda b, a, box: (lambda r: None if r['parse_error'] else bool(r['boxes']))(parse_boxes(loc.ask([a],
            CAMERA + 'Box any object that is held between the gripper jaws (moving with the gripper). Ignore everything lying on the floor. '
            'Answer with only a JSON list [[x1,y1,x2,y2],...] normalized to 0-1000, or [] if the gripper holds nothing.', 60)[0],
            a.shape[1], a.shape[0])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=['grab-20260926-125919=held', 'grab-20260926-130150=missed',
                                                      'grab-20260926-133128=missed'])
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    ik, loc = WristIK(), LlamaServerLocator()
    tests = strategies(loc)
    if args.only:
        tests = {k: v for k, v in tests.items() if k in args.only}
    samples = []
    for spec in args.runs:
        run, outcome = spec.split('=')
        folder = Path('recordings')/run
        d = json.loads((folder/'capture.json').read_text())
        sent = next(e['t'] for e in d['events'] if e.get('sent'))
        states = [(e['t']-sent, e['pi']['data']) for e in d['events'] if e.get('pi', {}).get('type') == 'state']
        before = cv2.imread(str(folder/min(d['raw_frames'], key=lambda f: abs(f['t']-sent+.3))['file']))
        box = parse_boxes(loc.ask([before], PROMPT)[0], before.shape[1], before.shape[0])['boxes']
        box = box[0]['box'] if box else [0, 0, before.shape[1], before.shape[0]]
        closed = False
        for f in d['raw_frames']:
            t = f['t']-sent
            s = min(states, key=lambda x: abs(x[0]-t))[1]
            closed |= t > .5 and s['gripper'] < -.7
            z = ik.fk(s)[2, 3]*1000
            if closed and t < 8 and z <= 170:
                samples.append({'run': run, 'held': outcome == 'held', 't': t, 'z': z, 'before': before, 'box': box,
                                'after': cv2.imread(str(folder/f['file']))})
            if closed and z > 170:
                break
    print(f'{len(samples)} frames: ' + ', '.join(f"{r}: {sum(s['run'] == r for s in samples)}" for r in dict.fromkeys(s['run'] for s in samples)))
    results = {}
    for name, test in tests.items():
        rows = []
        start = time.perf_counter()
        for s in samples:
            rows.append({'run': s['run'], 'held': s['held'], 'z': round(s['z']), 't': round(s['t'], 2),
                         'pred': test(s['before'], s['after'], s['box'])})
        ms = (time.perf_counter()-start)*1000/len(samples)
        results[name] = rows
        line = [f'{name:34s} {ms:5.0f} ms/frame']
        for label, band in (('all', (-1e9, 1e9)), ('z<60', (-1e9, 60)), ('z>=80', (80, 1e9))):
            sel = [r for r in rows if band[0] <= r['z'] < band[1]]
            held = [r for r in sel if r['held']]
            miss = [r for r in sel if not r['held']]
            line.append(f"{label}: held {sum(r['pred'] is True for r in held)}/{len(held)} "
                        f"missed {sum(r['pred'] is False for r in miss)}/{len(miss)}")
        print(' | '.join(line), flush=True)
    out = Path('recordings')/'grasp-prompt-eval.json'
    out.write_text(json.dumps(results, indent=1))
    print('saved', out)


if __name__ == '__main__':
    main()
