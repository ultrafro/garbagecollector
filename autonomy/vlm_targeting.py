"""Hybrid targeting: a slow VLM finds/verifies trash, a fast tracker follows it between answers.

The VLM answers late (seconds), so each answer refers to an old frame. When it
arrives, the tracker is re-seeded on that old frame and replayed through the
buffered frames to "now" (catch-up). Drive permission is only granted while the
track is recently verified, fully in view, and still looks like what the VLM saw.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# Detection prompt: focused on small litter on the floor. Compared offline against the earlier generic
# "discarded packaging or litter" prompt: same wrapper recall (11/11, and every sweep frame), but silent on a
# cluttered scene with a cracker box and shelf items (10/10 frames vs 0/10).
PROMPT = ('Find small pieces of litter lying on the floor that someone would pick up and throw away '
          '(e.g. a crumpled wrapper, a scrap of paper, a tissue, an empty can). '
          'Ignore anything large, anything on furniture, and the rug, floor, shadows and patterns. Include partially visible litter. '
          'Box the WHOLE object, not its printed logo. Answer with only a JSON list of boxes [[x1,y1,x2,y2],...] '
          'in integer coordinates normalized to 0-1000, or [] if there is none.')

GRASP_PROMPT = ('This photo is from a camera mounted on a robot gripper, looking past the two gripper jaws toward the floor. '
                'The gripper just closed to pick up a piece of trash and lifted slightly. '
                'Is a piece of trash (such as a plastic wrapper) now held by the gripper: very close to the camera, pinched between '
                'the jaws near the bottom of the image? Trash lying on the floor further away does not count. '
                'Answer with one word: HELD or EMPTY.')

# Grasp check questions, asked in this order on successive frames (see scripts/eval_grasp_prompts.py).
# Each entry: (name, prompt, held keyword, empty keyword, crop to the jaw area).
GRASP_CHECKS = [
    ('held_or_empty', GRASP_PROMPT, 'HELD', 'EMPTY', False),
    ('between_jaws', 'These photos come from a camera mounted on a robot gripper; the two gripper jaws are at the bottom of the image. '
                     'Is there any object (plastic, wrapper, paper, film) pinched between the two gripper jaws, moving with the gripper? '
                     'Things lying on the floor do not count. Answer YES or NO.', 'YES', 'NO', False),
    ('jaw_crop', "Close-up of the area between a robot gripper's two jaws. Is some object (plastic, wrapper, film, paper) held in it? "
                 'Answer YES or NO.', 'YES', 'NO', True),
]

# Asked about each new target (and a tracked one every few seconds), on the full frame with the target outlined
# so the context shows: a crop of a label inside a sealed bag looks like a scrap of paper. Offline: accepted a lemon
# peel (10/10 frames; it reads as a "shell" without the hint), all wrapper views (16/16), and rejected 7/7 non-trash
# scenes (zip-lock bag with contents, cracker box, shelf clutter).
_KEEP = ('part of something bigger, inside a closed bag or container, a label or print on another object, a box or package, '
         'a tool, toy, clothing or cable, office supplies such as binder clips, pens or paper clips, or anything large or heavy')
CLASSIFY_PROMPT = ('A small floor-cleaning robot sees the object outlined in yellow. Look at the whole scene around it. '
                   'Is the outlined object loose TRASH lying on the floor that someone would throw away (an empty wrapper, '
                   'a scrap of paper, a used tissue, an empty can, bottle or cup, or food waste: a fruit or lemon peel, core, '
                   'pit, nut shell or eggshell; a curled peel often looks like a shell, and a shell or peel lying on the floor '
                   f'is trash), or should the robot leave it alone (KEEP): {_KEEP}? If the outlined thing is seen through clear '
                   'plastic, it is inside a bag or container: answer KEEP. Answer TRASH or KEEP.')


def crop_box(image, box, pad=.1):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    return image[int(max(0, y1-pad*bh)):int(min(h, y2+pad*bh)), int(max(0, x1-pad*bw)):int(min(w, x2+pad*bw))]


def grasp_answer(answer, held_word, empty_word):
    """True/False from the last keyword in the answer, None if neither appears."""
    found = re.findall(rf'\b({held_word}|{empty_word})\b', answer, re.I)
    return found[-1].upper() == held_word.upper() if found else None


def jaw_crop(image):
    h, w = image.shape[:2]
    return image[h//2:, w//6:5*w//6]


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    inter = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.0


def parse_boxes(raw, width, height):
    """Qwen answer ([[x1,y1,x2,y2],...] normalized 0-1000, possibly fenced or as dicts) -> pixel boxes."""
    try:
        parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
        if isinstance(parsed, dict):
            parsed = parsed.get('objects', [])
        if parsed and isinstance(parsed[0], (int, float)):
            parsed = [parsed]
        parsed = [item.get('box', item.get('bbox_2d')) if isinstance(item, dict) else item for item in parsed]
        boxes = []
        for box in parsed:
            if len(box) == 4 and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in box) and box[2] > box[0] and box[3] > box[1]:
                boxes.append({'label': 'trash', 'box': [box[0]*width/1000, box[1]*height/1000, box[2]*width/1000, box[3]*height/1000]})
        return {'boxes': boxes, 'raw': raw, 'parse_error': False}
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return {'boxes': [], 'raw': raw, 'parse_error': True}


class QwenLocator:
    """Qwen3-VL through Hugging Face transformers on the GPU.

    On Windows this is launch-overhead bound (~130 ms/token for 4B 4-bit); LlamaServerLocator is faster.
    """

    def __init__(self, model_id='Qwen/Qwen3-VL-4B-Instruct', four_bit=True):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        extra = {}
        if four_bit:
            extra['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, device_map={'': 'cuda:0'}, dtype=torch.float16, attn_implementation='sdpa', **extra).eval()
        self.locate(np.zeros((360, 640, 3), np.uint8))   # warm-up so the first live request is not slow

    def locate(self, bgr):
        from PIL import Image
        height, width = bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    return_tensors='pt').to('cuda')
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        new_tokens = generated[:, inputs['input_ids'].shape[1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return {**parse_boxes(raw, width, height), 'tokens': int(new_tokens.shape[1])}


class LlamaServerLocator:
    """Qwen3-VL GGUF served by llama.cpp's llama-server (separate process, OpenAI-compatible API)."""

    def __init__(self, url='http://127.0.0.1:8091', timeout=30.):
        self.url, self.timeout = url.rstrip('/'), timeout

    def ask(self, images, prompt, max_tokens=80):
        """Send BGR images plus a text prompt; return (answer text, completion tokens)."""
        import base64
        import urllib.request
        content = []
        for bgr in images:
            ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError('JPEG encode failed')
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(jpeg.tobytes()).decode()}})
        content.append({'type': 'text', 'text': prompt})
        body = {'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': max_tokens}
        request = urllib.request.Request(self.url + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.loads(response.read())
        return reply['choices'][0]['message']['content'] or '', reply.get('usage', {}).get('completion_tokens', 0)

    def locate(self, bgr):
        height, width = bgr.shape[:2]
        raw, tokens = self.ask([bgr], PROMPT)
        return {**parse_boxes(raw, width, height), 'tokens': tokens}

    def path_clear(self, bgr):
        """True only for an explicit CLEAR answer; anything else counts as blocked."""
        answer, _ = self.ask([bgr], PATH_PROMPT, 6)
        return grasp_answer(answer, 'CLEAR', 'BLOCKED') is True

    def graspable(self, bgr, box):
        """False when the VLM says to leave the outlined object alone (KEEP); True otherwise."""
        outlined = bgr.copy()
        cv2.rectangle(outlined, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)
        answer, _ = self.ask([outlined], CLASSIFY_PROMPT, 6)
        return grasp_answer(answer, 'TRASH', 'KEEP') is not False


class SimulatedLocator:
    """Replay: runs the VLM immediately but releases the answer only after its measured latency."""

    def __init__(self, locate, cache=None):
        self.locate, self.cache, self.pending = locate, cache if cache is not None else {}, None

    @property
    def busy(self):
        return self.pending is not None

    def submit(self, image, t, index, classify=False):
        key = str(index)
        if key not in self.cache:
            start = time.perf_counter()
            answer = self.locate(image)
            answer.setdefault('latency_s', time.perf_counter()-start)
            self.cache[key] = answer
        answer = self.cache[key]
        self.pending = (t+answer['latency_s'], {**answer, 'index': index, 't': t})

    def poll(self, t):
        if self.pending and t >= self.pending[0]:
            answer, self.pending = self.pending[1], None
            return answer
        return None


class ThreadedLocator:
    """Live: one VLM request in flight on a background thread.

    A failed request (e.g. llama-server down) yields no answer -- never an empty "no trash" answer --
    and backs off, so the track simply goes stale and the base holds.
    """

    def __init__(self, locate, retry_s=1.0, graspable=None):
        self.locate, self.retry_s, self.graspable = locate, retry_s, graspable
        self.thread, self.answer, self.error, self.retry_at = None, None, None, 0.

    @property
    def busy(self):
        return self.thread is not None or time.monotonic() < self.retry_at

    def submit(self, image, t, index, classify=False):
        """`classify`: also ask, for up to 3 found objects, whether each is trash (answer['verdicts'])."""
        def work():
            start = time.perf_counter()
            try:
                answer = self.locate(image.copy())
                if classify and self.graspable and answer['boxes']:
                    answer['verdicts'] = [bool(self.graspable(image, b['box'])) for b in answer['boxes'][:3]]
            except Exception as exc:                  # network/model failure: report, do not guess
                self.answer, self.error = None, f'{type(exc).__name__}: {exc}'
                return
            self.answer, self.error = {**answer, 'latency_s': time.perf_counter()-start, 'index': index, 't': t}, None
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def poll(self, t):
        if self.thread and not self.thread.is_alive():
            answer, self.thread, self.answer = self.answer, None, None
            if answer is None:
                self.retry_at = time.monotonic()+self.retry_s
            return answer
        return None


@dataclass
class Settings:
    track_scale: float = .5          # CSRT runs on a downscaled frame
    catchup_stride: int = 2          # replay every Nth buffered frame after a late VLM answer (CSRT ~20 ms/update)
    verify_max_age: float = 2.0      # s since the frame the VLM last confirmed
    edge_margin: float = .015        # fraction of width/height counted as "touching the edge"
    appearance_min: float = .45      # histogram correlation vs. verified patch (coarse: rug-dominated patches still match)
    appearance_frames: int = 3       # consecutive bad frames before the track is dropped
    empty_answers_to_drop: int = 2   # consecutive "no trash" VLM answers before a live track is dropped
    min_agreement: float = .2        # a re-check box must overlap the track this much, or it is a different object
    reject_memory_s: float = 30.     # forget an object's trash/keep votes this long after it was last seen
    reject_match: float = .6         # patch correlation above which a detection "looks like" a remembered object
    votes_kept: int = 5              # trash/keep votes remembered per object; a keep majority means ignore it
    reclassify_s: float = 3.         # re-ask "can it be picked up?" about a tracked target this often
    coast_s: float = 1.5             # after the fast tracker loses the view, wait this long for a VLM re-seed
    buffer_frames: int = 90


@dataclass
class Snapshot:
    index: int
    t: float
    status: str                      # SEARCHING | DRIVE | HOLD
    reason: str
    box: list | None                 # tracked box clipped to the frame (what a controller would centre)
    verified_age: float | None
    appearance: float | None
    track_ms: float
    catchup_ms: float = 0.
    vlm_answer: dict | None = None   # answer applied on this frame, if any
    vlm_submitted: bool = False
    overfills: bool = False          # clipped on opposite edges: too close/big for the view
    events: list = field(default_factory=list)


class HybridTargeter:
    def __init__(self, locator, settings: Settings | None = None):
        self.locator, self.s = locator, settings or Settings()
        self.tracker = None
        self.box = None
        self.verified_t = None
        self.template = None
        self.bad_appearance = 0
        self.empty_answers = 0
        self.rejected = (None, [])       # (time, boxes) of the last objects judged too big to pick up
        self.objects = []                # [{'t', 'box', 'patch', 'votes'}]: trash(True)/keep(False) votes per object seen
        self.classified_t = None         # when the tracked target was last judged graspable
        self.coast_t = None              # fast tracker lost the view at this time; waiting for the VLM to re-seed it
        self.clipped_t = None        # last time the track touched a frame edge (incl. during catch-up)
        self.history = {}            # index -> tracker box, for matching late VLM answers
        self.buffer = deque(maxlen=self.s.buffer_frames)
        self.index = 0

    def _small(self, image):
        return cv2.resize(image, None, fx=self.s.track_scale, fy=self.s.track_scale, interpolation=cv2.INTER_AREA)

    def _start_tracker(self, small, box):
        k = self.s.track_scale
        x1, y1, x2, y2 = (v*k for v in box)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(small, (int(x1), int(y1), max(2, int(x2-x1)), max(2, int(y2-y1))))

    def _update_tracker(self, small):
        ok, (x, y, w, h) = self.tracker.update(small)
        k = self.s.track_scale
        return [x/k, y/k, (x+w)/k, (y+h)/k] if ok else None

    def _edges(self, box, shape):
        h, w = shape[:2]
        mx, my = self.s.edge_margin*w, self.s.edge_margin*h
        x1, y1, x2, y2 = box
        return [name for name, hit in (('top', y1 <= my), ('bottom', y2 >= h-my), ('left', x1 <= mx), ('right', x2 >= w-mx)) if hit]

    @staticmethod
    def _histogram(image, box):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))
        if x2-x1 < 4 or y2-y1 < 4:
            return None
        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _drop(self, snap, reason):
        self.tracker, self.box, self.template, self.verified_t, self.clipped_t = None, None, None, None, None
        self.classified_t = self.coast_t = None
        self.bad_appearance = 0
        snap.events.append(reason)

    def _apply(self, answer, image, snap):
        """Re-seed on the frame the VLM saw, then replay buffered frames up to now."""
        snap.vlm_answer = answer
        old = self.history.get(answer['index'])
        seen = next((f[3] for f in self.buffer if f[0] == answer['index']), image)   # the frame the VLM looked at
        self.objects = [o for o in self.objects if answer['t']-o['t'] <= self.s.reject_memory_s]
        entries = []
        for i, b in enumerate(answer['boxes']):
            patch = self._patch(seen, b['box'])
            entry = self._match(b['box'], patch)
            verdict = answer['verdicts'][i] if i < len(answer.get('verdicts', [])) else None
            if verdict is not None:
                if entry is None:
                    entry = {'votes': []}
                    self.objects.append(entry)
                entry['votes'] = (entry['votes'] + [verdict])[-self.s.votes_kept:]
            if entry is not None:
                entry.update(t=answer['t'], box=b['box'], patch=patch)
            entries.append(entry)
        if answer.get('verdicts'):
            self.classified_t = answer['t']
        keep = [b['box'] for b, e in zip(answer['boxes'], entries) if self._keep(e)]
        if keep:
            self.rejected = (answer['t'], keep)
            if answer.get('verdicts'):
                snap.events.append(f"VLM: ignoring {len(keep)} object(s) that are not trash")
            if self.box is not None and old is not None and any(iou(b, old) >= self.s.min_agreement for b in keep):
                self._drop(snap, 'tracked object voted not trash -> track dropped')
                return False
            answer['boxes'] = [b for b, e in zip(answer['boxes'], entries) if not self._keep(e)]
            entries = [e for e in entries if not self._keep(e)]
        if self.box is None and answer['boxes']:
            # A new target must earn it: at least two trash votes and more trash than keep. One lucky
            # "trash" answer about an ambiguous object (e.g. a binder clip) is not enough to start chasing it.
            # (Only when the locator asks the trash question at all; a replay without it acquires as before.)
            confirmed = [b for b, e in zip(answer['boxes'], entries)
                         if self._trash(e) or not getattr(self.locator, 'graspable', None)]
            if not confirmed and any(e is not None and e['votes'] for e in entries):
                snap.events.append('VLM: candidate needs another trash vote before acquiring')
            answer['boxes'] = confirmed
        if not answer['boxes']:
            return self._unconfirmed(snap, 'no trash visible')
        entry_of = {id(b): e for b, e in zip(answer['boxes'], entries)}
        choice = max(answer['boxes'], key=lambda b: (iou(b['box'], old), (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1])))
        answer['chosen'] = choice['box']
        answer['agreement_iou'] = iou(choice['box'], old) if old else None
        corrected = False
        if self.box is not None and old is not None and answer['agreement_iou'] < self.s.min_agreement:
            # The VLM sees trash somewhere the fast tracker is not (the tracker drifted). Snap back onto it only if
            # it is the single candidate and already voted trash; never silently switch to anything else.
            if len(answer['boxes']) == 1 and self._trash(entry_of.get(id(choice))):
                corrected = True
            else:
                return self._unconfirmed(snap, f"tracked object not found (best overlap {answer['agreement_iou']:.2f})")
        self.empty_answers = 0
        frames = [f for f in self.buffer if f[0] >= answer['index']]
        if not frames or frames[0][0] != answer['index']:
            return False                                   # request frame fell out of the buffer
        start = time.perf_counter()
        was_tracking = self.box is not None
        box, clipped_t = self._catch_up(frames, choice['box'], answer['t'], self.s.catchup_stride)
        if box is None and self.s.catchup_stride > 1:
            box, clipped_t = self._catch_up(frames, choice['box'], answer['t'], 1)    # fast motion: every frame
        snap.catchup_ms = (time.perf_counter()-start)*1000
        if box is None:
            # The VLM did confirm the object; only the fast tracker could not follow the motion since.
            # Coast (report no box, so auto holds still) and let the next VLM answer re-seed it.
            self.tracker, self.box, self.verified_t, self.coast_t = None, list(choice['box']), answer['t'], snap.t
            self.clipped_t = clipped_t
            snap.events.append('catch-up tracking failed; coasting until the next VLM check')
            return True
        self.coast_t = None
        self.box, self.verified_t, self.bad_appearance, self.clipped_t = box, answer['t'], 0, clipped_t
        self.template = self._histogram(frames[0][3], choice['box'])
        snap.events.append(('VLM corrected a drifted track' if corrected else 'VLM re-verified' if was_tracking
                            else 'VLM acquired target') +
                           (f" (agreement IoU {answer['agreement_iou']:.2f})" if answer['agreement_iou'] is not None else ''))
        return True

    def _catch_up(self, frames, box, t0, stride):
        """Seed the tracker on the frame the VLM saw and replay buffered frames to now; (box, last clipped time)."""
        self._start_tracker(frames[0][2], box)
        clipped_t = t0 if self._edges(box, frames[0][3].shape) else None
        replay = frames[1:-1][stride-1::stride]+frames[-1:] if len(frames) > 1 else []
        for _, frame_t, small, full in replay:
            box = self._update_tracker(small)
            if box is None:
                return None, clipped_t
            if self._edges(box, full.shape):
                clipped_t = frame_t
        return box, clipped_t

    @staticmethod
    def _patch(image, box):
        """Normalised 32x32 grey snapshot of a detection, for recognising it again after the view moves."""
        crop = crop_box(image, box, pad=0.)
        if crop.size == 0 or min(crop.shape[:2]) < 4:
            return None
        grey = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        grey -= grey.mean()
        norm = float(np.linalg.norm(grey))
        return (grey / norm) if norm > 1e-3 else None

    def _match(self, box, patch):
        """The remembered object this detection is (same place, or same look at a similar size), if any."""
        area = max(1., (box[2]-box[0])*(box[3]-box[1]))
        best, best_score = None, 0.
        for entry in self.objects:
            score = iou(box, entry['box'])
            score = score + 1. if score >= .3 else 0.
            old_area = max(1., (entry['box'][2]-entry['box'][0])*(entry['box'][3]-entry['box'][1]))
            if (not score and patch is not None and entry['patch'] is not None and .4 <= area/old_area <= 2.5):
                correlation = float((patch*entry['patch']).sum())
                score = correlation if correlation >= self.s.reject_match else 0.
            if score > best_score:
                best, best_score = entry, score
        return best

    @staticmethod
    def _keep(entry):
        """Leave it alone only when 'keep' is the majority of its votes (ties count as trash)."""
        return entry is not None and entry['votes'] and sum(entry['votes'])*2 < len(entry['votes'])

    @staticmethod
    def _trash(entry):
        return entry is not None and sum(entry['votes']) >= 2 and sum(entry['votes'])*2 > len(entry['votes'])

    def give_up(self, image):
        """Stop pursuing the tracked object for a while (e.g. it could not be grabbed): vote it firmly 'keep'."""
        if self.box is None:
            return
        self.objects.append({'t': time.monotonic(), 'box': list(self.box), 'patch': self._patch(image, self.box),
                             'votes': [False]*self.s.votes_kept})
        self.tracker, self.box, self.template, self.verified_t, self.clipped_t = None, None, None, None, None
        self.classified_t = None

    def _unconfirmed(self, snap, why):
        """A VLM answer that does not confirm the track: drop it after enough of these in a row."""
        self.empty_answers += 1
        if self.tracker is not None:
            if self.empty_answers >= self.s.empty_answers_to_drop:
                self._drop(snap, f'VLM: {why} ({self.empty_answers}x) -> track dropped')
            else:
                snap.events.append(f'VLM: {why} ({self.empty_answers}/{self.s.empty_answers_to_drop}); keeping track')
        return False

    def step(self, image, t) -> Snapshot:
        index, self.index = self.index, self.index+1
        small = self._small(image)
        self.buffer.append((index, t, small, image))
        snap = Snapshot(index, t, 'SEARCHING', '', None, None, None, 0.)

        answer = self.locator.poll(t)
        caught_up = self._apply(answer, image, snap) if answer is not None else False
        if self.tracker is not None and not caught_up:
            start = time.perf_counter()
            box = self._update_tracker(small)
            snap.track_ms = (time.perf_counter()-start)*1000
            if box is None:
                self.tracker, self.coast_t = None, t
                snap.events.append('tracker lost the view; coasting until the next VLM check')
            else:
                self.box = box
        if self.tracker is None and self.box is not None and t - self.coast_t > self.s.coast_s:
            self._drop(snap, f'tracker lost the view and no VLM re-seed within {self.s.coast_s:.1f}s -> track dropped')

        if not self.locator.busy:
            # Only a new target needs the "can it be picked up?" question; re-checks of a live track stay fast.
            self.locator.submit(image, t, index, classify=self.box is None or self.classified_t is None
                                or t-self.classified_t > self.s.reclassify_s)
            snap.vlm_submitted = True

        if self.box is not None and self.template is not None:
            hist = self._histogram(image, self.box)
            snap.appearance = float(cv2.compareHist(self.template, hist, cv2.HISTCMP_CORREL)) if hist is not None else 0.
            # Display only: a rug-dominated patch still "matches", and a real wrapper can fail it, so it drops nothing.

        self.history[index] = self.box
        for old in [k for k in self.history if k < index-self.s.buffer_frames]:
            del self.history[old]

        if self.box is None:
            snap.reason = 'waiting for VLM to find trash'
            return snap
        if self.tracker is None:
            snap.status, snap.reason = 'HOLD', 'tracker lost the view; waiting for the VLM to re-find it'
            snap.verified_age = t-self.verified_t
            return snap
        h, w = image.shape[:2]
        snap.box = [max(0., self.box[0]), max(0., self.box[1]), min(float(w), self.box[2]), min(float(h), self.box[3])]
        snap.verified_age = t-self.verified_t
        edges = self._edges(self.box, image.shape)
        if edges:
            self.clipped_t = t
        if snap.verified_age > self.s.verify_max_age:
            snap.status, snap.reason = 'HOLD', f'verification stale ({snap.verified_age:.1f}s)'
        elif {'top', 'bottom'} <= set(edges) or {'left', 'right'} <= set(edges):
            # Clipped on opposite sides: bigger than the view, so tilting the wrist cannot fix it.
            snap.status, snap.reason, snap.overfills = 'HOLD', 'fills the view (' + '/'.join(edges) + '): back up', True
        elif edges:
            snap.status, snap.reason = 'HOLD', 'clipped at ' + '/'.join(edges) + ' edge: recenter with wrist'
        elif self.clipped_t is not None and self.clipped_t >= self.verified_t:
            snap.status, snap.reason = 'HOLD', 'clipped since last VLM check: await fresh one'
        else:
            snap.status, snap.reason = 'DRIVE', 'verified, in view'
        return snap
