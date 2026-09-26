import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from autonomy.control_server import ControlHub, JOINTS
from autonomy.wrist_ik import WristIK

GRIP = {'shoulder_pan': 0., 'shoulder_lift': .3, 'elbow_flex': 1.0, 'wrist_flex': -.45, 'wrist_roll': 0., 'gripper': -.8}


def _check(half_box, top_box, held_answer, bottom_answer, tmp_path, grip=-.8):
    hub = ControlHub.__new__(ControlHub)
    hub.wrist_ik = WristIK()
    hub.state = dict(GRIP, status='connected')
    hub.state_seen = time.monotonic()
    hub.motion_dir = Path(tmp_path) / 'motions'
    hub.args = SimpleNamespace(grab_verify_height=.12, grab_lift_speed=10., grasp_parallax_min=.6,
                               grasp_empty_gripper=-.8, grasp_gripper_margin=.25)
    hub.grip_reading = (grip, 50.)
    frame = cv2.imencode('.jpg', np.zeros((360, 640, 3), np.uint8))[1].tobytes()
    boxes = iter([half_box, top_box])
    notices = []

    async def send(message):
        if message['type'] == 'joints':
            hub.state.update(message['values'])
            hub.state_seen = time.monotonic()
            hub.latest_jpeg = (time.monotonic() + 1., frame)       # every arm command yields a newer frame

    async def notice(message, error=False): notices.append(message)

    def locate(image):
        box = next(boxes)
        return {'boxes': [{'box': box}] if box else []}

    def ask(images, prompt, tokens):
        return (bottom_answer if 'bottom strip' in prompt else held_answer), 1

    hub.latest_jpeg = (0., frame)
    hub.pi_send, hub.grab_notice = send, notice
    hub.vlm_client = SimpleNamespace(locate=locate, ask=ask)
    held = asyncio.run(hub._verify_grasp(dict(GRIP)))
    return held, notices[-1]


def test_same_size_box_and_both_questions_held(tmp_path):
    held, message = _check([200, 250, 440, 360], [210, 240, 450, 360], 'HELD', 'YES', tmp_path)
    assert held is True and 'parallax=held' in message


def test_box_shrinks_on_the_floor_and_questions_empty(tmp_path):
    held, message = _check([200, 100, 440, 300], [290, 150, 350, 210], 'EMPTY', 'NO', tmp_path)
    assert held is False and 'parallax=empty' in message


def test_box_left_behind_outvotes_one_optimistic_question(tmp_path):
    held, _ = _check([200, 100, 440, 300], None, 'HELD', 'NO', tmp_path)   # gone at the top: not held
    assert held is False


def test_snapshots_saved_for_labelling(tmp_path):
    _check([200, 250, 440, 360], [210, 240, 450, 360], 'HELD', 'YES', tmp_path)
    saved = list((Path(tmp_path) / 'grasp-checks').glob('*/check.json'))
    assert saved and (saved[0].parent / 'top.jpg').exists()


def test_jaws_held_open_count_as_held_even_if_the_camera_sees_nothing(tmp_path):
    held, message = _check(None, None, 'EMPTY', 'NO', tmp_path, grip=-.2)     # e.g. a lemon below the frame
    assert held is True and 'gripper=held' in message and 'gripper stopped at -0.20' in message


def test_normal_close_leaves_the_decision_to_the_camera(tmp_path):
    held, message = _check([200, 100, 440, 300], [290, 150, 350, 210], 'EMPTY', 'NO', tmp_path, grip=-.79)
    assert held is False and 'gripper=-' in message
