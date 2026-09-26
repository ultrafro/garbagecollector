import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from autonomy.control_server import ControlHub
from autonomy.wrist_ik import WristIK

GRIP = {'shoulder_pan': 0., 'shoulder_lift': .3, 'elbow_flex': 1.0, 'wrist_flex': -.45, 'wrist_roll': 0., 'gripper': -.8}


def _check(held_answer, near_answer, tmp_path, grip=-.8):
    hub = ControlHub.__new__(ControlHub)
    hub.wrist_ik = WristIK()
    hub.state = dict(GRIP, status='connected')
    hub.state_seen = time.monotonic()
    hub.motion_dir = Path(tmp_path) / 'motions'
    hub.args = SimpleNamespace(grab_verify_height=.12, grab_lift_speed=10., grasp_empty_gripper=-.8,
                               grasp_gripper_margin=.08)
    hub.grip_reading = (grip, 50.)
    frame = cv2.imencode('.jpg', np.zeros((360, 640, 3), np.uint8))[1].tobytes()
    notices = []

    async def send(message):
        if message['type'] == 'joints':
            hub.state.update(message['values'])
            hub.state_seen = time.monotonic()
            hub.latest_jpeg = (time.monotonic() + 1., frame)       # every arm command yields a newer frame

    async def notice(message, error=False): notices.append(message)

    def ask(images, prompt, tokens):
        return (near_answer if 'any edge or corner' in prompt else held_answer), 1

    hub.latest_jpeg = (0., frame)
    hub.pi_send, hub.grab_notice = send, notice
    hub.vlm_client = SimpleNamespace(ask=ask)
    held = asyncio.run(hub._verify_grasp(dict(GRIP)))
    return held, notices[-1]


def test_either_camera_question_saying_held_is_enough(tmp_path):
    assert _check('HELD', 'NO', tmp_path)[0] is True          # the sushi-wrapper case: one question sees it
    assert _check('EMPTY', 'YES', tmp_path)[0] is True


def test_both_questions_empty_and_jaws_closed_is_a_miss(tmp_path):
    held, message = _check('EMPTY', 'NO', tmp_path)
    assert held is False and 'gripper=-' in message


def test_jaws_held_open_count_as_held_even_if_the_camera_sees_nothing(tmp_path):
    held, message = _check('EMPTY', 'NO', tmp_path, grip=-.2)      # e.g. a lemon below the frame
    assert held is True and 'gripper=held' in message and 'gripper stopped at -0.20' in message


def test_slightly_open_jaws_from_wrapper_film_count_as_held(tmp_path):
    assert _check('EMPTY', 'NO', tmp_path, grip=-.69)[0] is True


def test_snapshot_saved_for_labelling(tmp_path):
    _check('HELD', 'YES', tmp_path)
    saved = list((Path(tmp_path) / 'grasp-checks').glob('*/check.json'))
    assert saved and (saved[0].parent / 'top.jpg').exists()
