import asyncio
import time
from types import SimpleNamespace

from autonomy.control_server import ControlHub, JOINTS


def _hub(clear_answers):
    hub = ControlHub.__new__(ControlHub)
    hub.auto, hub.target, hub.target_seen = True, None, 0.
    hub.args = SimpleNamespace(turn_speed=400., turn_sign=-1., drive_sign=-1., wander_min=.3, wander_max=.3,
                               wander_speed=1., wander_check_s=.1)
    hub.sent, hub.notices, answers = [], [], iter(clear_answers)

    async def send(message): hub.sent.append(message)
    async def notice(message, error=False): hub.notices.append(message)
    async def path_clear(): return next(answers, True)
    hub.pi_send, hub.grab_notice, hub._path_clear = send, notice, path_clear
    return hub


def _forward_drives(hub):
    return [m for m in hub.sent if m['type'] == 'drive' and m['x']]


def test_wander_turns_then_drives_while_clear():
    hub = _hub([True] * 10)
    asyncio.run(hub._wander())
    assert any(m['type'] == 'drive' and m['theta'] for m in hub.sent)           # turned to a new heading
    assert _forward_drives(hub) and hub.sent[-1]['type'] == 'stop'             # drove, then stopped
    assert hub.mode == 'search'


def test_wander_does_not_drive_when_path_blocked_or_unknown():
    for answer in (False, None):
        hub = _hub([answer])
        asyncio.run(hub._wander())
        assert _forward_drives(hub) == [] and any('Wander stopped' in n for n in hub.notices)


def test_full_search_turn_triggers_wander():
    hub = ControlHub.__new__(ControlHub)
    hub.auto, hub.auto_distance_reached = True, False
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['status'] = 'connected'
    hub.home = dict.fromkeys(JOINTS, 0.)
    hub.state_seen = hub.frame_seen = time.monotonic()
    hub.target_seen = 0.                                                       # nothing seen: searching
    hub.frame_size, hub.target = (640, 360), None
    hub.stop_height, hub.auto_travel_speed, hub.auto_arm_speed = .6, 10., 10.
    hub.wrist_delta, hub.auto_wrist_stop_deg = 0., 43.1
    hub.search_turn_deg = 359.9                                               # almost a full turn already
    hub.args = SimpleNamespace(control_hz=100, lost_timeout=.7, speed=.1, turn_speed=8., min_turn_speed=2.,
                               wrist_gain=.03, patrol_speed=8., turn_sign=-1., drive_sign=-1., wrist_range=1.,
                               wrist_sign=1., auto_grab=False, wander_after_deg=360., auto_stall_timeout=6.,
                               steer_band=0., vertical_band=0., auto_joint_speed=100., search_grace=0., wrist_feedforward=0.)
    wanders = []

    async def send(message): pass
    async def status(decision=None): pass
    async def wander(): wanders.append(1); hub.auto = False
    hub._ik_ground_angle_deg = lambda pose: 10.
    hub.pi_send, hub.send_autonomy_status, hub._wander = send, status, wander

    async def run():
        try:
            await asyncio.wait_for(hub.auto_loop(), 1.)
        except asyncio.TimeoutError:
            pass
    asyncio.run(run())
    assert wanders == [1] and hub.search_turn_deg == 0.
