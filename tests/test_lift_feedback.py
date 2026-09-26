import asyncio
import time
import numpy as np
from autonomy.control_server import ControlHub
from autonomy.wrist_ik import WristIK


def make_hub(moving):
    hub = ControlHub.__new__(ControlHub)
    hub.wrist_ik = WristIK()
    hub.state = dict.fromkeys(hub.wrist_ik.names, 0.)
    hub.state['gripper'] = 0.
    hub.state_seen = time.monotonic()
    hub.grab_debug = {}
    sent, notices = [], []
    async def send(message):
        sent.append(message)
        if moving:
            for name in hub.state:
                hub.state[name] += .35*(message['values'][name] - hub.state[name])
        hub.state_seen = time.monotonic()
    async def notice(message, **kwargs):
        notices.append(message)
    async def status():
        pass
    hub.pi_send, hub.grab_notice, hub.send_autonomy_status = send, notice, status
    return hub, sent, notices


def test_transient_lag_waits_then_succeeds():
    hub, sent, notices = make_hub(True)
    pose = dict(hub.state, shoulder_lift=.15)
    assert np.linalg.norm(hub.wrist_ik.fk(pose)[:3, 3]-hub.wrist_ik.fk(hub.state)[:3, 3]) > .008
    assert asyncio.run(hub._wait_lift_settled(pose, lambda measured: None))
    assert len(sent) > 2
    assert not notices
    assert all(m['values']['gripper'] == 0 for m in sent)


def test_sustained_stall_holds_measured_pose():
    hub, sent, notices = make_hub(False)
    pose = dict(hub.state, shoulder_lift=.15)
    assert not asyncio.run(hub._wait_lift_settled(pose, lambda measured: None))
    assert len(sent) >= 3
    assert sent[-1]['values'] == hub.state
    assert 'stopped rising' in notices[-1]
    assert 'stalled' not in notices[-1].lower(), 'a stall must no longer read as a failure'


def test_stale_feedback_does_not_issue_motion():
    hub, sent, notices = make_hub(False)
    hub.state_seen -= 2
    assert not asyncio.run(hub._wait_lift_settled(dict(hub.state), lambda measured: None))
    assert not sent
    assert 'stale telemetry' in notices[-1]
