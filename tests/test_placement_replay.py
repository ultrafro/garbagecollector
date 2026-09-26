import asyncio
import time
import pytest
from types import SimpleNamespace
from autonomy.control_server import ControlHub, JOINTS


@pytest.mark.parametrize('carry_names', [['lift','over','in'], ['lift','over','in2'], ['lift','clear','over','in2']])
def test_placement_keeps_grip_until_release_and_waits(monkeypatch, carry_names):
    hub = ControlHub.__new__(ControlHub)
    hub.args = SimpleNamespace(grab_gripper=-1.2, place_speed=2., place_slow_keypoints={'release', 'out'})
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['status'] = 'connected'
    hub.state_seen = time.monotonic()
    names = carry_names + ['release','over-return','reset']
    hub.placement_keypoints = [dict(name=n, joints=dict.fromkeys(JOINTS, .1*i)) for i,n in enumerate(names)]
    release = hub.placement_keypoints[len(carry_names)]['joints']['gripper']
    hub.placement_keypoints[-2]['joints'] = dict(hub.placement_keypoints[1]['joints'], gripper=release)
    commands, waits = [], []
    async def send(message):
        if message['type']=='joints':
            commands.append((hub.grab_phase, message['values']))
            hub.state.update(message['values'])
            hub.state_seen = time.monotonic()
    async def noop(*args, **kwargs): pass
    async def hold(pose, duration, phase): waits.append((duration, phase))
    monkeypatch.setattr(asyncio, 'sleep', noop)
    hub.pi_send, hub.grab_notice, hub._grab_pose = send, noop, hold
    asyncio.run(hub.place_with_keypoints())
    assert all(p['gripper']==-1.2 for phase,p in commands if phase in ['placement-'+n for n in carry_names])
    assert commands[-1][0]=='placement-reset'
    assert commands[-1][1]==hub.placement_keypoints[-1]['joints']
    phases = list(dict.fromkeys(phase for phase,p in commands))
    assert phases == ['placement-'+n for n in names]
    returned = [p for phase,p in commands if phase == 'placement-over-return']
    assert all(abs(p['gripper']-release)<1e-6 for p in returned)
    assert returned[-1] == pytest.approx(hub.placement_keypoints[-2]['joints'])
    assert waits == []
