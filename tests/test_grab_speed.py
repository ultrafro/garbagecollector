"""Exercise the real dive loop with feedback slower than command updates."""
import asyncio
import time
from types import SimpleNamespace
import numpy as np
from autonomy.control_server import ControlHub, JOINTS


def test_speed_changes_command_rate_between_feedback_packets():
    async def run(speed):
        hub = ControlHub.__new__(ControlHub)
        hub.auto = False
        hub.state = dict.fromkeys(JOINTS, 0.)
        hub.state['status'] = 'connected'
        hub.state_seen = time.monotonic()
        hub.target = SimpleNamespace(center=(320, 180), x1=200, x2=440)
        hub.target_seen = hub.state_seen
        hub.frame_seen = hub.state_seen
        hub.frame_size = (640, 360)
        hub.grab_debug = None
        hub.grab_dive_log = []
        hub.contact_threshold = 140.
        hub.ground_offset_cm = 0.
        hub.grab_speed = speed
        hub.args = SimpleNamespace(grab_track_hold=1.5, grab_open=1.2,
            grab_ground_z=-1., grab_approach_step=.0015, grab_approach_period=.1,
            grab_joint_slew=.015, grab_pan_sign=-1., wrist_sign=1., grab_max_reaim=.35,
            grab_servo_gain=.5, grab_elevation_weight=.5, camera_hfov=90., camera_vfov=60.)
        stamps = []
        def fk(pose):
            t = np.eye(4)
            t[:3, :3] = np.diag([1., -1., -1.])
            t[2, 3] = .1 - pose['shoulder_lift']
            return t
        def solve(seed, target, **kwargs):
            stamps.append(time.monotonic())
            if len(stamps) == 4:
                raise asyncio.CancelledError
            return dict(seed, shoulder_lift=.1-target[2])
        hub.wrist_ik = SimpleNamespace(fk=fk, capture=lambda p:(fk(p),fk(p)[:3,2]), solve_position=solve)
        async def noop(*args, **kwargs): pass
        async def opened(*args, **kwargs): return True
        hub.pi_send = hub.grab_notice = hub.send_autonomy_status = noop
        hub._grab_pose = opened
        hub._write_dive_log = lambda rows: ''
        try:
            await asyncio.wait_for(hub.grab_routine(), 1.3)
        except asyncio.CancelledError:
            pass
        return stamps[-1]-stamps[0]
    slow = asyncio.run(run(10))
    fast = asyncio.run(run(50))
    assert fast < slow * .6, (slow, fast)
