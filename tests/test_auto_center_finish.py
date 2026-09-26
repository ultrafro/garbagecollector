import asyncio
import time
from types import SimpleNamespace
from autonomy.control_server import ControlHub, JOINTS
from autonomy.tracking import Detection


def test_distance_latches_while_wrist_centers():
    hub = ControlHub.__new__(ControlHub)
    hub.auto = True
    hub.auto_distance_reached = False
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['status'] = 'connected'
    hub.home = dict.fromkeys(JOINTS, 0.)
    hub.state_seen = hub.frame_seen = hub.target_seen = time.monotonic()
    hub.frame_size = (640,360)
    hub.target = Detection('wrapper', .9, 270,200,370,300)
    hub.stop_height = .6
    hub.auto_travel_speed = hub.auto_arm_speed = 10.
    hub.wrist_delta = 0.
    hub.auto_wrist_stop_deg = 43.1
    hub.args = SimpleNamespace(control_hz=1000, lost_timeout=10., speed=.1,
        turn_speed=8., min_turn_speed=2., wrist_gain=.03, patrol_speed=4.,
        turn_sign=-1., drive_sign=-1., wrist_range=1., wrist_sign=1., auto_grab=False, steer_band=0., vertical_band=0., auto_joint_speed=100., search_grace=0., wrist_feedforward=0.)
    angle = [50.]
    hub._ik_ground_angle_deg = lambda pose: angle[0]
    commands=[]
    async def send(message): commands.append(message)
    count=[0]
    async def status(decision):
        count[0]+=1
        assert decision['forward_m_s']==0
        if count[0] <= 2:
            assert hub.auto and hub.mode=='center-vertical'
            assert [m for m in commands if m['type']=='joints'][-1]['values']['wrist_flex']>0
            angle[0]=35. # Once close, falling below threshold must not restart travel.
            if count[0]==2: hub.target=Detection('wrapper',.9,270,130,370,230)
        else:
            assert not hub.auto and hub.mode=='manual'
            raise asyncio.CancelledError
    hub.pi_send, hub.send_autonomy_status = send,status
    async def run():
        try: await hub.auto_loop()
        except asyncio.CancelledError: pass
    asyncio.run(run())
    assert count[0]==3


def _finished_approach_hub(auto_grab, grab_running):
    """A hub whose next auto tick completes the approach (centered target, wrist past the stop angle)."""
    hub = ControlHub.__new__(ControlHub)
    hub.auto, hub.auto_distance_reached = True, True
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['status'] = 'connected'
    hub.home = dict.fromkeys(JOINTS, 0.)
    hub.state_seen = hub.frame_seen = hub.target_seen = time.monotonic()
    hub.frame_size = (640, 360)
    hub.target = Detection('trash', 1., 270, 130, 370, 230)
    hub.stop_height, hub.auto_travel_speed, hub.auto_arm_speed = .6, 10., 10.
    hub.wrist_delta, hub.auto_wrist_stop_deg = 0., 43.1
    hub.args = SimpleNamespace(control_hz=1000, lost_timeout=10., speed=.1, turn_speed=8., min_turn_speed=2.,
                               wrist_gain=.03, patrol_speed=4., turn_sign=-1., drive_sign=-1., wrist_range=1.,
                               wrist_sign=1., auto_grab=auto_grab, steer_band=0., vertical_band=0., auto_joint_speed=100., search_grace=0., wrist_feedforward=0.)
    hub._ik_ground_angle_deg = lambda pose: 50.
    grabs = []

    async def send(message): pass
    async def notice(message, error=False): pass
    async def start_grab(resume_auto=False): grabs.append(resume_auto)
    async def status(decision=None): pass
    hub.pi_send, hub.send_autonomy_status, hub.grab_notice, hub.start_grab = send, status, notice, start_grab

    async def run():
        if grab_running:
            hub.grab_task = asyncio.create_task(asyncio.sleep(10))
        try:
            await asyncio.wait_for(hub.auto_loop(), .5)     # the loop never returns; a few hundred ticks is plenty
        except asyncio.TimeoutError:
            pass
        if hub.grab_task:
            hub.grab_task.cancel()
    asyncio.run(run())
    return hub, grabs


def test_completed_approach_starts_one_grab():
    hub, grabs = _finished_approach_hub(auto_grab=True, grab_running=False)
    assert not hub.auto and grabs == [True]     # auto-started grabs resume auto afterwards


def test_no_grab_when_disabled_or_retry_already_running():
    assert _finished_approach_hub(auto_grab=False, grab_running=False)[1] == []
    assert _finished_approach_hub(auto_grab=True, grab_running=True)[1] == []


def _run_grab(results, resume_auto):
    """Run grab_with_retries against scripted grab_routine results; return the hub afterwards."""
    hub = ControlHub.__new__(ControlHub)
    hub.auto, hub.vlm_client, hub.grab_attempt = False, object(), 0
    hub.args = SimpleNamespace(grab_attempts=3, grab_scan_cycles=1, grab_retry_backup=0., grab_retry_timeout=1., place_backup=.1)
    script = iter(results)

    async def routine(): return next(script)
    async def notice(message, error=False): pass
    async def nothing(*args): return True
    backups = hub.backups = []
    async def back_up(metres): backups.append(metres)
    hub.grab_routine, hub.grab_notice, hub._back_up, hub._reapproach = routine, notice, back_up, nothing
    asyncio.run(hub.grab_with_retries(resume_auto))
    return hub


def test_auto_resumes_only_after_an_auto_started_grab_places_trash():
    hub = _run_grab(["missed", "placed"], resume_auto=True)
    assert hub.auto and hub.backups[-1] == .1                    # backs up after placing, then resumes
    manual = _run_grab(["placed"], resume_auto=False)
    assert not manual.auto and manual.backups == []               # manual Test grab: stay put, stay in manual
    assert not _run_grab([None], resume_auto=True).auto            # refused/failed grab: do not resume


def test_auto_arm_moves_are_rate_limited_from_the_current_pose():
    hub = ControlHub.__new__(ControlHub)
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['wrist_flex'] = 1.0                                   # arm left tilted by the bin routine
    hub.args = SimpleNamespace(auto_joint_speed=.8)
    hub._start_auto = ControlHub._start_auto.__get__(hub)
    hub._start_auto()
    home = dict.fromkeys(JOINTS, 0.)
    first = hub._slew_auto_joints(home, .1)
    assert abs(first['wrist_flex'] - .92) < 1e-9                    # moved 0.08 rad (0.8 rad/s x 0.1 s), not straight home
    for _ in range(20):
        last = hub._slew_auto_joints(home, .1)
    assert last['wrist_flex'] == 0.                                  # arrives, then holds the target


def _lost_target_hub(lost_for):
    hub = ControlHub.__new__(ControlHub)
    hub.auto, hub.auto_distance_reached = True, False
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['status'] = 'connected'
    hub.home = dict.fromkeys(JOINTS, 0.)
    hub.state_seen = hub.frame_seen = time.monotonic()
    hub.target_seen = time.monotonic() - lost_for
    hub.frame_size, hub.target = (640, 360), None
    hub.stop_height, hub.auto_travel_speed, hub.auto_arm_speed = .6, 10., 10.
    hub.wrist_delta, hub.auto_wrist_stop_deg = .3, 43.1
    hub.args = SimpleNamespace(control_hz=1000, lost_timeout=.7, speed=.1, turn_speed=8., min_turn_speed=2., wrist_gain=.03,
                               patrol_speed=8., turn_sign=-1., drive_sign=-1., wrist_range=1., wrist_sign=1., auto_grab=False,
                               steer_band=0., vertical_band=0., auto_joint_speed=100., search_grace=2.5, wander_after_deg=0., wrist_feedforward=0.,
                               auto_stall_timeout=6.)
    hub._ik_ground_angle_deg = lambda pose: 10.
    drives = []

    async def send(message):
        if message['type'] == 'drive':
            drives.append(message)
    async def status(decision=None):
        if len(drives) >= 3:
            raise asyncio.CancelledError
    hub.pi_send, hub.send_autonomy_status = send, status

    async def run():
        try:
            await asyncio.wait_for(hub.auto_loop(), 1.)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    asyncio.run(run())
    return hub, drives


def test_briefly_lost_target_holds_wrist_and_does_not_rotate():
    hub, drives = _lost_target_hub(lost_for=1.)
    assert all(d['theta'] == 0 for d in drives) and hub.wrist_delta == .3


def test_long_lost_target_searches_and_relaxes_wrist():
    hub, drives = _lost_target_hub(lost_for=5.)
    assert any(d['theta'] != 0 for d in drives) and hub.wrist_delta < .3


def _approach_ticks(vlm_status):
    hub = ControlHub.__new__(ControlHub)
    hub.auto, hub.auto_distance_reached = True, False
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['status'] = 'connected'
    hub.home = dict.fromkeys(JOINTS, 0.)
    hub.state_seen = hub.frame_seen = hub.target_seen = time.monotonic()
    hub.frame_size = (640, 360)
    hub.target = Detection('trash', 1., 300, 160, 340, 200)            # dead centre: no error correction
    hub.stop_height, hub.auto_travel_speed, hub.auto_arm_speed = .6, 10., 10.
    hub.wrist_delta, hub.auto_wrist_stop_deg = 0., 43.1
    hub.targeter = object()
    hub.vlm = SimpleNamespace(status=vlm_status, reason='test', overfills=False)
    hub.auto_progress_t = time.monotonic()
    hub.args = SimpleNamespace(control_hz=50, lost_timeout=10., speed=.1, turn_speed=8., min_turn_speed=2., wrist_gain=.03,
                               patrol_speed=8., turn_sign=-1., drive_sign=-1., wrist_range=1., wrist_sign=1., auto_grab=False,
                               steer_band=.25, vertical_band=.35, auto_joint_speed=100., search_grace=2.5, wander_after_deg=0.,
                               auto_stall_timeout=60., wrist_feedforward=2.)
    hub._ik_ground_angle_deg = lambda pose: 10.
    ticks = []

    async def send(message): pass
    async def status(decision=None):
        ticks.append(1)
        hub.state_seen = hub.frame_seen = hub.target_seen = time.monotonic()
        hub.state['wrist_flex'] = hub.wrist_delta                      # arm keeps up with its command
        if len(ticks) >= 10:
            raise asyncio.CancelledError
    hub.pi_send, hub.send_autonomy_status = send, status

    async def run():
        try:
            await asyncio.wait_for(hub.auto_loop(), 2.)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    asyncio.run(run())
    return hub


def test_wrist_tilts_ahead_while_driving_forward():
    assert _approach_ticks('DRIVE').wrist_delta > 0                     # target centred, yet the wrist leads the approach


def test_no_feedforward_tilt_while_held():
    assert _approach_ticks('HOLD').wrist_delta == 0                     # base stopped by HOLD: nothing to anticipate
