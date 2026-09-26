"""Replay recorded dives through the load contact detector.

The fixture holds every arm joint's present-load series from five guarded wrist
-ray probes (autonomy/load_probe.py) plus one real grab dive. Two probes ran
through open air for their whole travel; the rest ended loaded up, which is the
signature the detector has to catch.
"""
import json
from pathlib import Path

import pytest

from autonomy.contact import ARM_MM, DEFAULT_THRESHOLD, JOINTS, WATCHED, LoadContactDetector

PROBES = json.loads((Path(__file__).parent / 'fixtures' / 'load_probes.json').read_text())
WATCHED_INDEX = JOINTS.index(WATCHED)


def probe(name):
    return next(p for p in PROBES if p['name'] == name)


# Grouped by what the wrist_flex detector should do with each recorded run:
#   CLEAN_REACH   - moving through open air; must never fire
#   ELBOW_ONLY    - loaded against the reach limit (elbow/floor), not wrist_flex;
#                   must not fire, because wrist_flex barely moves
#   WRIST_CONTACT - the real dig-in that loads wrist_flex; must fire at default
#   HIGH_TORQUE   - a pose whose sustained movement load overlaps contact and
#                   cannot be separated by a wrist_flex threshold (documented)
CLEAN_REACH = ['reach-only-214mm', 'free-travel-80mm', 'free-travel-80mm-b']
ELBOW_ONLY = ['contact-116mm-39deg', 'contact-94mm-60deg', 'contact-92mm-60deg']
WRIST_CONTACT = ['ground-contact-134mm']
HIGH_TORQUE = ['free-motion-dive-22mm', 'real-dive-30mm']


def replay(probe, **kwargs):
    """Return the travel distance where contact fired, or None."""
    detector = LoadContactDetector(**kwargs)
    for mm, loads in zip(probe['mm'], probe['loads']):
        if detector.update(loads, mm):
            return mm
    return None


@pytest.mark.parametrize('name', CLEAN_REACH)
def test_clean_reach_never_reports_contact(name):
    assert replay(probe(name)) is None


@pytest.mark.parametrize('name', ELBOW_ONLY)
def test_elbow_or_floor_contact_does_not_fire_on_wrist_flex(name):
    """These loaded the elbow against the reach limit; wrist_flex barely moves,
    so a wrist_flex detector correctly stays silent."""
    assert replay(probe(name)) is None


@pytest.mark.parametrize('name', WRIST_CONTACT)
def test_real_wrist_contact_fires_at_the_default(name):
    """The ground dig-in that this threshold was calibrated to catch."""
    assert replay(probe(name)) is not None


def test_only_the_watched_joint_can_stop_the_dive():
    """Regression: a good dive was stopped by wrist_roll drifting to 40.

    wrist_roll read exactly 0 in every probe, so it had no measured noise to set
    a threshold from. Only WATCHED may fire; the rest are recorded, not armed.
    """
    detector = LoadContactDetector(threshold=32.)
    for _ in range(10):
        loads = [400.] * len(JOINTS)
        loads[WATCHED_INDEX] = 0.
        assert detector.update(loads, 100.) is None
    assert detector.debug()['contact_loads']['wrist_roll'] == 400.


def test_contact_is_detected_on_negative_load():
    """decode_load is signed; pressing the joint the other way must still fire."""
    detector = LoadContactDetector(threshold=72.)
    fired = None
    for _ in range(4):
        loads = [0.] * len(JOINTS)
        loads[WATCHED_INDEX] = -80.
        fired = fired or detector.update(loads, 100.)
    assert fired == WATCHED


def test_the_breakaway_transient_cannot_fire():
    """Leaving rest costs more than light contact, and is over within ~10 mm."""
    detector = LoadContactDetector(threshold=32.)
    for mm in (0., 2., 4., 6., 8., 10., 12., 14.):
        loads = [0.] * len(JOINTS)
        loads[WATCHED_INDEX] = 400.
        assert detector.update(loads, mm) is None
    assert not detector.armed
    assert detector.update([400.] * len(JOINTS), ARM_MM) is None   # armed, first sample
    assert detector.update([400.] * len(JOINTS), ARM_MM + 2) is None
    assert detector.update([400.] * len(JOINTS), ARM_MM + 4) == WATCHED


@pytest.mark.parametrize('name', HIGH_TORQUE)
def test_high_torque_pose_is_a_known_wrist_flex_limitation(name):
    """Honest guard: in a high-torque pose, sustained movement load overlaps
    contact. No wrist_flex threshold separates it -- this documents why the
    default is pose-specific and why shoulder_lift is the cleaner signal there.
    """
    assert replay(probe(name)) is not None


def test_a_sustained_floor_press_fires():
    """The live failure that started this: wrist_flex held at -60 on the floor."""
    detector = LoadContactDetector(threshold=44.)
    fired = None
    for mm in range(0, 60, 2):
        loads = [0.] * len(JOINTS)
        loads[WATCHED_INDEX] = -60. if mm >= 20 else -20.
        fired = fired or detector.update(loads, float(mm))
    assert fired == WATCHED


def test_threshold_can_be_changed_mid_dive():
    """The control page slider applies immediately, not on the next grab."""
    detector = LoadContactDetector(threshold=400.)
    loads = [0.] * len(JOINTS)
    loads[WATCHED_INDEX] = 80.
    for _ in range(5):
        assert detector.update(loads, 100.) is None
    fired = None
    for _ in range(3):
        fired = fired or detector.update(loads, 100., threshold=72.)
    assert fired == WATCHED


def test_detector_ignores_missing_samples():
    detector = LoadContactDetector()
    assert detector.update(None, 100.) is None
    assert detector.update([], 100.) is None


def test_contact_requires_consecutive_samples():
    detector = LoadContactDetector(threshold=72., samples=3, smoothing=1)
    hot = [0.] * len(JOINTS); hot[WATCHED_INDEX] = 80.
    cold = [0.] * len(JOINTS); cold[WATCHED_INDEX] = 8.
    assert not detector.update(hot, 100.)
    assert not detector.update(cold, 100.)    # dropping back resets the run
    assert not detector.update(hot, 100.)
    assert not detector.update(hot, 100.)
    assert detector.update(hot, 100.) == WATCHED


def test_open_gripper_waits_for_a_slow_servo_instead_of_a_fixed_hold():
    """A gripper crossing its full range needs longer than the nominal hold.

    Regression: the fixed .8 s wait reported an arriving gripper as an endpoint
    calibration fault, aborting the grab before the dive ever started.
    """
    import asyncio
    import time
    from types import SimpleNamespace
    from autonomy.control_server import ControlHub

    hub = ControlHub.__new__(ControlHub)
    hub.state = {'gripper': 0.}
    hub.state_seen = time.monotonic()
    hub.grab_phase = None

    async def noop(*args, **kwargs):
        pass
    hub.grab_notice = noop

    started = None

    async def send(message):
        nonlocal started
        started = started or time.monotonic()
        # Servo needs ~1.5 s to cross the range - well past the .8 s hold.
        travelled = min(1., (time.monotonic() - started) / 1.5)
        hub.state['gripper'] = 1.2 * travelled
        hub.state_seen = time.monotonic()
    hub.pi_send = send

    arrived = asyncio.run(hub._grab_pose({'gripper': 1.2}, .8, 'open-gripper', arrive='gripper'))
    assert arrived
    assert hub.state['gripper'] == pytest.approx(1.2, abs=.12)


def test_open_gripper_still_fails_when_the_servo_never_arrives():
    import asyncio
    import time
    from autonomy.control_server import ControlHub

    hub = ControlHub.__new__(ControlHub)
    hub.state = {'gripper': .2}
    hub.state_seen = time.monotonic()
    hub.grab_phase = None

    async def noop(*args, **kwargs):
        pass
    hub.grab_notice = noop

    async def send(message):
        hub.state_seen = time.monotonic()   # fresh telemetry, stuck gripper
    hub.pi_send = send

    assert not asyncio.run(hub._grab_pose({'gripper': 1.2}, .2, 'open-gripper', arrive='gripper', timeout=.6))


def test_closing_grip_does_not_require_arrival():
    """Closing on an object legitimately stops short, so it must not gate."""
    import asyncio
    import time
    from autonomy.control_server import ControlHub

    hub = ControlHub.__new__(ControlHub)
    hub.state = {'gripper': .9}
    hub.state_seen = time.monotonic()
    hub.grab_phase = None

    async def noop(*args, **kwargs):
        pass
    hub.grab_notice = noop

    async def send(message):
        hub.state_seen = time.monotonic()
    hub.pi_send = send

    assert asyncio.run(hub._grab_pose({'gripper': 0.}, .2, 'grip'))


def test_a_lagging_arm_never_aborts_the_grab():
    """Load is the only thing that stops the dive.

    A servo that trails the command used to trip a joint tracking fault and
    abandon the grab without ever closing the gripper. Lag, tracking error and
    an exhausted ray now just end the dive where it is and clamp.
    """
    import asyncio
    import time
    from types import SimpleNamespace
    import numpy as np
    from autonomy.control_server import ControlHub
    from autonomy.wrist_ik import WristIK

    hub = ControlHub.__new__(ControlHub)
    hub.wrist_ik = WristIK()
    hub.state = dict.fromkeys(hub.wrist_ik.names, 0.)
    hub.state['gripper'] = .3
    hub.state['servo_load'] = [0.] * 6
    hub.state_seen = time.monotonic()
    hub.home = hub.target = None
    hub.auto = False
    hub.grab_debug = None
    hub.contact_threshold = 72.
    hub.grab_target_width = .95
    hub.grab_ik_tolerance = .001
    hub.frame_size = (0, 0)
    hub.args = SimpleNamespace(grab_open=1.2, grab_max_reaim=np.deg2rad(12),
                               grab_max_travel=.25, grab_gripper=0., grab_grip_time=.2, grab_approach_step=.004, grab_approach_period=.0,
                               grab_ground_z=-0.0635)

    commands = []
    # A servo that only ever covers half the commanded step, so the commanded
    # pose pulls away from the measured one unless the dive waits.
    async def send(message):
        commands.append(message)
        if message['type'] == 'joints':
            for name, value in message['values'].items():
                hub.state[name] = hub.state.get(name, 0.) + (value - hub.state.get(name, 0.)) * .5
        hub.state_seen = time.monotonic()
    hub.pi_send = send

    async def grab_pose(pose, duration, phase, **kwargs):
        hub.state.update(pose)
        await send({'type': 'joints', 'values': pose})
        return True
    hub._grab_pose = grab_pose

    notices = []
    async def notice(message, error=False):
        notices.append((error, message))
    hub.grab_notice = notice

    async def noop(*args, **kwargs):
        pass
    hub.send_autonomy_status = noop
    hub._write_dive_log = lambda rows: ''

    async def run():
        async def telemetry():
            while True:
                hub.state_seen = time.monotonic()
                await asyncio.sleep(.01)
        task = asyncio.create_task(telemetry())
        try:
            await asyncio.wait_for(hub.grab_routine(), timeout=25)
        except asyncio.TimeoutError:
            pass
        finally:
            task.cancel()
    asyncio.run(run())

    assert not [m for error, m in notices if error], f'lag must not fail the grab: {notices}'
    assert any('dive stopped' in m for _, m in notices), 'the dive should report where it stopped'
    # It must still have closed the gripper on the way out.
    grips = [c for c in commands if c['type'] == 'joints'
             and c['values'].get('gripper') == pytest.approx(0., abs=1e-9)]
    assert grips, 'the gripper must close even when the arm lagged the whole dive'


def _dive_hub(tmp_path_unused=None):
    """A ControlHub wired to a simulated arm, enough to run grab_routine."""
    import time
    from types import SimpleNamespace
    import numpy as np
    from autonomy.control_server import ControlHub
    from autonomy.wrist_ik import WristIK

    hub = ControlHub.__new__(ControlHub)
    hub.wrist_ik = WristIK()
    hub.state = dict(zip(hub.wrist_ik.names, [-.0046, -1.2149, 1.4205, .1672, -.0138]))
    hub.state['gripper'] = .3
    hub.state['servo_load'] = [10., 20., 30., 40., 0., 5.]
    hub.state['servo_current'] = [1., 2., 3., 4., 5., 6.]
    hub.state_seen = time.monotonic()
    hub.home = hub.target = None
    hub.auto = False
    hub.grab_debug = None
    hub.grab_dive_log = []
    hub.contact_threshold = 112.
    hub.grab_target_width = .95
    hub.grab_ik_tolerance = .001
    hub.frame_size = (0, 0)
    hub.args = SimpleNamespace(grab_open=1.2, grab_max_reaim=np.deg2rad(12),
                               grab_max_travel=.25, grab_gripper=0., grab_grip_time=.2, grab_approach_step=.004, grab_approach_period=.0,
                               grab_ground_z=-0.0635)

    async def send(message):
        if message['type'] == 'joints':
            hub.state.update(message['values'])
        hub.state_seen = time.monotonic()
    hub.pi_send = send

    async def grab_pose(pose, duration, phase, **kwargs):
        hub.state.update(pose)
        await send({'type': 'joints', 'values': pose})
        return True
    hub._grab_pose = grab_pose

    async def noop(*args, **kwargs):
        pass
    hub.grab_notice = noop
    hub.send_autonomy_status = noop
    return hub


def _dive_logs():
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / 'screenshots'
    return set(folder.glob('grab-dive-*.csv'))


def test_a_completed_dive_writes_a_log_with_the_calibration_columns():
    import asyncio
    import csv
    before = _dive_logs()
    hub = _dive_hub()

    async def run():
        async def telemetry():
            while True:
                hub.state_seen = __import__('time').monotonic()
                await asyncio.sleep(.01)
        task = asyncio.create_task(telemetry())
        try:
            await asyncio.wait_for(hub.grab_routine(), timeout=40)
        except asyncio.TimeoutError:
            pass
        finally:
            task.cancel()
    asyncio.run(run())

    written = _dive_logs() - before
    assert written, 'a completed dive must leave a log'
    rows = list(csv.DictReader(max(written, key=lambda p: p.stat().st_mtime).open()))
    assert rows
    for column in ('t_s', 'mm', 'measured_mm', 'xyz_error_mm', 'threshold', 'armed', 'run', 'fired',
                   'load_wrist_flex', 'load_elbow_flex', 'current_wrist_flex'):
        assert column in rows[0], f'{column} missing; needed to calibrate'
    for path in written:
        path.unlink()


def test_a_cancelled_dive_still_writes_its_log():
    """STOP mid-dive is exactly when the telemetry matters most.

    Regression: the log was written only on a clean loop exit, so pressing STOP
    because the dive was going wrong discarded the run that would explain it.
    """
    import asyncio
    before = _dive_logs()
    hub = _dive_hub()

    async def run():
        async def telemetry():
            while True:
                hub.state_seen = __import__('time').monotonic()
                await asyncio.sleep(.01)
        beat = asyncio.create_task(telemetry())
        grab = asyncio.create_task(hub.grab_routine())
        await asyncio.sleep(1.2)          # let the dive take some samples
        grab.cancel()                      # this is STOP
        try:
            await grab
        except asyncio.CancelledError:
            pass
        beat.cancel()
    asyncio.run(run())

    written = _dive_logs() - before
    assert written, 'STOP must still leave a log'
    for path in written:
        path.unlink()


def test_moving_average_fires_on_the_ground_contact_run_but_not_reach_only():
    """Calibrated on the real dives: MA3 + default threshold catches the
    ground dig-in and stays silent on the pure-reach run.

    Raw wrist_flex had no working threshold -- 72 missed the brief 80-unit
    contact peak, lower risked the 60-unit free-travel noise. Averaging three
    samples opens a real window; this pins it to the recorded runs.
    """
    from autonomy.contact import LoadContactDetector, DEFAULT_THRESHOLD, WATCHED, JOINTS
    W = JOINTS.index(WATCHED)

    def replay(probe):
        d = LoadContactDetector()          # defaults: MA3, threshold 64, arm 16mm
        for mm, loads in zip(probe['mm'], probe['loads']):
            if d.update(loads, mm):
                return mm
        return None

    ground = probe('ground-contact-134mm')
    reach = probe('reach-only-214mm')
    assert replay(ground) is not None, 'must catch the ground dig-in'
    assert replay(reach) is None, 'must not fire on a pure reach through open air'


def test_moving_average_rejects_a_moderate_isolated_spike_but_not_a_sustained_load():
    """A moderate one-sample spike is averaged away; a sustained load fires.

    With MA3 and the 3-consecutive rule a lone spike fills three windows at
    (spike + 2*baseline)/3, so it only rejects spikes below ~3*threshold. A
    120-unit spike over a 20 baseline averages to 53 < 64 and is ignored; a
    sustained 90 clears it. (A very large spike -- the 152/160 seen in a
    high-torque pose -- is not rejected; see the module docstring.)
    """
    from autonomy.contact import LoadContactDetector, WATCHED, JOINTS
    W = JOINTS.index(WATCHED)
    d = LoadContactDetector(threshold=64., smoothing=3, samples=3)
    def frame(v):
        f = [0.] * len(JOINTS); f[W] = v; return f
    for v in (20., 20., 120., 20., 20.):
        assert d.update(frame(v), 100.) is None
    fired = None
    for _ in range(4):
        fired = fired or d.update(frame(90.), 100.)
    assert fired == WATCHED
