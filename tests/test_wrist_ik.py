import numpy as np
import pytest
import asyncio
import time
from types import SimpleNamespace
from autonomy.wrist_ik import WristIK


@pytest.mark.parametrize('pan,elbow,roll', [(0, 0, 0), (.4, .5, .3), (-.3, 1., -.5)])
def test_captured_ray_translation_preserves_orientation(pan, elbow, roll):
    ik = WristIK()
    pose = dict.fromkeys(ik.names, 0.)
    pose.update(shoulder_pan=pan, elbow_flex=elbow, wrist_roll=roll, gripper=.3)
    start, ray = ik.capture(pose)
    assert len(ik.names) == 5
    assert np.linalg.norm(ray) == pytest.approx(1.)
    for distance in np.arange(.002, .022, .002):
        pose = ik.solve(pose, start, distance)
        tip = ik.fk(pose)
        np.testing.assert_allclose(tip[:3, 3], start[:3, 3] + ray * distance, atol=.001)
        np.testing.assert_allclose(tip[:3, :3], start[:3, :3], atol=.02)
        assert pose['gripper'] == .3


def test_zero_pose_points_out_front_not_up():
    ik = WristIK()
    _, ray = ik.capture(dict.fromkeys(ik.names, 0.))
    np.testing.assert_allclose(ray, [1, 0, 0], atol=2e-5)


def test_reported_robot_pose_can_advance_on_ray():
    ik = WristIK()
    pose = dict(zip(ik.names, [-.00460194, -1.21491278, 1.42046621, .16720391, -.01380583]))
    pose['gripper'] = -.0076699
    start, ray = ik.capture(pose)
    for distance in [.002, .004, .006, .008, .010]:
        pose = ik.solve(pose, start, distance)
        tip = ik.fk(pose)
        np.testing.assert_allclose(tip[:3, 3], start[:3, 3] + ray * distance, atol=.001)
    assert ik.upper[ik.names.index('elbow_flex')] == 1.69


def test_unreachable_and_uncalibrated_commands_rejected():
    ik = WristIK()
    pose = dict.fromkeys(ik.names, 0.)
    start, _ = ik.capture(pose)
    with pytest.raises(ValueError):
        ik.solve(pose, start, 1.)
    pose['elbow_flex'] = 2.
    with pytest.raises(ValueError, match='calibration'):
        ik.solve(pose, start, .002)


def test_grab_starts_at_measured_pose_without_home_target_or_base_motion():
    from autonomy.control_server import ControlHub
    hub = ControlHub.__new__(ControlHub)
    hub.wrist_ik = WristIK()
    hub.state = dict.fromkeys(hub.wrist_ik.names, 0.)
    hub.state['gripper'] = .3
    hub.state_seen = time.monotonic()
    hub.home = None
    hub.target = None
    hub.auto = False
    hub.grab_debug = None
    hub.contact_threshold = 72.
    hub.grab_target_width = .95
    hub.grab_ik_tolerance = .001
    hub.args = SimpleNamespace(grab_open=1.2, grab_max_reaim=np.deg2rad(12), grab_max_travel=.25,
                               grab_gripper=0., grab_grip_time=.2, grab_approach_step=.004, grab_approach_period=.0, grab_ground_z=-0.0635)
    commands = []
    # Bound the vertical descent after two steps to keep the test quick; the
    # dive then ends where it is and closes the gripper.
    original = hub.wrist_ik.solve_position
    def bounded_test(seed, position, **kwargs):
        bounded_test.calls += 1
        if bounded_test.calls > 2:
            raise ValueError('test workspace boundary')
        return original(seed, position, **kwargs)
    bounded_test.calls = 0
    hub.wrist_ik.solve_position = bounded_test
    async def send(message):
        commands.append(message)
        if message['type'] == 'joints':
            hub.state.update(message['values'])
        hub.state_seen = time.monotonic()
    async def noop(*args, **kwargs):
        pass
    hub.pi_send = send
    async def grab_pose(pose, duration, phase, **kwargs):
        await send({'type': 'joints', 'values': pose})
        return True
    hub._grab_pose = grab_pose
    hub.grab_notice = noop
    hub.send_autonomy_status = noop
    async def run():
        async def telemetry():
            while True:
                hub.state_seen = time.monotonic()
                await asyncio.sleep(.01)
        task = asyncio.create_task(telemetry())
        try:
            await hub.grab_routine()
        finally:
            task.cancel()
    asyncio.run(run())
    motion = [c for c in commands if c['type'] == 'joints']
    assert motion[0]['values']['gripper'] == 1.2      # opened before diving
    assert not any(c['type'] in ('drive', 'home') for c in commands)  # no base/home move
    # The descent is bounded at the stubbed boundary; the grab then closes the
    # gripper and lifts rather than abandoning. The gripper opens, stays open
    # through the descent, then closes.
    descent = [c for c in motion if c['values']['gripper'] == 1.2]
    assert len(descent) >= 1
    assert motion[-1]['values']['gripper'] == 0.      # ends holding closed
    assert not hub.grab_active


def test_rotation_between_maps_one_axis_onto_another():
    from autonomy.wrist_ik import rotation_between
    for a, b in [([1, 0, 0], [0, 1, 0]), ([0, 0, 1], [1, 0, 0]),
                 ([1, 2, 3], [-2, 1, .5]), ([1, 0, 0], [1, 0, 0]),
                 ([1, 0, 0], [-1, 0, 0])]:
        a = np.array(a, float) / np.linalg.norm(a)
        b = np.array(b, float) / np.linalg.norm(b)
        r = rotation_between(a, b)
        np.testing.assert_allclose(r @ a, b, atol=1e-9)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-9)
        assert np.linalg.det(r) == pytest.approx(1.)


def test_clamp_direction_bounds_the_reaim():
    from autonomy.wrist_ik import clamp_direction
    reference = np.array([1., 0., 0.])
    limit = np.deg2rad(12)
    inside = np.array([np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5)), 0.])
    ray, angle = clamp_direction(inside, reference, limit)
    np.testing.assert_allclose(ray, inside, atol=1e-9)
    assert np.degrees(angle) == pytest.approx(5., abs=1e-6)

    outside = np.array([np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40)), 0.])
    ray, angle = clamp_direction(outside, reference, limit)
    assert angle == pytest.approx(limit)
    assert np.degrees(np.arccos(np.dot(ray, reference))) == pytest.approx(12., abs=1e-6)
    # Clamping keeps the direction it was steered toward, just shortened.
    assert np.dot(np.cross(reference, ray), np.cross(reference, outside)) > 0


def test_dive_keeps_the_tool_pointing_along_its_travel_direction():
    """Regression: centring rotated the tool while position stayed on the
    captured ray, so the gripper travelled one way and pointed another -- 43 deg
    apart after 100 mm, which also contorted the arm into an early abort."""
    from autonomy.wrist_ik import clamp_direction, rotation_between
    ik = WristIK()
    pose = dict(zip(ik.names, [-.00460194, -1.21491278, 1.42046621, .16720391, -.01380583]))
    pose['gripper'] = -.0077
    start, forward = ik.capture(pose)
    limit = np.deg2rad(12)
    dive, progress, worst = dict(pose), 0., 0.
    for _ in range(60):
        nxt = progress + .002
        guidance = dict(dive)
        for name in ('shoulder_pan', 'wrist_flex'):      # persistent off-centre target
            i = ik.names.index(name)
            guidance[name] = float(np.clip(guidance[name] + .015, ik.lower[i], ik.upper[i]))
        aim = ik.fk(guidance)
        ray, reaim = clamp_direction(aim[:3, 2], forward, limit)
        aim[:3, :3] = rotation_between(forward, ray) @ start[:3, :3]
        dive = ik.solve(dive, aim, 0., position_tolerance=.001,
                        desired_position=start[:3, 3] + ray * nxt)
        progress = nxt
        assert reaim <= limit + 1e-9
        worst = max(worst, np.degrees(np.arccos(np.clip(np.dot(ik.fk(dive)[:3, 2], ray), -1, 1))))
    assert progress > .1, 'steering the ray should also clear the old 100 mm abort'
    assert worst < 2., f'tool drifted {worst:.1f} deg from its travel direction'


def test_solve_position_reaches_the_ground_plane_when_the_locked_ray_cannot():
    """Vertical descent to the floor needs orientation freed.

    Locking the captured orientation runs the arm out of reach ~100 mm down;
    solve_position lets the wrist tilt and reaches the 2.5in-pedestal ground
    plane at z=-0.0635 m (~157 mm of descent) from the normal reaching pose.
    """
    ik = WristIK()
    pose = dict(shoulder_pan=0., shoulder_lift=-1.21, elbow_flex=1.42,
                wrist_flex=.167, wrist_roll=-.014, gripper=0.)
    start = ik.fk(pose)
    x, y, z0 = start[:3, 3]
    ground = -0.0635

    # locked-orientation ray solve cannot get there
    with pytest.raises(ValueError):
        p = dict(pose)
        for i in range(1, 120):
            p = ik.solve(p, start, 0., position_tolerance=.003,
                         desired_position=np.array([x, y, z0 - i * .002]))

    # position-only descent reaches the ground
    p = dict(pose)
    prog, total = 0., z0 - ground
    while z0 - prog > ground:
        nd = min(prog + .002, total)
        p = ik.solve_position(p, np.array([x, y, z0 - nd]), position_tolerance=.004)
        prog = nd
    tip = ik.fk(p)
    assert abs(tip[2, 3] - ground) < .005
    assert tip[2, 2] < -0.5, 'tool should tilt downward as it descends'


def test_solve_position_holds_xy_while_dropping_z():
    ik = WristIK()
    pose = dict(shoulder_pan=0., shoulder_lift=-1.0, elbow_flex=1.2,
                wrist_flex=.1, wrist_roll=0., gripper=0.)
    start = ik.fk(pose)
    x, y, z0 = start[:3, 3]
    p = ik.solve_position(dict(pose), np.array([x, y, z0 - .05]), position_tolerance=.004)
    tip = ik.fk(p)
    assert abs(tip[0, 3] - x) < .005 and abs(tip[1, 3] - y) < .005
    assert abs(tip[2, 3] - (z0 - .05)) < .005


def test_approach_along_forward_vector_reaches_ground_when_steep_enough():
    """The dive follows the captured forward vector to the ground plane.

    A steep approach reaches the floor; a shallow one runs out of reach short
    of it (honest geometry -- the ground is simply too far along a flat line).
    """
    ik = WristIK()
    ground = -0.0635

    def approach(wrist_flex):
        pose = dict(shoulder_pan=0., shoulder_lift=-1.0, elbow_flex=1.3,
                    wrist_flex=wrist_flex, wrist_roll=0., gripper=0.)
        start = ik.fk(pose)
        sx = start[:3, 3].copy()
        fwd = start[:3, 2] / np.linalg.norm(start[:3, 2])
        dive, prog = dict(pose), 0.
        while ik.fk(dive)[2, 3] > ground:
            nd = prog + .002
            t = sx + fwd * nd
            if t[2] <= ground:
                t[2] = ground
            try:
                dive = ik.solve_position(dive, t, position_tolerance=.004)
            except ValueError:
                break
            prog = nd
        return ik.fk(dive)[2, 3]

    assert abs(approach(1.0) - ground) < .006, 'a steep approach must reach the ground'
    assert approach(0.167) > ground + .005, 'a shallow approach stops short, by geometry'


def test_solve_position_can_pitch_the_camera_without_losing_position():
    import numpy as np
    from autonomy.wrist_ik import WristIK
    ik = WristIK()
    pose = {'shoulder_pan': -.015, 'shoulder_lift': -1.2226, 'elbow_flex': 1.4527, 'wrist_flex': .6289,
            'wrist_roll': -.0107, 'gripper': 1.2}                        # a recorded grab-start pose
    start = ik.fk(pose)
    target = start[:3, 3] + np.array([.01, 0., -.01])
    axis = start[:3, 2] / np.linalg.norm(start[:3, 2])
    steeper = float(np.clip(axis[2] - .15, -1, 1))                       # point the camera ~9 degrees further down
    solved = ik.solve_position(pose, target, position_tolerance=.0005, orient_weight=.01,
                               orientation=start[:3, :3], elevation=steeper, elevation_weight=.5)
    result = ik.fk(solved)
    assert np.linalg.norm(result[:3, 3] - target) < .0005
    z_axis = result[:3, 2] / np.linalg.norm(result[:3, 2])
    assert abs(z_axis[2] - steeper) < .01
