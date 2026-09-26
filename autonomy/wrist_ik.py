"""URDF tool-frame FK and bounded Cartesian ray IK (metres/radians)."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rotation_between(a, b):
    """Shortest rotation taking unit vector `a` onto unit vector `b`."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis)
    cosine = float(np.dot(a, b))
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3)
        # Antiparallel: any perpendicular axis gives the half turn.
        perpendicular = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, perpendicular)
        return rotation(axis, np.pi)
    return rotation(axis / sine, float(np.arctan2(sine, cosine)))


def clamp_direction(ray, reference, max_angle):
    """Pull `ray` back toward `reference` so it stays within `max_angle`."""
    ray = np.asarray(ray, float) / np.linalg.norm(ray)
    reference = np.asarray(reference, float) / np.linalg.norm(reference)
    cosine = float(np.clip(np.dot(ray, reference), -1, 1))
    angle = float(np.arccos(cosine))
    if angle <= max_angle:
        return ray, angle
    perpendicular = ray - cosine * reference
    norm = np.linalg.norm(perpendicular)
    if norm < 1e-9:
        return reference, 0.
    perpendicular /= norm
    return reference * np.cos(max_angle) + perpendicular * np.sin(max_angle), max_angle


class WristIK:
    def __init__(self):
        root = ET.parse(Path(__file__).resolve().parents[1] / 'web/so101/so101_new_calib.urdf').getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, link = [], 'gripper_frame_link'
        while link in by_child:
            joint = by_child[link]
            chain.insert(0, joint)
            link = joint.find('parent').get('link')
        self.chain = []
        self.names, self.lower, self.upper = [], [], []
        for joint in chain:
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            r, p, y = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            transform = np.eye(4)
            transform[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            transform[:3, 3] = xyz
            name = joint.get('name') if joint.get('type') != 'fixed' else None
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ') if name else None
            self.chain.append((name, transform, axis))
            if name:
                self.names.append(name)
                self.lower.append(float(joint.find('limit').get('lower')))
                self.upper.append(float(joint.find('limit').get('upper')))

    def fk(self, pose):
        transform = np.eye(4)
        for name, origin, axis in self.chain:
            transform = transform @ origin
            if name:
                motion = np.eye(4)
                motion[:3, :3] = rotation(axis, pose[name])
                transform = transform @ motion
        return transform

    def capture(self, pose):
        start = self.fk(pose)
        # Tool +Z points out through the fingers: the fixed tool transform
        # rotates gripper -Z by pi about Y. Not a camera/world axis guess.
        return start, start[:3, 2].copy()

    def solve_position(self, seed, position, position_tolerance=.003, orient_weight=0.,
                       orientation=None, max_step=.06, elevation=None, elevation_weight=0.):
        """Solve for a target gripper *position*, letting orientation float.

        The captured-ray solve() locks the full orientation, which runs the arm
        out of reach after ~100 mm of straight-down travel. A vertical descent
        to the ground only needs the position; freeing the orientation lets the
        wrist tilt downward and reach the floor (~157 mm). `orient_weight` may
        add a soft pull toward `orientation` (default: the seed's) if some
        stability is wanted, but 0 -- pure position -- reaches furthest.
        `elevation` with `elevation_weight` pulls only the up/down angle of the gripper/camera +Z axis toward
        `elevation` (the z component of a unit direction). That is the one pointing freedom this arm has beyond
        position: apart from the base pan every joint bends in one vertical plane, so sideways aim follows the pan.
        """
        position = np.asarray(position, float)
        q = np.array([seed[n] for n in self.names])
        R = (self.fk(seed)[:3, :3] if orientation is None else np.asarray(orientation))
        violations = [f'{name}={value:.5f} rad (model range {low:.5f} to {high:.5f})'
                      for name, value, low, high in zip(self.names, q, self.lower, self.upper)
                      if not np.isfinite(value) or value < low - .02 or value > high + .02]
        if violations:
            raise ValueError('Measured joints outside URDF limits: ' + '; '.join(violations))

        def residual(values):
            t = self.fk(dict(seed, **dict(zip(self.names, values))))
            r = t[:3, 3] - position
            if orient_weight > 0:
                r = np.r_[r, orient_weight * (t[:3, :3] - R).ravel()]
            if elevation is not None and elevation_weight > 0:
                z_axis = t[:3, 2] / np.linalg.norm(t[:3, 2])
                r = np.r_[r, elevation_weight * (z_axis[2] - elevation)]
            return r

        for _ in range(80):
            error = residual(q)
            jac = np.column_stack([(residual(q + np.eye(len(q))[i] * 1e-5) - error) / 1e-5 for i in range(len(q))])
            step = np.linalg.solve(jac.T @ jac + np.eye(len(q)) * 1e-6, -jac.T @ error)
            q = np.clip(q + np.clip(step, -max_step, max_step), self.lower, self.upper)
            if np.linalg.norm(step) < 1e-7:
                break
        result = dict(seed, **{n: float(v) for n, v in zip(self.names, q)})
        error = float(np.linalg.norm(self.fk(result)[:3, 3] - position))
        if error > position_tolerance:
            raise ValueError(f'Position unreachable within tolerance: {error*1000:.1f} mm')
        return result

    def solve(self, seed, start, distance, position_tolerance=.001, desired_position=None):
        desired = start[:3, 3] + start[:3, 2] * distance if desired_position is None else np.asarray(desired_position)
        q = np.array([seed[n] for n in self.names])
        violations = [f'{name}={value:.5f} rad (model range {low:.5f} to {high:.5f})'
                      for name, value, low, high in zip(self.names, q, self.lower, self.upper)
                      if not np.isfinite(value) or value < low - .02 or value > high + .02]
        if violations:
            raise ValueError('Measured joints outside URDF limits: ' + '; '.join(violations)
                             + '; check model limits and hardware zero calibration')

        def residual(values):
            pose = dict(seed, **dict(zip(self.names, values)))
            t = self.fk(pose)
            # Five pose joints cannot satisfy arbitrary six-DOF commands.
            # Keep the entire captured orientation, reject infeasible steps.
            return np.r_[t[:3, 3] - desired, .08 * (t[:3, :3] - start[:3, :3]).ravel()]

        for _ in range(80):
            error = residual(q)
            jac = np.column_stack([(residual(q + np.eye(len(q))[i] * 1e-5) - error) / 1e-5 for i in range(len(q))])
            step = np.linalg.solve(jac.T @ jac + np.eye(len(q)) * 1e-6, -jac.T @ error)
            q = np.clip(q + np.clip(step, -.04, .04), self.lower, self.upper)
            if np.linalg.norm(step) < 1e-7:
                break
        result = dict(seed, **{n: float(v) for n, v in zip(self.names, q)})
        t = self.fk(result)
        position_error = float(np.linalg.norm(t[:3, 3] - desired))
        angle = float(np.arccos(np.clip((np.trace(start[:3, :3].T @ t[:3, :3]) - 1) / 2, -1, 1)))
        if position_error > position_tolerance or angle > .02:
            raise ValueError(f'Ray is unreachable within tolerance: position error {position_error*1000:.1f} mm, orientation error {np.degrees(angle):.1f} deg')
        if max(abs(result[n] - seed[n]) for n in self.names) > .08:
            raise ValueError('Ray step requires excessive joint motion near a singularity')
        return result
