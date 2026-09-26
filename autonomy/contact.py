"""Contact detection from STS3215 servo load.

The rule is the simple one: stop when the load is high. The dive watches
wrist_flex -- that is the joint that takes the load when the gripper presses
into what it is reaching for -- and sustained load at or above the threshold is
contact. The threshold is live-adjustable from the control page.

Two details are not obvious and both caused real misses:

* ``decode_load`` (pi/lekiwi_ws.py) returns a *signed* load -- 10-bit magnitude
  plus a direction bit. Pressing a joint one way makes the raw value fall, so
  anything watching the signed number misses contact in that direction
  entirely. Effort is the magnitude; sign is only direction. Everything here
  works on ``abs(load)``.
* Which joint carries a contact depends on the arm's pose. The elbow dominated
  the reach-limit probes, but pressing into the floor loads wrist_flex, so every
  channel is watched against its own threshold.

WATCHED names the joint the dive stops on. Other channels are still recorded in
the debug payload and the dive CSV so their behaviour stays visible, but they
cannot stop the dive: an earlier version let any channel fire and stopped a good
dive on wrist_roll drifting to 40, a channel that read exactly 0 in every probe
and so had no measured noise to set a threshold from.

Thresholds come from measured free travel (screenshots/load-probe-*.csv and
grab-dive-*.csv), taken after ARM_MM of travel so the break-away transient is
excluded -- leaving rest costs ~110 on the elbow, more than light contact does,
and is over within the first 10 mm.

Moving the joint is itself expensive: wrist_flex spikes to 152 and 160 within
the first 10 mm of a dive, purely from accelerating each step, and sits in a
68-84 band between spikes. Recorded wrist_flex maxima while moving freely:

  reach-limit probes (different pose)      64
  real dive, first 22 mm                  160

Those 152/160 spikes are single samples and land inside the first 10 mm, before
ARM_MM. What the joint actually *sustains* while moving is the 68-84 band, and
`samples` consecutive readings are required, so the threshold has to clear 84
rather than 160. Two defaults were wrong in opposite directions:

  72   sat inside the 68-84 sustained band and fired on the third armed sample
       of every dive, 22 mm in, with nothing touched
  200  cleared the isolated spikes as well, so nothing ever fired and the dive
       ran to its travel limit

112 clears the sustained band with margin while staying well under the spikes.
It is still a noise floor, not a measured grasp: no recorded run here contains a
confirmed contact on wrist_flex, so the control page slider is how the working
value gets found.

An earlier version measured *rise above a running floor* rather than absolute
load. It is recorded here because the failure is instructive: that scheme has to
wait for the break-away transient to turn over before it arms, so a dive that
meets its object early -- transient and contact merging into one ramp -- arms
after the peak and scores a 24 -> 156 elbow climb as a rise of 8. Absolute
thresholds have no such blind spot.
"""

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

WATCHED = 'elbow_flex'

# The watched load is smoothed with a short moving average before comparison.
# Movement torque throws isolated single-sample spikes (recorded up to 160 with
# nothing touched); averaging a few samples flattens those while a real dig-in,
# which is sustained, survives. On the ground-contact calibration run this
# widened the free-vs-contact gap and opened a 60-68 threshold window that raw
# load did not have: free travel peaks at MA3 59, contact reaches MA3 77.
SMOOTHING = 3

DEFAULT_THRESHOLD = 140.

# Travel (mm) before the detector may fire, to clear the break-away transient.
ARM_MM = 16.


class LoadContactDetector:
    """Stops the dive on sustained high load in the watched joint."""

    def __init__(self, threshold=DEFAULT_THRESHOLD, samples=3, arm_mm=ARM_MM,
                 watched=WATCHED, smoothing=SMOOTHING):
        self.threshold = float(threshold)
        self.samples = int(samples)
        self.arm_mm = float(arm_mm)
        self.watched = watched
        self.smoothing = max(1, int(smoothing))
        self.loads = dict.fromkeys(JOINTS)
        self._history = []
        self.smoothed = None
        self.run = 0
        self.travel_mm = 0.
        self.triggered = None

    @property
    def armed(self):
        return self.travel_mm >= self.arm_mm

    @property
    def load(self):
        return self.loads[self.watched]

    def update(self, loads, travel_mm, threshold=None):
        """Feed one fresh servo_load reading; return the contacting joint or None.

        `threshold` may be passed each call so the control page's slider takes
        effect mid-dive rather than only on the next grab.
        """
        self.travel_mm = float(travel_mm)
        if threshold is not None:
            self.threshold = float(threshold)
        if not loads:
            return None
        for name, load in zip(JOINTS, loads):
            if load is not None:
                self.loads[name] = abs(float(load))
        load = self.loads[self.watched]
        if load is None:
            return None
        self._history.append(load)
        del self._history[:-self.smoothing]
        self.smoothed = sum(self._history) / len(self._history)
        if self.armed and self.smoothed >= self.threshold:
            self.run += 1
        else:
            self.run = 0
        self.triggered = self.watched if self.run >= self.samples else None
        return self.triggered

    def summary(self, name=None):
        return (f"{self.watched} load {self.smoothed:.0f} (MA{self.smoothing}) "
                f"at or above the {self.threshold:.0f} threshold")

    def debug(self):
        return {"contact_joint": self.triggered, "contact_armed": self.armed,
                "contact_watched": self.watched, "contact_travel_mm": round(self.travel_mm, 1),
                "contact_arm_mm": self.arm_mm, "contact_loads": dict(self.loads),
                "contact_load": self.load, "contact_smoothed": self.smoothed,
                "contact_smoothing": self.smoothing, "contact_threshold": self.threshold,
                "contact_samples": self.run}
