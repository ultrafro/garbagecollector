import unittest

from autonomy.tracking import Detection, associated, closest, track


class TrackingTests(unittest.TestCase):
    def test_closest_is_largest_box(self):
        small = Detection("cup", .9, 0, 0, 10, 10)
        large = Detection("bottle", .7, 0, 0, 30, 20)
        self.assertIs(closest([small, large]), large)

    def test_locked_target_does_not_switch_to_larger_competitor(self):
        locked = Detection("bottle", .4, 100, 100, 180, 180)
        same_object_new_alias = Detection("discarded packaging", .3, 105, 103, 185, 183)
        larger_other = Detection("cup", .9, 400, 50, 620, 330)
        self.assertIs(associated(locked, [larger_other, same_object_new_alias], 640, 360), same_object_new_alias)

    def test_locked_target_rejects_distant_object(self):
        locked = Detection("bottle", .4, 50, 50, 100, 100)
        other = Detection("cup", .9, 500, 250, 620, 350)
        self.assertIsNone(associated(locked, [other], 640, 360))

    def test_search_rotates(self):
        command = track(None, 640, 360, patrol_theta=3)
        self.assertEqual((command.mode, command.forward, command.theta), ("search", 0, 3))

    def test_turns_before_approaching(self):
        right = Detection("cup", .9, 500, 100, 600, 200)
        command = track(right, 640, 360)
        self.assertEqual(command.mode, "align")
        self.assertGreater(command.theta, 0)
        self.assertEqual(command.forward, 0)

    def test_turn_correction_clears_servo_deadband(self):
        slightly_right = Detection("cup", .9, 345, 100, 385, 200)
        self.assertEqual(track(slightly_right, 640, 360, min_theta=4).theta, 4)

    def test_approaches_centered_distant_target(self):
        centered = Detection("cup", .9, 300, 140, 340, 200)
        command = track(centered, 640, 360)
        self.assertEqual(command.mode, "approach")
        self.assertGreater(command.forward, 0)

    def test_stops_at_target_size(self):
        centered = Detection("cup", .9, 280, 130, 360, 230)
        self.assertEqual(track(centered, 640, 360, target_height_ratio=100 / 360).mode, "arrived")


    def test_steers_while_driving_within_band(self):
        slightly_right = Detection("cup", .9, 360, 150, 400, 210)          # x error 0.19, vertically centred
        command = track(slightly_right, 640, 360, ignore_distance=True, steer_band=.25, max_theta=12)
        self.assertEqual(command.mode, "steer")
        self.assertGreater(command.forward, 0)                             # keeps driving...
        self.assertGreater(command.theta, 0)                               # ...while turning toward it
        self.assertLess(command.theta, 12)                                 # proportionally, not flat out

    def test_slows_as_target_drifts_off_centre(self):
        centred = Detection("cup", .9, 300, 150, 340, 210)
        low = Detection("cup", .9, 300, 250, 340, 310)                     # y error 0.55
        kwargs = dict(ignore_distance=True, steer_band=.25, vertical_band=.5, max_forward=.1)
        self.assertEqual(track(centred, 640, 360, **kwargs).forward, .1)
        self.assertEqual(track(low, 640, 360, **kwargs).forward, 0.)      # wrist catches up first
        self.assertEqual(track(low, 640, 360, **kwargs).mode, "approach")  # still centred horizontally

    def test_outside_steer_band_turns_in_place(self):
        far_right = Detection("cup", .9, 500, 150, 560, 210)
        command = track(far_right, 640, 360, ignore_distance=True, steer_band=.25)
        self.assertEqual((command.mode, command.forward), ("align", 0.0))


if __name__ == "__main__":
    unittest.main()
