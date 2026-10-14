"""Unit tests for mp_osc.tracking (no camera / MediaPipe required)."""

import os
import sys
import unittest
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mp_osc.tracking import Tracker  # noqa: E402


@dataclass
class FakeDetection:
    landmarks: np.ndarray

    @property
    def center(self):
        return self.landmarks.mean(axis=0)

    def bbox(self):
        return np.concatenate([self.landmarks[:, :2].min(axis=0), self.landmarks[:, :2].max(axis=0)])


def det(x, y, spread=0.05):
    pts = np.array([[x - spread, y - spread, 0], [x + spread, y + spread, 0]], dtype=np.float32)
    return FakeDetection(pts)


class TrackerTest(unittest.TestCase):
    def test_new_detection_creates_track_and_enter_event(self):
        tr = Tracker()
        up = tr.update([det(0.5, 0.5)], now=0.0)
        self.assertEqual([t.id for t in up.active], [0])
        self.assertEqual(up.entered, [0])
        self.assertEqual(up.left, [])
        self.assertEqual(up.active[0].frames, 1)

    def test_id_is_stable_while_moving_slowly(self):
        tr = Tracker(max_distance=0.25)
        ids = set()
        for i in range(20):
            up = tr.update([det(0.1 + i * 0.02, 0.5)], now=i / 30)
            ids.update(t.id for t in up.active)
            self.assertEqual(up.entered, [0] if i == 0 else [])
        self.assertEqual(ids, {0})

    def test_ids_follow_people_when_detection_order_swaps(self):
        tr = Tracker()
        a, b = det(0.2, 0.5), det(0.8, 0.5)
        tr.update([a, b], now=0.0)
        up = tr.update([det(0.81, 0.5), det(0.21, 0.5)], now=0.033)  # swapped order
        by_id = {t.id: t for t in up.active}
        self.assertAlmostEqual(float(by_id[0].center[0]), 0.21, places=3)
        self.assertAlmostEqual(float(by_id[1].center[0]), 0.81, places=3)
        self.assertEqual(up.entered, [])

    def test_far_jump_creates_new_id(self):
        tr = Tracker(max_distance=0.25, timeout=10.0)
        tr.update([det(0.1, 0.1)], now=0.0)
        up = tr.update([det(0.9, 0.9)], now=0.033)
        self.assertEqual(up.entered, [1])
        self.assertEqual([t.id for t in up.active], [1])
        self.assertEqual(len(tr.tracks), 2)  # the old one is still coasting

    def test_leave_after_timeout_and_new_id_on_return(self):
        tr = Tracker(timeout=0.5)
        tr.update([det(0.5, 0.5)], now=0.0)
        up = tr.update([], now=0.3)
        self.assertEqual(up.left, [])          # still within timeout
        self.assertEqual(up.active, [])
        up = tr.update([], now=0.6)
        self.assertEqual(up.left, [0])
        up = tr.update([det(0.5, 0.5)], now=0.7)
        self.assertEqual(up.entered, [1])      # a returning person gets a fresh id

    def test_coasting_track_is_rematched_before_timeout(self):
        tr = Tracker(timeout=0.5)
        tr.update([det(0.5, 0.5)], now=0.0)
        tr.update([], now=0.2)                 # one missed frame
        up = tr.update([det(0.52, 0.5)], now=0.25)
        self.assertEqual(up.entered, [])
        self.assertEqual([t.id for t in up.active], [0])

    def test_velocity_estimate(self):
        tr = Tracker(velocity_smooth=0.0)      # no smoothing: instantaneous velocity
        tr.update([det(0.0, 0.5)], now=0.0)
        up = tr.update([det(0.1, 0.5)], now=0.1)   # 0.1 units in 0.1 s -> 1.0 units/s
        t = up.active[0]
        self.assertAlmostEqual(float(t.velocity[0]), 1.0, places=4)
        self.assertAlmostEqual(float(t.velocity[1]), 0.0, places=4)
        self.assertAlmostEqual(t.speed, 1.0, places=4)

    def test_velocity_is_smoothed_by_default(self):
        tr = Tracker()
        tr.update([det(0.0, 0.5)], now=0.0)
        up = tr.update([det(0.1, 0.5)], now=0.1)
        self.assertGreater(float(up.active[0].velocity[0]), 0.0)
        self.assertLess(float(up.active[0].velocity[0]), 1.0)

    def test_landmark_smoothing(self):
        tr = Tracker(smooth=0.5, max_distance=10.0)
        tr.update([det(0.0, 0.0)], now=0.0)
        up = tr.update([det(1.0, 1.0)], now=0.1)
        t = up.active[0]
        self.assertAlmostEqual(float(t.center[0]), 0.5, places=4)
        self.assertAlmostEqual(float(t.landmarks[0, 0]), 0.5 - 0.05, places=4)
        # the raw detection is still available
        self.assertAlmostEqual(float(t.detection.center[0]), 1.0, places=4)

    def test_no_smoothing_keeps_raw_landmarks(self):
        tr = Tracker(smooth=0.0, max_distance=10.0)
        tr.update([det(0.0, 0.0)], now=0.0)
        up = tr.update([det(1.0, 1.0)], now=0.1)
        self.assertAlmostEqual(float(up.active[0].center[0]), 1.0, places=4)

    def test_reset_returns_live_ids(self):
        tr = Tracker()
        tr.update([det(0.2, 0.2), det(0.8, 0.8)], now=0.0)
        self.assertEqual(tr.reset(), [0, 1])
        self.assertEqual(tr.tracks, {})

    def test_active_sorted_by_id(self):
        tr = Tracker()
        tr.update([det(0.2, 0.2), det(0.8, 0.8)], now=0.0)
        up = tr.update([det(0.8, 0.8), det(0.2, 0.2)], now=0.1)
        self.assertEqual([t.id for t in up.active], [0, 1])

    def test_many_people_greedy_matching(self):
        tr = Tracker(max_distance=0.2)
        xs = [0.1, 0.3, 0.5, 0.7, 0.9]
        tr.update([det(x, 0.5) for x in xs], now=0.0)
        # everybody shifts right by 0.05; order reversed
        up = tr.update([det(x + 0.05, 0.5) for x in reversed(xs)], now=0.1)
        self.assertEqual(up.entered, [])
        for t in up.active:
            self.assertAlmostEqual(float(t.center[0]), xs[t.id] + 0.05, places=3)


if __name__ == "__main__":
    unittest.main()
