"""Z from Kinect: sampling a depth map at landmark positions."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc.blobs import BlobDetection  # noqa: E402
from mp_osc.depth import apply_depth, sample_depth  # noqa: E402
from mp_osc.detectors import Detections  # noqa: E402
from tests.test_osc_out import fake_face, fake_hand, fake_pose  # noqa: E402


def depth_map(h=100, w=200):
    d = np.zeros((h, w), dtype=np.float32)
    d[:, :100] = 1.0                 # left half: 1 m
    d[:, 100:] = 2.0                 # right half: 2 m
    d[70:90, 10:30] = 0.0            # a hole (no reading): x 0.05..0.15, y 0.7..0.9
    d[10, 150] = 9.0                 # one wild pixel
    return d


class SampleDepthTest(unittest.TestCase):
    def test_median_in_a_window_and_edges(self):
        d = depth_map()
        xs = np.array([0.25, 0.75, 0.0, 0.999, 1.2, -0.1])
        ys = np.array([0.5, 0.5, 0.0, 0.999, 0.5, 0.5])
        z = sample_depth(d, xs, ys)
        self.assertEqual(list(z[:4]), [1.0, 2.0, 1.0, 2.0])            # corners are clipped, not dropped
        self.assertEqual(list(z[4:]), [0.0, 0.0])                       # outside the image -> 0

    def test_holes_and_outliers(self):
        d = depth_map()
        z = sample_depth(d, np.array([0.1]), np.array([0.8]))           # window fully inside the hole
        self.assertEqual(float(z[0]), 0.0)
        z = sample_depth(d, np.array([0.155]), np.array([0.8]), window=5)  # window touches the hole: valid pixels win
        self.assertEqual(float(z[0]), 1.0)
        z = sample_depth(d, np.array([0.75]), np.array([0.1]))          # the wild pixel is ignored by the median
        self.assertEqual(float(z[0]), 2.0)
        self.assertEqual(sample_depth(d, np.array([]), np.array([])).shape, (0,))

    def test_apply_depth_rewrites_every_z(self):
        d = depth_map()
        det = Detections(poses=[fake_pose(0.25, 0.5)], hands=[fake_hand(0.75, 0.5)], faces=[fake_face(0.75, 0.5)],
                         blobs=[BlobDetection(center=np.array([0.25, 0.5, 0.0], np.float32),
                                              landmarks=np.zeros((16, 2), np.float32),
                                              box=np.array([0.2, 0.4, 0.3, 0.6], np.float32),
                                              area=0.01, angle=0.0, size=np.array([0.1, 0.1], np.float32), pixels=10)])
        det.poses[0].landmarks[:, 2] = 0.123
        apply_depth(det, d)
        self.assertTrue(np.all(det.poses[0].landmarks[:, 2] == 1.0))     # metres, not MediaPipe's 0.123
        self.assertTrue(np.all(det.hands[0].landmarks[:, 2] == 2.0))
        self.assertTrue(np.all(det.faces[0].landmarks[:, 2] == 2.0))
        self.assertEqual(float(det.blobs[0].center[2]), 1.0)
        self.assertEqual(float(det.poses[0].center[2]), 1.0)             # derived centers follow
        self.assertEqual(det.poses[0].landmarks.shape, (33, 5))            # visibility/presence untouched
        self.assertTrue(np.all(det.poses[0].landmarks[:, 3] == 1.0))


if __name__ == "__main__":
    unittest.main()
