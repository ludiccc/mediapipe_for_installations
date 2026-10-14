"""Tests for the frame sources. The webcam is simulated with a fake cv2.VideoCapture."""

import os
import sys
import time
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc import capture  # noqa: E402

PORTRAIT = os.path.join(ROOT, "brother-977170_1280.jpg")


class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture producing numbered frames at ~200 fps."""

    instances = []

    def __init__(self, index):
        self.index = index
        self.opened = index in (0, 1)
        self.props = {}
        self.count = 0
        self.released = False
        self.fail_after = None
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value

    def get(self, prop):
        return 30.0 if prop == cv2.CAP_PROP_FPS else self.props.get(prop, 0)

    def read(self):
        if self.fail_after is not None and self.count >= self.fail_after:
            return False, None
        time.sleep(0.005)
        self.count += 1
        frame = np.full((8, 8, 3), self.count % 256, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class WebcamSourceTest(unittest.TestCase):
    def setUp(self):
        FakeCapture.instances.clear()
        patcher = mock.patch.object(capture.cv2, "VideoCapture", FakeCapture)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_open_source_picks_webcam_for_digits(self):
        src = capture.open_source("1", 1280, 720)
        self.addCleanup(src.release)
        self.assertIsInstance(src, capture.WebcamSource)
        fake = FakeCapture.instances[-1]
        self.assertEqual(fake.index, 1)
        self.assertEqual(fake.props[cv2.CAP_PROP_FRAME_WIDTH], 1280)
        self.assertEqual(fake.props[cv2.CAP_PROP_FRAME_HEIGHT], 720)
        self.assertEqual(src.fps, 30.0)
        self.assertTrue(src.live)

    def test_unavailable_camera_raises_helpful_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            capture.open_source("5")
        self.assertIn("could not open webcam 5", str(ctx.exception))
        self.assertIn("--source", str(ctx.exception))

    def test_each_read_returns_a_newer_frame(self):
        src = capture.WebcamSource(0)
        self.addCleanup(src.release)
        seen = []
        for _ in range(5):
            ok, frame = src.read()
            self.assertTrue(ok)
            seen.append(int(frame[0, 0, 0]))
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(len(set(seen)), 5)   # never the same frame twice

    def test_slow_consumer_gets_the_latest_frame_not_a_backlog(self):
        src = capture.WebcamSource(0)
        self.addCleanup(src.release)
        src.read()
        time.sleep(0.2)                       # ~40 frames go by while we are "busy"
        ok, frame = src.read()
        self.assertTrue(ok)
        self.assertGreater(int(frame[0, 0, 0]), 20)

    def test_timestamps_are_monotonic_wall_clock(self):
        src = capture.WebcamSource(0)
        self.addCleanup(src.release)
        src.read()
        t1 = src.timestamp()
        src.read()
        t2 = src.timestamp()
        self.assertGreater(t2, t1)

    def test_camera_failure_ends_the_stream(self):
        src = capture.WebcamSource(0, timeout=2.0)
        self.addCleanup(src.release)
        FakeCapture.instances[-1].fail_after = 3
        results = [src.read()[0] for _ in range(3)]
        self.assertEqual(results, [True, True, True])
        t = time.monotonic()
        ok, _ = src.read()
        self.assertFalse(ok)
        self.assertLess(time.monotonic() - t, 1.5)   # reported quickly, not after the timeout

    def test_release_stops_thread_and_camera(self):
        src = capture.WebcamSource(0)
        src.read()
        src.release()
        self.assertTrue(FakeCapture.instances[-1].released)
        self.assertFalse(src._thread.is_alive())


class FileSourcesTest(unittest.TestCase):
    def test_image_source_repeats_and_copies(self):
        src = capture.open_source(PORTRAIT)
        self.assertIsInstance(src, capture.ImageSource)
        ok, a = src.read()
        ok2, b = src.read()
        self.assertTrue(ok and ok2)
        self.assertEqual(a.shape, (853, 1280, 3))
        a[:] = 0
        self.assertGreater(int(b.max()), 0)   # a copy: callers may draw on it

    def test_missing_files(self):
        with self.assertRaises(RuntimeError):
            capture.open_source("/nope/missing.png")
        with self.assertRaises(RuntimeError):
            capture.open_source("/nope/missing.mp4")

    def test_video_loop_and_timestamps(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.mp4")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 32))
            for i in range(4):
                w.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
            w.release()

            src = capture.open_source(path, loop=False)
            self.assertIsInstance(src, capture.VideoFileSource)
            self.assertFalse(src.live)
            self.assertEqual(src.fps, 25.0)
            frames = [src.read()[0] for _ in range(5)]
            self.assertEqual(frames, [True] * 4 + [False])
            src.release()

            src = capture.open_source(path, loop=True)
            stamps = []
            for _ in range(6):                 # wraps around after 4 frames
                ok, _ = src.read()
                self.assertTrue(ok)
                stamps.append(src.timestamp())
            np.testing.assert_allclose(np.diff(stamps), 1 / 25, atol=1e-9)   # keeps increasing across the loop
            src.release()


if __name__ == "__main__":
    unittest.main()
