"""Tests for the Blobs detector (OpenCV shapes) and its OSC output.

Synthetic frames only: a grey floor seen from the ceiling with bright and dark
shapes moving over it. No camera, no MediaPipe model.
"""

import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc.app import App  # noqa: E402
from mp_osc.blobs import BLOB_MODES, BlobDetector, resample_contour  # noqa: E402
from mp_osc.config import Config, parse_args  # noqa: E402
from mp_osc.detectors import Analyzer  # noqa: E402
from mp_osc.osc_out import MessageBuilder  # noqa: E402
from mp_osc.tracking import Tracker  # noqa: E402

W, H = 320, 240


def floor(level=110, noise=0):
    img = np.full((H, W, 3), level, dtype=np.uint8)
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    return img


def with_shapes(img, shapes):
    """shapes: list of (kind, x, y, size, level) in pixels."""
    out = img.copy()
    for kind, x, y, size, level in shapes:
        if kind == "circle":
            cv2.circle(out, (x, y), size, (level, level, level), -1)
        else:
            cv2.rectangle(out, (x - size, y - size), (x + size, y + size), (level, level, level), -1)
    return out


def learn(det, img, n=None):
    for _ in range(n or det.learn_frames):
        det.process(img)


class ResampleTest(unittest.TestCase):
    def test_fixed_count_and_starts_at_top(self):
        square = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
        pts = resample_contour(np.roll(square, 2, axis=0), 8)
        self.assertEqual(pts.shape, (8, 2))
        np.testing.assert_allclose(pts[0], [10, 10])               # topmost-leftmost first
        self.assertAlmostEqual(float(np.linalg.norm(pts[1] - pts[0])), 20.0, places=3)  # evenly spaced

    def test_degenerate_contours(self):
        self.assertEqual(resample_contour(np.zeros((0, 2)), 4).shape, (4, 2))
        pts = resample_contour(np.array([[3, 4]], dtype=np.float32), 4)
        np.testing.assert_allclose(pts, [[3, 4]] * 4)


class BackgroundModeTest(unittest.TestCase):
    def test_learns_background_then_finds_shapes(self):
        det = BlobDetector(learn_frames=5, min_area=0.002)
        empty = floor(noise=5)
        self.assertTrue(det.learning)
        self.assertEqual(det.process(empty), [])                    # nothing while learning
        self.assertLess(det.learn_progress, 1.0)
        learn(det, empty, 4)
        self.assertFalse(det.learning)
        self.assertIsNotNone(det.background)

        self.assertEqual(det.process(empty), [])                    # empty scene -> no blobs
        scene = with_shapes(empty, [("circle", 80, 60, 20, 230), ("rect", 220, 170, 15, 20)])
        blobs = det.process(scene)
        self.assertEqual(len(blobs), 2)
        big, small = blobs                                          # biggest first
        self.assertGreater(big.area, small.area)
        self.assertAlmostEqual(big.center[0], 80 / W, delta=0.01)
        self.assertAlmostEqual(big.center[1], 60 / H, delta=0.01)
        self.assertAlmostEqual(small.center[0], 220 / W, delta=0.01)
        self.assertAlmostEqual(small.center[1], 170 / H, delta=0.01)
        self.assertAlmostEqual(big.area, np.pi * 20 ** 2 / (W * H), delta=0.002)
        x0, y0, x1, y1 = small.box
        self.assertAlmostEqual((x1 - x0) * W, 30, delta=5)          # blur/closing add a pixel or two
        self.assertAlmostEqual((y1 - y0) * H, 30, delta=5)
        self.assertEqual(big.landmarks.shape, (16, 2))
        self.assertTrue(np.all((big.landmarks >= 0) & (big.landmarks <= 1)))
        self.assertEqual(det.mask.shape, (H, W))
        self.assertGreater(int(det.mask.max()), 0)

    def test_learn_again_and_threshold(self):
        det = BlobDetector(learn_frames=3, threshold=25)
        learn(det, floor())
        scene = with_shapes(floor(), [("rect", 100, 100, 20, 125)])  # only 15 levels brighter
        self.assertEqual(det.process(scene), [])
        det.threshold = 10
        self.assertEqual(len(det.process(scene)), 1)
        det.learn()                                                   # the shape becomes background
        self.assertTrue(det.learning)
        learn(det, scene)
        self.assertEqual(det.process(scene), [])

    def test_area_limits_max_blobs_and_roi(self):
        det = BlobDetector(learn_frames=2, min_area=0.01, max_area=0.2, max_blobs=2)
        learn(det, floor())
        shapes = [("rect", 40, 20, 5, 240),       # too small (100 px < 1%)
                  ("rect", 80, 120, 75, 240),      # too big (150x150 px > 20%)
                  ("circle", 200, 200, 22, 240), ("circle", 260, 60, 24, 240), ("circle", 260, 150, 20, 240)]
        blobs = det.process(with_shapes(floor(), shapes))
        self.assertEqual(len(blobs), 2)                               # 3 valid, capped to 2 biggest
        self.assertGreaterEqual(blobs[0].area, blobs[1].area)
        det.max_blobs = 8
        self.assertEqual(len(det.process(with_shapes(floor(), shapes))), 3)
        det.roi = (0.5, 0.5, 1.0, 1.0)                               # bottom-right quadrant only
        bottom = det.process(with_shapes(floor(), shapes))
        self.assertEqual(len(bottom), 2)
        self.assertTrue(all(b.center[0] > 0.5 and b.center[1] > 0.5 for b in bottom))

    def test_adaptive_background_absorbs_changes_but_not_blobs(self):
        det = BlobDetector(learn_frames=2, adapt=0.2, min_area=0.002)
        learn(det, floor(100))
        brighter = floor(140)                                          # lighting changed everywhere
        det.process(brighter)
        self.assertEqual(int(det.mask.max()), 255)                     # first: everything differs
        for _ in range(40):
            det.process(brighter)
        self.assertEqual(det.process(brighter), [])
        self.assertEqual(int(det.mask.max()), 0)                       # absorbed
        # a blob that stays still is *not* absorbed
        scene = with_shapes(brighter, [("circle", 100, 100, 25, 250)])
        for _ in range(60):
            blobs = det.process(scene)
        self.assertEqual(len(blobs), 1)
        static = BlobDetector(learn_frames=2, adapt=0.0)
        learn(static, floor(100))
        for _ in range(40):
            static.process(brighter)
        self.assertEqual(int(static.mask.max()), 255)                  # static background never adapts


class OtherModesTest(unittest.TestCase):
    def test_bright_and_dark_thresholds(self):
        scene = with_shapes(floor(120), [("circle", 100, 100, 25, 250), ("rect", 240, 150, 20, 10)])
        bright = BlobDetector(mode="bright", threshold=200).process(scene)
        dark = BlobDetector(mode="dark", threshold=50).process(scene)
        self.assertEqual(len(bright), 1)
        self.assertEqual(len(dark), 1)
        self.assertAlmostEqual(bright[0].center[0], 100 / W, delta=0.01)
        self.assertAlmostEqual(dark[0].center[0], 240 / W, delta=0.01)
        self.assertAlmostEqual(abs(dark[0].angle) % 90, 0.0, delta=1.0)   # axis aligned square

    def test_motion_mode_needs_no_calibration(self):
        det = BlobDetector(mode="motion", threshold=16)
        for _ in range(30):
            det.process(floor(100))
        moving = det.process(with_shapes(floor(100), [("circle", 150, 120, 25, 240)]))
        self.assertEqual(len(moving), 1)
        self.assertFalse(det.learning)

    def test_mode_validation(self):
        with self.assertRaises(ValueError):
            BlobDetector(mode="nope")
        det = BlobDetector(mode="bright")
        det.set_mode("background")
        self.assertTrue(det.learning)
        self.assertEqual(set(BLOB_MODES), {"background", "motion", "bright", "dark"})
        with self.assertRaises(ValueError):
            det.set_mode("x")

    def test_orientation_of_an_elongated_shape(self):
        scene = floor(0)
        cv2.ellipse(scene, (160, 120), (60, 15), 30, 0, 360, (255, 255, 255), -1)
        b = BlobDetector(mode="bright", threshold=128).process(scene)[0]
        self.assertAlmostEqual(b.angle, 30.0, delta=3.0)
        self.assertAlmostEqual(b.size[0] * W, 120, delta=6)           # long axis
        self.assertAlmostEqual(b.size[1] * H, 30, delta=4)            # short axis
        scene = floor(0)
        cv2.ellipse(scene, (160, 120), (60, 15), -40, 0, 360, (255, 255, 255), -1)
        b = BlobDetector(mode="bright", threshold=128).process(scene)[0]
        self.assertAlmostEqual(b.angle, -40.0, delta=3.0)


class TrackingAndOscTest(unittest.TestCase):
    def test_ids_follow_moving_blobs(self):
        det = BlobDetector(learn_frames=2, min_area=0.002)
        learn(det, floor())
        tracker = Tracker(max_distance=0.25, timeout=0.5)
        ids_a, ids_b = [], []
        for i in range(10):
            scene = with_shapes(floor(), [("circle", 40 + i * 8, 60, 15, 240), ("circle", 250, 200 - i * 5, 15, 240)])
            upd = tracker.update(det.process(scene), i / 30)
            by_x = sorted(upd.active, key=lambda t: t.center[0])
            ids_a.append(by_x[0].id)
            ids_b.append(by_x[1].id)
        self.assertEqual(len(set(ids_a)), 1)
        self.assertEqual(len(set(ids_b)), 1)
        self.assertNotEqual(ids_a[0], ids_b[0])
        a = next(t for t in upd.active if t.id == ids_a[0])
        self.assertGreater(a.velocity[0], 0.0)                        # moving right
        # one disappears -> leave after the timeout
        for i in range(10, 40):
            upd = tracker.update(det.process(with_shapes(floor(), [("circle", 120, 60, 15, 240)])), i / 30)
            if upd.left:
                break
        self.assertEqual(upd.left, [ids_b[0]])

    def test_mp_messages_for_blobs(self):
        cfg = Config(blobs=True, preview=False)
        det = BlobDetector(learn_frames=2, min_area=0.002)
        learn(det, floor())
        blobs = det.process(with_shapes(floor(), [("circle", 80, 60, 20, 230), ("rect", 220, 170, 12, 20)]))
        tracker = Tracker()
        upd = tracker.update(blobs, 0.0)
        msgs = MessageBuilder(cfg).frame(3, 0.1, 30.0, [], [], [], [], [], [], [], [], [],
                                         upd.active, upd.entered, upd.left)
        by_addr = {}
        for a, args in msgs:
            by_addr.setdefault(a, []).append(args)
        self.assertEqual(by_addr["/mp/frame"][0][3:], [0, 0, 0, 2])
        self.assertEqual(by_addr["/mp/blob/enter"], [[0], [1]])
        self.assertEqual(by_addr["/mp/blobs"], [[2, 0, 1]])         # count + ids, biggest first
        self.assertEqual(len(by_addr["/mp/blob/center"]), 2)
        self.assertEqual(by_addr["/mp/blob/center"][0][0], 0)
        self.assertAlmostEqual(by_addr["/mp/blob/center"][0][1], 80 / W, delta=0.01)
        self.assertEqual(len(by_addr["/mp/blob/bbox"][0]), 5)
        self.assertEqual(len(by_addr["/mp/blob/velocity"][0]), 4)
        info = by_addr["/mp/blob/info"][0]
        self.assertEqual(len(info), 5)                                 # id area angle w h
        self.assertAlmostEqual(info[1], np.pi * 400 / (W * H), delta=0.002)
        self.assertEqual(len(by_addr["/mp/blob/contour"][0]), 1 + 16 * 2)
        self.assertTrue(all(type(v) in (int, float) for args in by_addr["/mp/blob/contour"] for v in args))

        # leave + no blobs: the count message still goes out (so receivers see 0)
        upd = tracker.update([], 1.0)
        msgs = MessageBuilder(cfg).frame(4, 1.0, 30.0, [], [], [], [], [], [], [], [], [],
                                         upd.active, upd.entered, upd.left)
        addrs = [a for a, _ in msgs]
        self.assertEqual(addrs.count("/mp/blob/leave"), 2)
        self.assertIn(("/mp/blobs", [0]), msgs)
        self.assertEqual(len(MessageBuilder(cfg).leave_all([], [], [], [4, 5])), 2)

        # --no-landmarks drops the contour, address id style moves the id
        cfg2 = Config(blobs=True, preview=False, send_landmarks=False, id_style="address")
        upd = Tracker().update(blobs, 0.0)
        msgs = MessageBuilder(cfg2).frame(3, 0.1, 30.0, [], [], [], [], [], [], [], [], [],
                                          upd.active, upd.entered, upd.left)
        addrs = [a for a, _ in msgs]
        self.assertIn("/mp/blob/0/center", addrs)
        self.assertNotIn("/mp/blob/0/contour", addrs)

    def test_kinect_format_comes_from_blobs_when_enabled(self):
        det = BlobDetector(learn_frames=2, min_area=0.002)
        learn(det, floor())
        blobs = det.process(with_shapes(floor(), [("circle", 80, 60, 20, 230)]))
        upd = Tracker().update(blobs, 0.0)
        cfg = Config(blobs=True, formats=("kinect",), preview=False)
        msgs = MessageBuilder(cfg).frame(0, 0.0, 30.0, [], [], [], [], [], [], [], [], [],
                                         upd.active, upd.entered, upd.left)
        self.assertEqual(len(msgs), 1)
        addr, args = msgs[0]
        self.assertEqual(addr, "/kinect/person")
        self.assertEqual(args[0], 0)
        self.assertAlmostEqual(args[1], 80 / W, delta=0.01)
        self.assertAlmostEqual(args[2], 60 / H, delta=0.01)
        self.assertEqual(args[3], 0.0)
        # explicit choice: poses (none here) -> nothing
        cfg = Config(blobs=True, pose=True, formats=("kinect",), kinect_from="pose", preview=False)
        msgs = MessageBuilder(cfg).frame(0, 0.0, 30.0, [], [], [], [], [], [], [], [], [],
                                         upd.active, upd.entered, upd.left)
        self.assertEqual(msgs, [])


class ConfigAndAnalyzerTest(unittest.TestCase):
    def test_cli_options(self):
        cfg = parse_args(["--blobs", "--blob-mode", "dark", "--blob-threshold", "40",
                          "--blob-min-area", "0.01", "--blob-roi", "0.1", "0.2", "0.9", "0.8", "--no-preview"])
        self.assertTrue(cfg.blobs)
        self.assertFalse(cfg.pose)                 # blobs alone: pose is not switched on by default
        self.assertEqual(cfg.blob_mode, "dark")
        self.assertEqual(cfg.blob_threshold, 40)
        self.assertEqual(cfg.blob_roi, (0.1, 0.2, 0.9, 0.8))
        self.assertTrue(cfg.kinect_source_is_blobs)
        self.assertEqual(cfg.effective_kinect_stream, "depth")
        self.assertEqual(parse_args(["--blobs", "--pose"]).effective_kinect_stream, "color")
        self.assertEqual(parse_args(["--blobs", "--kinect-stream", "color"]).effective_kinect_stream, "color")
        with self.assertRaises(ValueError):
            Config(blob_threshold=0)
        with self.assertRaises(ValueError):
            Config(blob_min_area=0.6, blob_max_area=0.5)
        with self.assertRaises(ValueError):
            Config(blob_roi=(0, 0, 2, 1))

    def test_analyzer_runs_blobs_without_models(self):
        cfg = Config(blobs=True, blob_mode="bright", blob_threshold=200, preview=False, min_confidence=0.5)
        a = Analyzer(cfg)
        self.assertTrue(a.blobs_enabled)
        self.assertFalse(a.pose_enabled)
        scene = with_shapes(floor(120), [("circle", 100, 100, 25, 250)])
        det = a.analyze(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB), 0)
        self.assertEqual(len(det.blobs), 1)
        self.assertEqual(det.poses, [])
        a.configure(False, False, False, False, False)
        self.assertFalse(a.blobs_enabled)
        self.assertEqual(a.analyze(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB), 1).blobs, [])
        a.close()


class AppBlobsTest(unittest.TestCase):
    """The whole pipeline on a synthetic video: ids, enter/leave, runtime controls."""

    class CapturingSender:
        def __init__(self):
            self.messages, self.sent, self.paused = [], 0, False
            self.ip, self.port = "127.0.0.1", 5005

        def send(self, messages):
            if not self.paused:
                self.messages.extend(messages)
                self.sent += len(messages)

        def retarget(self, ip, port):
            self.ip, self.port = ip, int(port)

        def by_address(self, address):
            return [args for a, args in self.messages if a == address]

    def make_video(self, path):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for _ in range(10):
            writer.write(floor())                                      # empty scene: background
        for i in range(30):
            writer.write(with_shapes(floor(), [("circle", 60 + i * 5, 120, 22, 240)]))
        for _ in range(25):
            writer.write(floor())                                      # gone again
        writer.release()

    def test_blob_enters_moves_and_leaves(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "overhead.mp4")
            self.make_video(path)
            cfg = Config(source=path, blobs=True, blob_learn_frames=8, blob_min_area=0.002, preview=False,
                         quiet=True, loop=False, formats=("mp", "kinect"))
            sender = self.CapturingSender()
            app = App(cfg, sender=sender)
            self.assertEqual(app.analyzer_names(), ["blobs"])
            app.run()
        self.assertEqual(sender.by_address("/mp/blob/enter"), [[0]])
        self.assertEqual(sender.by_address("/mp/blob/leave"), [[0]])
        centers = sender.by_address("/mp/blob/center")
        self.assertGreater(len(centers), 20)
        self.assertLess(centers[0][1], centers[-1][1])                 # moved to the right
        kin = sender.by_address("/kinect/person")
        self.assertEqual(len(kin), len(centers))
        self.assertTrue(all(0.0 <= v <= 1.0 for args in kin for v in args[1:]))
        speeds = [args[3] for args in kin]
        self.assertGreater(max(speeds), 0.0)
        self.assertEqual(sender.messages[-1][1][0], "stopped")

    def test_runtime_controls(self):
        cfg = Config(source=os.path.join(ROOT, "brother-977170_1280.jpg"), blobs=True, blob_mode="dark",
                     blob_threshold=60, preview=False, quiet=True)
        sender = self.CapturingSender()
        app = App(cfg, sender=sender)
        try:
            app.step()
            self.assertGreater(app.counts[3], 0)
            app.step_blob_threshold(+1)
            self.assertEqual(app.cfg.blob_threshold, 65)
            self.assertEqual(app.analyzer.blob_detector.threshold, 65)
            app.set_blob_threshold(8)
            app.step_blob_threshold(-1)
            self.assertEqual(app.cfg.blob_threshold, 7)                 # fine steps below 10
            app.step_blob_min_area(+1)
            self.assertAlmostEqual(app.cfg.blob_min_area, 0.006)
            self.assertAlmostEqual(app.analyzer.blob_detector.min_area, 0.006)

            app.set_detectors(blobs=False)
            self.assertFalse(app.analyzer.blobs_enabled)
            self.assertTrue(sender.by_address("/mp/blob/leave"))        # receivers were told
            app.step()
            self.assertEqual(app.counts[3], 0)
            app.set_detectors(blobs=True)
            app.learn_background()                                       # harmless in 'dark' mode
            app.step()
            self.assertGreater(app.counts[3], 0)
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
