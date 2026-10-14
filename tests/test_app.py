"""End-to-end tests: real MediaPipe models on still images and a synthetic video.

No webcam or display is needed: the app is driven with ``--source <image|video>``
and ``--no-preview``. The OSC side is checked both with an in-process capturing
sender and over a real UDP socket.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc.app import App  # noqa: E402
from mp_osc.config import Config  # noqa: E402

PORTRAIT = os.path.join(ROOT, "brother-977170_1280.jpg")   # 1 person, 2 faces, 1 hand
GROUP = os.path.join(ROOT, "Equipo_taller5_2023.jpg")      # 5 people (lite model finds 2)


class CapturingSender:
    """Stands in for OscSender: keeps every (address, args) in memory."""

    def __init__(self):
        self.messages = []
        self.sent = 0

    def send(self, messages):
        self.messages.extend(messages)
        self.sent += len(messages)

    def by_address(self, address):
        return [args for a, args in self.messages if a == address]

    def per_frame(self, address):
        """One entry per /mp/frame: the args of ``address`` in that frame, or None."""
        out = []
        for a, args in self.messages:
            if a == "/mp/frame":
                out.append(None)
            elif a == address and out:
                out[-1] = args
        return out


def make_video(path, frames=24, step=12, width=640, height=480):
    """Slide a crop window over the portrait so the person moves right-to-left."""
    image = cv2.imread(PORTRAIT)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    assert writer.isOpened()
    y0 = 60
    for i in range(frames):
        x0 = i * step
        writer.write(image[y0:y0 + height, x0:x0 + width])
    writer.release()


class StillImageTest(unittest.TestCase):
    def run_app(self, **kwargs):
        cfg = Config(preview=False, quiet=True, frames=4, **kwargs)
        sender = CapturingSender()
        App(cfg, sender=sender).run()
        return sender

    def test_pose_hands_face_on_portrait(self):
        s = self.run_app(source=PORTRAIT, pose=True, hands=True, face=True)

        addrs = [a for a, _ in s.messages]
        self.assertEqual(addrs[0], "/mp/status")
        self.assertEqual(s.messages[0][1][0], "started")
        self.assertEqual(s.messages[-1][1][0], "stopped")
        self.assertEqual(addrs[1], "/mp/face/blendshape_names")

        frames = s.by_address("/mp/frame")
        self.assertEqual(len(frames), 4)
        self.assertEqual([f[0] for f in frames], [0, 1, 2, 3])
        self.assertEqual(frames[-1][3:], [1, 1, 2, 0])   # poses, hands, faces, blobs

        centers = s.by_address("/mp/pose/center")
        self.assertEqual(len(centers), 4)
        self.assertEqual(centers[0][0], 0)
        self.assertTrue(0.0 < centers[0][1] < 1.0 and 0.0 < centers[0][2] < 1.0)

        lm = s.by_address("/mp/pose/landmarks")[0]
        self.assertEqual(len(lm), 1 + 33 * 4)
        self.assertTrue(all(isinstance(v, float) for v in lm[1:]))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in lm[4::4]))  # visibility column

        self.assertEqual(s.by_address("/mp/pose/enter"), [[0]])
        self.assertEqual(s.by_address("/mp/pose/leave"), [[0]])     # sent on shutdown
        self.assertEqual(sorted(a[0] for a in s.by_address("/mp/face/enter")), [0, 1])
        self.assertEqual(sorted(a[0] for a in s.by_address("/mp/face/leave")), [0, 1])

        info = s.by_address("/mp/hand/info")[0]
        self.assertIn(info[1], ("Left", "Right"))
        self.assertGreater(s.by_address("/mp/hand/pinch")[0][1], 0.0)
        self.assertEqual(len(s.by_address("/mp/hand/landmarks")[0]), 1 + 21 * 3)
        self.assertNotIn("/mp/hand/gesture", addrs)

        blend = s.by_address("/mp/face/blendshapes")[0]
        self.assertEqual(len(blend), 53)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in blend[1:]))
        self.assertNotIn("/mp/face/landmarks", addrs)

    def test_gesture_recognizer(self):
        s = self.run_app(source=PORTRAIT, gesture=True)
        gestures = s.by_address("/mp/hand/gesture")
        self.assertEqual(len(gestures), 4)
        self.assertIsInstance(gestures[0][1], str)
        self.assertIsInstance(gestures[0][2], float)
        self.assertEqual(len(s.by_address("/mp/hand/landmarks")[0]), 1 + 21 * 3)
        self.assertEqual(s.by_address("/mp/frame")[0][3], 0)   # pose not enabled

    def test_multiple_people_get_distinct_ids(self):
        s = self.run_app(source=GROUP, pose=True, max_poses=6)
        ids = sorted(a[0] for a in s.by_address("/mp/pose/enter"))
        self.assertGreaterEqual(len(ids), 2)
        self.assertEqual(ids, list(range(len(ids))))
        per_frame = s.by_address("/mp/frame")[-1][3]
        self.assertEqual(per_frame, len(ids))
        xs = sorted(c[1] for c in s.by_address("/mp/pose/center") if c[0] in ids[:2])
        self.assertGreater(xs[-1] - xs[0], 0.1)   # different people, different x

    def test_optional_outputs_and_address_style(self):
        s = self.run_app(source=PORTRAIT, pose=True, face=True, id_style="address",
                         send_world=True, send_face_landmarks=True, send_landmarks=False, prefix="/cam")
        addrs = {a for a, _ in s.messages}
        self.assertIn("/cam/pose/0/center", addrs)
        self.assertIn("/cam/pose/0/world", addrs)
        self.assertNotIn("/cam/pose/0/landmarks", addrs)
        self.assertIn("/cam/face/0/landmarks", addrs)
        self.assertEqual(len(s.by_address("/cam/face/0/landmarks")[0]), 478 * 3)
        self.assertEqual(len(s.by_address("/cam/pose/0/world")[0]), 33 * 3)

    def test_legacy_format_matches_processing_parser(self):
        s = self.run_app(source=PORTRAIT, pose=True, hands=True, formats=("legacy",))
        addrs = {a for a, _ in s.messages}
        self.assertIn("pose_0", addrs)
        self.assertIn("hand_landmarks", addrs)
        self.assertNotIn("/mp/frame", addrs)
        # the Processing sketch splits each string argument on spaces and reads 3 floats
        for arg in s.by_address("pose_0")[0]:
            parts = arg.split(" ")
            self.assertEqual(len(parts), 5)
            xyz = [float(p) for p in parts[:3]]
            self.assertEqual(len(xyz), 3)


class VideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "walk.mp4")
        make_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_video(self, **kwargs):
        cfg = Config(source=self.video, preview=False, quiet=True, loop=False, pose=True, **kwargs)
        sender = CapturingSender()
        App(cfg, sender=sender).run()
        return sender

    def test_id_stays_stable_while_person_moves(self):
        s = self.run_video()
        self.assertEqual(len(s.by_address("/mp/frame")), 24)   # --no-loop: stops at the end
        self.assertEqual(s.by_address("/mp/pose/enter"), [[0]])
        centers = s.by_address("/mp/pose/center")
        self.assertEqual(len(centers), 24)
        self.assertTrue(all(c[0] == 0 for c in centers))
        # the crop window moves right, so the person drifts left in the frame
        self.assertLess(centers[-1][1], centers[0][1] - 0.2)

    def test_velocity_points_left_and_speed_is_positive(self):
        s = self.run_video()
        vel = np.array([v[1:] for v in s.by_address("/mp/pose/velocity")[3:]])
        # The crop moves 12 px/frame at 30 fps over a 640 px frame: vx ~ -0.56
        # units/s. Landmark noise is large on this synthetic clip, so check the
        # statistics, not every frame.
        self.assertLess(float(np.median(vel[:, 0])), -0.3)
        self.assertGreater(float((vel[:, 0] < 0).mean()), 0.7)
        self.assertTrue(all(v[2] > 0 for v in vel))             # speed
        for v in vel:
            self.assertAlmostEqual(v[2], float(np.hypot(v[0], v[1])), places=4)

    def test_video_timestamps_follow_the_file_frame_rate(self):
        s = self.run_video()
        times = [f[1] for f in s.by_address("/mp/frame")]
        np.testing.assert_allclose(np.diff(times), 1 / 30, atol=1e-6)

    def test_mirror_flips_x(self):
        # Full-body people (group photo): the model is symmetric enough there
        # to check x_mirrored ~= 1 - x. (On tightly cropped bodies the model
        # itself is asymmetric by up to ~0.1, so the synthetic clip is unusable.)
        def centers(mirror):
            cfg = Config(source=GROUP, preview=False, quiet=True, frames=1, pose=True,
                         max_poses=6, mirror=mirror)
            sender = CapturingSender()
            App(cfg, sender=sender).run()
            return [c[1] for c in sender.by_address("/mp/pose/center")]

        plain, mirrored = centers(False), centers(True)
        self.assertGreaterEqual(len(plain), 1)
        self.assertGreaterEqual(len(mirrored), 1)
        # the detector may not find exactly the same people in both runs, so
        # every mirrored person must correspond to somebody in the plain run
        for m in mirrored:
            closest = min(plain, key=lambda x: abs(x - (1.0 - m)))
            self.assertAlmostEqual(closest + m, 1.0, delta=0.06)

    def test_smoothing_reduces_jitter(self):
        raw = np.array([c[1:3] for c in self.run_video().by_address("/mp/pose/center")])
        smooth = np.array([c[1:3] for c in self.run_video(smooth=0.8).by_address("/mp/pose/center")])
        jitter = lambda a: np.abs(np.diff(a, n=2, axis=0)).mean()  # noqa: E731
        self.assertLess(jitter(smooth), jitter(raw))

    def test_kinect_format_values_in_range(self):
        s = self.run_video(formats=("kinect",), kinect_y="size", max_speed=1.0)
        people = s.by_address("/kinect/person")
        self.assertEqual(len(people), 24)
        for pid, x, y, speed in people:
            self.assertIsInstance(pid, int)
            for v in (x, y, speed):
                self.assertTrue(0.0 <= v <= 1.0)
        self.assertGreater(max(p[3] for p in people), 0.0)


class UdpTest(unittest.TestCase):
    """Messages really go out through python-osc over UDP."""

    def setUp(self):
        self.received = []
        self.lock = threading.Lock()
        d = Dispatcher()
        d.set_default_handler(lambda addr, *args: self.received.append((addr, list(args))))
        self.server = ThreadingOSCUDPServer(("127.0.0.1", 0), d)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def wait_for(self, address, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(a == address for a, _ in self.received):
                return
            time.sleep(0.05)
        self.fail(f"never received {address}")

    def test_messages_over_udp(self):
        cfg = Config(source=PORTRAIT, pose=True, preview=False, quiet=True, frames=3, port=self.port)
        App(cfg).run()
        self.wait_for("/mp/status")
        time.sleep(0.2)
        addrs = [a for a, _ in self.received]
        self.assertEqual(addrs.count("/mp/frame"), 3)
        center = next(args for a, args in self.received if a == "/mp/pose/center")
        self.assertIsInstance(center[0], int)
        self.assertIsInstance(center[1], float)

    def test_bundle_over_udp(self):
        cfg = Config(source=PORTRAIT, pose=True, preview=False, quiet=True, frames=2,
                     port=self.port, bundle=True)
        App(cfg).run()
        self.wait_for("/mp/status")
        time.sleep(0.2)
        addrs = [a for a, _ in self.received]
        self.assertEqual(addrs.count("/mp/frame"), 2)
        self.assertIn("/mp/pose/landmarks", addrs)

    def test_kinect_format_is_accepted_by_sonido_cuadrafonico_receiver(self):
        """The repository's own quad-sound receiver must understand our /kinect/person."""
        self.server.shutdown()
        self.server.server_close()
        sys.path.insert(0, os.path.join(ROOT, "Sonido_Cuadrafonico"))
        from recepcion_datos import ReceptorDatosOSC

        receptor = ReceptorDatosOSC(puerto=self.port, tiempo_sin_datos=0.5)
        receptor.iniciar()
        try:
            cfg = Config(source=PORTRAIT, pose=True, preview=False, quiet=True, frames=3,
                         port=self.port, formats=("kinect",))
            App(cfg).run()
            time.sleep(0.2)
            personas = receptor.personas_activas()
            self.assertEqual(list(personas), [0])
            p = personas[0]
            self.assertTrue(0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0)
            self.assertIn(("entra", 0), receptor.drenar_eventos())
        finally:
            receptor.detener()


class CliTest(unittest.TestCase):
    def test_command_line_run(self):
        cmd = [sys.executable, os.path.join(ROOT, "mediapipe_osc.py"), "--source", PORTRAIT,
               "--pose", "--frames", "2", "--no-preview", "--verbose", "--port", "5999"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertIn("/mp/status started", res.stdout)
        self.assertIn("/mp/pose/center 0 ", res.stdout)
        self.assertIn("stopped after 2 frames", res.stdout)

    def test_missing_source_is_reported(self):
        cmd = [sys.executable, os.path.join(ROOT, "mediapipe_osc.py"), "--source", "/nonexistent.mp4",
               "--no-preview"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ROOT)
        self.assertEqual(res.returncode, 1)
        self.assertIn("video file not found", res.stderr)


if __name__ == "__main__":
    unittest.main()
