"""Unit tests for the OSC message schema (no camera / MediaPipe inference required)."""

import os
import sys
import unittest

import numpy as np
from pythonosc.osc_message import OscMessage
from pythonosc.osc_bundle import OscBundle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mp_osc.config import Config, parse_args  # noqa: E402
from mp_osc.detectors import (  # noqa: E402
    BLENDSHAPE_NAMES, FaceDetection, HandDetection, PoseDetection,
)
from mp_osc.osc_out import MessageBuilder, encode  # noqa: E402
from mp_osc.tracking import Tracker  # noqa: E402


def fake_pose(x=0.5, y=0.5):
    lms = np.zeros((33, 5), dtype=np.float32)
    lms[:, 0] = x
    lms[:, 1] = y
    lms[:, 3] = 1.0  # visibility
    lms[:, 4] = 1.0  # presence
    lms[1:3, 1] = y - 0.3    # eyes above ...
    lms[27:29, 1] = y + 0.3  # ... ankles below: symmetric, so the center stays at (x, y)
    return PoseDetection(landmarks=lms, world=np.zeros((33, 3), dtype=np.float32))


def fake_hand(x=0.3, y=0.3, gesture=""):
    lms = np.full((21, 3), (x, y, 0.0), dtype=np.float32)
    world = np.zeros((21, 3), dtype=np.float32)
    world[4] = (0.0, 0.0, 0.0)
    world[8] = (0.03, 0.04, 0.0)  # pinch = 0.05 m
    return HandDetection(landmarks=lms, world=world, handedness="Right", handedness_score=0.9,
                         gesture=gesture, gesture_score=0.8 if gesture else 0.0)


def fake_face(x=0.6, y=0.4):
    lms = np.full((478, 3), (x, y, 0.0), dtype=np.float32)
    return FaceDetection(landmarks=lms, blendshapes=np.linspace(0, 1, 52, dtype=np.float32))


def build(cfg, poses=(), hands=(), faces=(), now=0.0):
    """Run detections through fresh trackers and the message builder."""
    tp, th, tf = Tracker(), Tracker(), Tracker()
    up_p, up_h, up_f = tp.update(list(poses), now), th.update(list(hands), now), tf.update(list(faces), now)
    mb = MessageBuilder(cfg)
    msgs = mb.frame(7, 1.5, 30.0, up_p.active, up_p.entered, up_p.left,
                    up_h.active, up_h.entered, up_h.left, up_f.active, up_f.entered, up_f.left)
    return {a: args for a, args in msgs}, msgs


class MpFormatTest(unittest.TestCase):
    def test_frame_message_and_pose_messages(self):
        cfg = Config(pose=True, hands=True, face=True, gesture=True, preview=False)
        by_addr, msgs = build(cfg, poses=[fake_pose()], hands=[fake_hand(gesture="Thumb_Up")], faces=[fake_face()])

        self.assertEqual(by_addr["/mp/frame"], [7, 1.5, 30.0, 1, 1, 1, 0])
        self.assertEqual(by_addr["/mp/pose/enter"], [0])
        self.assertEqual(by_addr["/mp/pose/center"][0], 0)
        np.testing.assert_allclose(by_addr["/mp/pose/center"][1:], [0.5, 0.5, 0.0], atol=1e-6)
        self.assertEqual(len(by_addr["/mp/pose/landmarks"]), 1 + 33 * 4)
        self.assertEqual(len(by_addr["/mp/pose/velocity"]), 4)
        xmin, ymin, xmax, ymax = by_addr["/mp/pose/bbox"][1:]
        self.assertAlmostEqual(ymin, 0.2, places=5)
        self.assertAlmostEqual(ymax, 0.8, places=5)
        self.assertNotIn("/mp/pose/world", by_addr)

        self.assertEqual(by_addr["/mp/hand/info"], [0, "Right", 0.9])
        self.assertAlmostEqual(by_addr["/mp/hand/pinch"][1], 0.05, places=5)
        self.assertEqual(by_addr["/mp/hand/gesture"], [0, "Thumb_Up", 0.8])
        self.assertEqual(len(by_addr["/mp/hand/landmarks"]), 1 + 21 * 3)

        self.assertEqual(len(by_addr["/mp/face/blendshapes"]), 1 + 52)
        self.assertNotIn("/mp/face/landmarks", by_addr)

        # messages are ordered: frame first, enter before data
        addrs = [a for a, _ in msgs]
        self.assertEqual(addrs[0], "/mp/frame")
        self.assertLess(addrs.index("/mp/pose/enter"), addrs.index("/mp/pose/center"))

    def test_all_arguments_are_plain_python_types(self):
        cfg = Config(pose=True, hands=True, face=True, send_world=True, send_face_landmarks=True)
        _, msgs = build(cfg, poses=[fake_pose()], hands=[fake_hand()], faces=[fake_face()])
        for addr, args in msgs:
            for a in args:
                self.assertIn(type(a), (int, float, str), f"{addr}: {type(a)}")

    def test_optional_messages(self):
        cfg = Config(pose=True, face=True, send_world=True, send_face_landmarks=True,
                     send_landmarks=False, send_blendshapes=False)
        by_addr, _ = build(cfg, poses=[fake_pose()], faces=[fake_face()])
        self.assertIn("/mp/pose/world", by_addr)
        self.assertEqual(len(by_addr["/mp/pose/world"]), 1 + 33 * 3)
        self.assertNotIn("/mp/pose/landmarks", by_addr)
        self.assertIn("/mp/face/landmarks", by_addr)
        self.assertEqual(len(by_addr["/mp/face/landmarks"]), 1 + 478 * 3)
        self.assertNotIn("/mp/face/blendshapes", by_addr)

    def test_address_id_style_and_prefix(self):
        cfg = Config(pose=True, id_style="address", prefix="body")
        by_addr, _ = build(cfg, poses=[fake_pose(), fake_pose(0.9, 0.5)])
        self.assertEqual(by_addr["/body/frame"][3], 2)
        np.testing.assert_allclose(by_addr["/body/pose/0/center"], [0.5, 0.5, 0.0], atol=1e-6)
        self.assertAlmostEqual(by_addr["/body/pose/1/center"][0], 0.9, places=6)
        self.assertNotIn("/body/pose/center", by_addr)

    def test_multiple_people_in_arg_style(self):
        cfg = Config(pose=True)
        _, msgs = build(cfg, poses=[fake_pose(0.2, 0.5), fake_pose(0.8, 0.5)])
        centers = [args for a, args in msgs if a == "/mp/pose/center"]
        self.assertEqual([c[0] for c in centers], [0, 1])
        np.testing.assert_allclose([c[1] for c in centers], [0.2, 0.8], atol=1e-6)
        enters = [args for a, args in msgs if a == "/mp/pose/enter"]
        self.assertEqual(enters, [[0], [1]])

    def test_status_and_blendshape_names(self):
        cfg = Config(face=True)
        msgs = MessageBuilder(cfg).status("started")
        self.assertEqual(msgs[0][0], "/mp/status")
        self.assertEqual(msgs[0][1][0], "started")
        self.assertEqual(msgs[1], ("/mp/face/blendshape_names", BLENDSHAPE_NAMES))
        self.assertEqual(len(MessageBuilder(Config(pose=True)).status("stopped")), 1)

    def test_leave_all(self):
        cfg = Config(pose=True, hands=True)
        msgs = MessageBuilder(cfg).leave_all([0, 1], [3], [])
        self.assertEqual(msgs, [("/mp/pose/leave", [0]), ("/mp/pose/leave", [1]), ("/mp/hand/leave", [3])])


class LegacyFormatTest(unittest.TestCase):
    def test_legacy_strings(self):
        cfg = Config(pose=True, hands=True, face=True, gesture=True, formats=("legacy",))
        by_addr, msgs = build(cfg, poses=[fake_pose()], hands=[fake_hand(gesture="Open_Palm")], faces=[fake_face()])
        self.assertNotIn("/mp/frame", by_addr)
        pose = by_addr["pose_0"]
        self.assertEqual(len(pose), 33)
        self.assertEqual(pose[5], "0.500 0.500 0.000 1.000 1.000")   # x y z visibility presence
        self.assertEqual(pose[1], "0.500 0.200 0.000 1.000 1.000")
        hand = by_addr["hand_landmarks"]
        self.assertEqual(len(hand), 21)
        self.assertEqual(hand[5], "0 5 0.300 0.300 0.000")              # hand idx x y z
        self.assertTrue(by_addr["distancia_pulgar_indice"][0].startswith("0 0.05"))
        self.assertEqual(by_addr["/gesture"], ["Open_Palm 0.80"])
        self.assertEqual(len(by_addr["face_0"]), 478)
        self.assertEqual(by_addr["face_0"][0], "0.600 0.400 0.000")

    def test_formats_can_be_combined(self):
        cfg = Config(pose=True, formats=("mp", "legacy", "kinect"))
        by_addr, _ = build(cfg, poses=[fake_pose()])
        self.assertIn("/mp/pose/center", by_addr)
        self.assertIn("pose_0", by_addr)
        self.assertIn("/kinect/person", by_addr)


class KinectFormatTest(unittest.TestCase):
    def test_kinect_person_message(self):
        cfg = Config(pose=True, formats=("kinect",))
        by_addr, msgs = build(cfg, poses=[fake_pose(0.25, 0.75)])
        self.assertEqual(list(by_addr), ["/kinect/person"])
        pid, x, y, speed = by_addr["/kinect/person"]
        self.assertIsInstance(pid, int)
        np.testing.assert_allclose((x, y, speed), (0.25, 0.75, 0.0), atol=1e-6)

    def test_invert_x_and_size_y(self):
        cfg = Config(pose=True, formats=("kinect",), invert_x=True, kinect_y="size")
        by_addr, _ = build(cfg, poses=[fake_pose(0.25, 0.5)])
        _, x, y, _ = by_addr["/kinect/person"]
        self.assertAlmostEqual(x, 0.75, places=5)
        self.assertAlmostEqual(y, 1.0 - 0.6, places=5)  # bbox height 0.6 -> y 0.4

    def test_speed_normalized_and_clipped(self):
        cfg = Config(pose=True, formats=("kinect",), max_speed=1.0)
        tp = Tracker(velocity_smooth=0.0, max_distance=10.0)
        tp.update([fake_pose(0.0, 0.5)], 0.0)
        up = tp.update([fake_pose(0.5, 0.5)], 0.1)  # 5 units/s -> clipped to 1.0
        msgs = MessageBuilder(cfg).frame(0, 0, 0, up.active, [], [], [], [], [], [], [], [])
        self.assertEqual(msgs[0][1][3], 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in msgs[0][1][1:]))


class EncodeTest(unittest.TestCase):
    def test_types_survive_a_roundtrip(self):
        msg = encode("/mp/test", [3, 0.5, "hello", np.float32(0.25), np.int64(9), True])
        parsed = OscMessage(msg.dgram)
        self.assertEqual(parsed.address, "/mp/test")
        self.assertEqual(parsed.params, [3, 0.5, "hello", 0.25, 9, True])
        self.assertIsInstance(parsed.params[0], int)
        self.assertIsInstance(parsed.params[1], float)

    def test_legacy_address_without_slash(self):
        parsed = OscMessage(encode("pose_0", ["0.1 0.2 0.3"]).dgram)
        self.assertEqual(parsed.address, "pose_0")

    def test_large_message_fits_in_a_datagram(self):
        msg = encode("/mp/face/landmarks", [0] + [0.5] * (478 * 3))
        self.assertLess(len(msg.dgram), 65507)

    def test_bundle(self):
        from pythonosc import osc_bundle_builder
        bb = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        bb.add_content(encode("/a", [1]))
        bb.add_content(encode("/b", [2.0]))
        bundle = OscBundle(bb.build().dgram)
        self.assertEqual(bundle.num_contents, 2)


class ConfigTest(unittest.TestCase):
    def test_defaults_enable_pose(self):
        cfg = parse_args([])
        self.assertTrue(cfg.pose)
        self.assertFalse(cfg.hands)
        self.assertEqual(cfg.source, "0")
        self.assertEqual(cfg.formats, ("mp",))

    def test_gesture_implies_hands(self):
        cfg = parse_args(["--gesture"])
        self.assertTrue(cfg.hands and cfg.gesture and not cfg.pose)

    def test_legacy_cam_args(self):
        cfg = parse_args(["--cam", "2", "--cam_width", "1280", "--cam_height", "720"])
        self.assertEqual((cfg.source, cfg.width, cfg.height), ("2", 1280, 720))

    def test_multiple_formats_and_prefix_normalisation(self):
        cfg = parse_args(["--format", "mp", "kinect", "--prefix", "cam1/"])
        self.assertEqual(cfg.formats, ("mp", "kinect"))
        self.assertEqual(cfg.prefix, "/cam1")

    def test_invalid_smooth_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--smooth", "1.5"])


if __name__ == "__main__":
    unittest.main()
