"""Tests for the receiver examples (python-examples, processing-examples,
openFrameworks-examples, touchdesigner-examples) and for osc_simulator.py.

* the simulator must speak exactly the app's schema (same addresses, same
  argument types as mp_osc.osc_out.MessageBuilder);
* every Python example runs headless against the simulator and reports data;
* every Processing sketch compiles against the stand-in PApplet/oscP5 classes in
  tests/stubs/processing and handles the simulator's messages (needs javac);
* every openFrameworks app compiles against the stand-in ofMain.h/ofxOsc.h in
  tests/stubs/openframeworks and handles the messages (needs g++);
* every TouchDesigner build script builds its network in the stand-in
  TouchDesigner API of tests/stubs/touchdesigner and parses the messages.

Tests that need a missing tool (javac, g++) are skipped.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pythonosc.osc_message import OscMessage  # noqa: E402

from mp_osc.config import Config  # noqa: E402
from mp_osc.osc_out import MessageBuilder  # noqa: E402
from mp_osc.tracking import Track  # noqa: E402
import numpy as np  # noqa: E402

import osc_simulator  # noqa: E402

EXAMPLES = ["body_landmarks", "hand_landmarks", "hand_gestures", "face_landmarks", "blobs"]
STUBS = os.path.join(ROOT, "tests", "stubs")
PYTHON = sys.executable


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def simulator(port, seconds):
    return subprocess.Popen([PYTHON, os.path.join(ROOT, "osc_simulator.py"), "--port", str(port),
                             "--seconds", str(seconds), "--quiet"])


def signature(args):
    return "".join("i" if isinstance(a, int) and not isinstance(a, bool) else
                   "f" if isinstance(a, float) else "s" if isinstance(a, str) else "?" for a in args)


# ---------------------------------------------------------------------------
class SimulatorSchemaTest(unittest.TestCase):
    """The simulator's messages have the addresses and types of the real app."""

    def app_messages(self):
        cfg = Config(pose=True, hands=True, gesture=True, face=True, blobs=True, send_face_landmarks=True)
        mb = MessageBuilder(cfg)
        from mp_osc.blobs import BlobDetection
        from mp_osc.detectors import FaceDetection, HandDetection, PoseDetection
        z2, z3 = np.zeros(2, np.float32), np.zeros(3, np.float32)

        def track(detection, landmarks):
            return Track(id=0, center=z3.copy(), detection=detection, first_seen=0.0, last_seen=0.0, landmarks=landmarks)

        pose = track(PoseDetection(landmarks=np.zeros((33, 5), np.float32), world=np.zeros((33, 3), np.float32)),
                     np.zeros((33, 5), np.float32))
        hand = track(HandDetection(landmarks=np.zeros((21, 3), np.float32), world=np.zeros((21, 3), np.float32),
                                   handedness="Left", handedness_score=0.9, gesture="None", gesture_score=0.5),
                     np.zeros((21, 3), np.float32))
        face = track(FaceDetection(landmarks=np.zeros((478, 3), np.float32), blendshapes=np.zeros(52, np.float32)),
                     np.zeros((478, 3), np.float32))
        blob = track(BlobDetection(center=z3.copy(), landmarks=np.zeros((16, 2), np.float32), box=np.zeros(4, np.float32),
                                   area=0.01, angle=0.0, size=z2.copy(), pixels=100),
                     np.zeros((16, 2), np.float32))
        out = mb.status("started", "mediapipe")
        out += mb.frame(0, 0.0, 30.0, [pose], [0], [], [hand], [0], [], [face], [0], [], [blob], [0], [])
        out += mb.leave_all([0], [0], [0], [0])
        out += mb.status("stopped")
        return out

    def test_same_addresses_and_types(self):
        real = {}
        for addr, args in self.app_messages():
            real.setdefault(addr, set()).add(signature([float(a) if isinstance(a, np.floating) else int(a) if isinstance(a, np.integer) else a for a in args]))
        sim = {}
        sent = []
        scene = osc_simulator.Scene({"pose", "hands", "gesture", "face", "blobs"})
        sent.append(("/mp/status", ["started", "x"]))
        sent.append(("/mp/face/blendshape_names", list(osc_simulator.BLENDSHAPE_NAMES)))
        sent.append(("/mp/z_source", ["mediapipe"]))
        for f in range(3):
            scene.step(lambda a, args: sent.append((a, args)), f / 30.0, 30.0)
        for kind in ("pose", "hand", "face", "blob"):
            for i in scene.all_ids(kind):
                sent.append((f"/mp/{kind}/leave", [i]))
        for addr, args in sent:
            sim.setdefault(addr, set()).add(signature(args))
        self.assertEqual(set(real), set(sim))
        for addr in real:
            if addr == "/mp/blobs":
                continue           # variable length: count + ids
            self.assertEqual(real[addr], sim[addr], addr)

    def test_encoding_roundtrip(self):
        msg = osc_simulator.encode("/mp/pose/center", [3, 0.5, 0.25, -0.1])
        parsed = OscMessage(msg.dgram)
        self.assertEqual(parsed.address, "/mp/pose/center")
        self.assertEqual(parsed.params, [3, 0.5, 0.25, -0.10000000149011612])
        self.assertIsInstance(parsed.params[0], int)

    def test_enter_and_leave_events(self):
        scene = osc_simulator.Scene({"pose"})
        addresses = []
        for f in range(13 * 30):                      # person 1 leaves at t=8 s and returns at 12 s
            scene.step(lambda a, args: addresses.append(a), f / 30.0, 30.0)
        self.assertEqual(addresses.count("/mp/pose/enter"), 3)
        self.assertEqual(addresses.count("/mp/pose/leave"), 1)
        self.assertNotIn("/mp/hand/center", addresses)


# ---------------------------------------------------------------------------
class PythonExamplesTest(unittest.TestCase):
    def test_every_example_receives_the_simulator(self):
        port = free_port()
        sim = simulator(port, 4 * len(EXAMPLES) + 5)
        try:
            time.sleep(0.5)
            for name in EXAMPLES:
                with self.subTest(example=name):
                    script = os.path.join(ROOT, "python-examples", name, f"{name}.py")
                    snap = os.path.join(ROOT, "python-examples", "build", f"{name}.png")
                    os.makedirs(os.path.dirname(snap), exist_ok=True)
                    r = subprocess.run([PYTHON, script, "--port", str(port), "--headless", "--seconds", "3", "--snapshot", snap],
                                       capture_output=True, text=True, timeout=60)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn("done:", r.stdout)
                    self.assertNotIn("seen: []", r.stdout, r.stdout)
                    self.assertTrue(os.path.exists(snap))
        finally:
            sim.terminate()
            sim.wait()
            shutil.rmtree(os.path.join(ROOT, "python-examples", "build"), ignore_errors=True)

    def test_nothing_received_is_reported(self):
        port = free_port()
        script = os.path.join(ROOT, "python-examples", "blobs", "blobs.py")
        r = subprocess.run([PYTHON, script, "--port", str(port), "--headless", "--seconds", "0.5"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 2)


# ---------------------------------------------------------------------------
@unittest.skipUnless(shutil.which("javac") and shutil.which("java"), "needs a JDK")
class ProcessingExamplesTest(unittest.TestCase):
    def test_sketches_compile_and_receive(self):
        out_root = os.path.join(ROOT, "processing-examples", "build", "stub")
        for name in EXAMPLES:
            with self.subTest(example=name):
                pde = os.path.join(ROOT, "processing-examples", name, f"{name}.pde")
                r = subprocess.run(["sh", os.path.join(STUBS, "processing", "build.sh"), pde, os.path.join(out_root, name)],
                                   capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        port = free_port()
        sim = simulator(port, 4 * len(EXAMPLES) + 5)
        try:
            time.sleep(0.5)
            for name in EXAMPLES:
                with self.subTest(example=name):
                    r = subprocess.run(["java", f"-Dosc.port={port}", "-cp", os.path.join(out_root, name), "Runner", name, "3"],
                                       capture_output=True, text=True, timeout=60)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn("oscEvent failures 0", r.stdout, r.stdout + r.stderr)
                    self.assertNotIn("received 0,", r.stdout)
                    self.assertRegex(r.stdout, r"### \w+: \[\d")     # a map with ids
        finally:
            sim.terminate()
            sim.wait()
            shutil.rmtree(os.path.join(ROOT, "processing-examples", "build"), ignore_errors=True)


# ---------------------------------------------------------------------------
@unittest.skipUnless(shutil.which("g++"), "needs g++")
class OpenFrameworksExamplesTest(unittest.TestCase):
    def test_apps_compile_and_receive(self):
        out_root = os.path.join(ROOT, "openFrameworks-examples", "build", "stub")
        for name in EXAMPLES:
            with self.subTest(example=name):
                r = subprocess.run(["sh", os.path.join(STUBS, "openframeworks", "build.sh"),
                                    os.path.join(ROOT, "openFrameworks-examples", name), os.path.join(out_root, name)],
                                   capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        port = free_port()
        sim = simulator(port, 4 * len(EXAMPLES) + 5)
        try:
            time.sleep(0.5)
            env = dict(os.environ, OF_STUB_SECONDS="3", OF_STUB_PORT=str(port))
            for name in EXAMPLES:
                with self.subTest(example=name):
                    r = subprocess.run([os.path.join(out_root, name)], capture_output=True, text=True, timeout=60, env=env)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertRegex(r.stdout, r"done: [1-9]\d* messages received, [1-9]")
                    self.assertNotIn("dropped [1-9]", r.stdout)
        finally:
            sim.terminate()
            sim.wait()
            shutil.rmtree(os.path.join(ROOT, "openFrameworks-examples", "build"), ignore_errors=True)


# ---------------------------------------------------------------------------
class TouchDesignerExamplesTest(unittest.TestCase):
    def test_scripts_build_and_parse(self):
        runner = os.path.join(STUBS, "touchdesigner", "run_example.py")
        for name in EXAMPLES:
            with self.subTest(example=name):
                script = os.path.join(ROOT, "touchdesigner-examples", name, f"{name}.py")
                r = subprocess.run([PYTHON, runner, script, "2"], capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("callback failures 0", r.stdout)
                self.assertNotIn("WARNING", r.stdout)

    def test_outside_touchdesigner_the_scripts_do_nothing(self):
        for name in EXAMPLES:
            script = os.path.join(ROOT, "touchdesigner-examples", name, f"{name}.py")
            r = subprocess.run([PYTHON, script], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0)
            self.assertIn("inside TouchDesigner", r.stdout)


if __name__ == "__main__":
    unittest.main()
