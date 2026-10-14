"""GUI integration test: the real OpenCV window driven by real X events.

Needs Xvfb and xdotool (Linux); skipped elsewhere. Everything the headless
tests cannot see is covered here: window creation, the mouse callback with
window coordinates, key delivery through the backend, and the analysis thread
running concurrently with the window loop.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import unittest

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc.themes import Theme, load_theme  # noqa: E402
from mp_osc.ui import PANEL_WIDTH, ControlPanel  # noqa: E402

PORTRAIT = os.path.join(ROOT, "brother-977170_1280.jpg")
DISPLAY = ":97"


class _StubApp:
    """Only what ControlPanel reads while building its layout."""
    class cfg:
        pose = True; hands = False; gesture = False; face = False; blobs = False
        formats = ("mp",); send_landmarks = True; blob_threshold = 25; blob_min_area = 0.004
        z_from_kinect = False; no_webcams = False; no_kinects = False
    class sender:
        ip = "127.0.0.1"; port = 0; paused = False
    theme = Theme()
    mirror = False
    draw_landmarks = True
    mode = "file"
    file_source = "x.jpg"
    def set_detectors(self, **k): pass
    def set_osc(self, **k): pass
    def set_format(self, *a): pass
    def set_mode(self, m): pass
    def request_quit(self): pass
    def learn_background(self): pass
    def step_blob_threshold(self, d): pass
    def step_blob_min_area(self, d): pass
    depth_available = False
    def camera_labels(self): return []
    def selected_camera_rows(self): return []
    def toggle_camera_row(self, r): pass
    def move_camera_row(self, r, d): pass
    def set_z_from_kinect(self, v): pass
    editor = None
    def toggle_mapping_editor_row(self, r): pass
    def gear_state_row(self, r): return ""
    def save_config(self, path=None): pass
    def load_config(self, path=None): pass
    def config_status(self): return ("", False)
    def scan_cameras(self): pass
    def source_note(self): return ("", False)
    def next_template(self): pass


@unittest.skipUnless(shutil.which("Xvfb") and shutil.which("xdotool"), "needs Xvfb and xdotool")
class RealWindowTest(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ, DISPLAY=DISPLAY)
        self.xvfb = subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", "1920x1080x24"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self._stop, self.xvfb)
        time.sleep(1.0)

    @staticmethod
    def _stop(proc):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def listener(self):
        got = []
        d = Dispatcher()
        d.set_default_handler(lambda a, *args: got.append((time.monotonic(), a, args)))
        server = ThreadingOSCUDPServer(("127.0.0.1", 0), d)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1], got

    def xdo(self, *args):
        return subprocess.run(["xdotool", *args], env=self.env, capture_output=True, text=True,
                              timeout=15).stdout.strip()

    def test_clicks_and_typing_in_the_real_window(self):
        port_a, got_a = self.listener()
        port_b, got_b = self.listener()
        app = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "mediapipe_osc.py"), "--source", PORTRAIT,
             "--pose", "--port", str(port_a)],
            env=self.env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(self._stop, app)

        wid = self.xdo("search", "--sync", "--name", "MediaPipe")
        self.assertTrue(wid, "window did not appear")

        def geometry():
            return dict(line.split("=") for line in self.xdo("getwindowgeometry", "--shell", wid).splitlines())

        # The window first shows a 640 px placeholder until the first analyzed
        # frame arrives (model loading); wait for the real size.
        deadline = time.monotonic() + 60.0
        geo = geometry()
        while int(geo["WIDTH"]) < 1280 and time.monotonic() < deadline:
            time.sleep(0.5)
            geo = geometry()
        time.sleep(1.0)
        wx, wy, ww = int(geo["X"]), int(geo["Y"]), int(geo["WIDTH"])
        self.assertEqual(ww, 1280 + PANEL_WIDTH)       # image + panel, no toolbar
        frame_w = ww - PANEL_WIDTH
        layout = ControlPanel(_StubApp())

        def click(label):
            w = layout.find(label)
            self.xdo("mousemove", "--sync", str(wx + frame_w + w.x + w.w // 2), str(wy + w.y + w.h // 2),
                     "click", "1")
            time.sleep(0.8)

        t_face = time.monotonic()
        click("Face")
        click("Port")
        for _ in range(5):
            self.xdo("key", "--delay", "150", "BackSpace")
        self.xdo("type", "--delay", "150", str(port_b))      # a human pace
        self.xdo("key", "Return")
        time.sleep(1.5)

        def quiet(got, timeout=10.0):
            """True once no message arrived for a full second (within ``timeout``)."""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                n = len(got)
                time.sleep(1.0)
                if len(got) == n:
                    return True
            return False

        def resumes(got, timeout=10.0):
            n = len(got)
            deadline = time.monotonic() + timeout
            while len(got) == n and time.monotonic() < deadline:
                time.sleep(0.1)
            return len(got) > n

        # Source radio group: Pause stops the stream, File resumes it.
        click("Pause")
        self.assertTrue(quiet(got_b), "messages kept coming while paused")
        click("File")
        self.assertTrue(resumes(got_b), "stream did not resume")

        # 't' switches the look; the window follows the new panel width.
        self.xdo("key", "t")
        ascii_width = load_theme("ascii", os.path.join(ROOT, "templates")).panel_width
        deadline = time.monotonic() + 20.0
        geo = geometry()
        while int(geo["WIDTH"]) != 1280 + ascii_width and time.monotonic() < deadline:
            time.sleep(0.5)
            geo = geometry()
        self.assertEqual(int(geo["WIDTH"]), 1280 + ascii_width)
        time.sleep(1.0)
        # and the panel still takes clicks at its new geometry: Pause again
        layout = ControlPanel(_StubApp(), load_theme("ascii", os.path.join(ROOT, "templates")))
        click("Pause")
        self.assertTrue(quiet(got_b), "messages kept coming while paused (after the template switch)")

        self.xdo("key", "q")
        out, _ = app.communicate(timeout=30)
        self.assertEqual(app.returncode, 0, out[-1500:])
        time.sleep(0.3)

        faces_before = [g for g in got_a if g[1] == "/mp/face/center" and g[0] < t_face]
        faces_after = [g for g in got_a + got_b if g[1] == "/mp/face/center" and g[0] > t_face + 0.5]
        self.assertEqual(faces_before, [])
        self.assertGreater(len(faces_after), 3)                 # the Face click switched the detector on

        # old receiver was told to stop, new one gets started + frames + the final stop
        self.assertEqual(got_a[-1][1:], ("/mp/status", ("stopped", got_a[-1][2][1])))
        addrs_b = [g[1] for g in got_b]
        self.assertEqual(addrs_b[0], "/mp/status")
        self.assertGreater(addrs_b.count("/mp/frame"), 3)
        self.assertEqual(got_b[-1][2][0], "stopped")
        self.assertIn("/mp/face/center", addrs_b)

    def test_closing_the_window_quits_the_app(self):
        """The [x] of the window (WM_DELETE_WINDOW) must end the app, not re-open the window."""
        try:
            from Xlib import X, display, protocol  # python-xlib, only needed by this test
        except ImportError:
            self.skipTest("needs python-xlib to send WM_DELETE_WINDOW")
        port, got = self.listener()
        app = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "mediapipe_osc.py"), "--source", PORTRAIT,
             "--pose", "--port", str(port), "--width", "640", "--height", "480"],
            env=self.env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(self._stop, app)
        wid = self.xdo("search", "--sync", "--name", "MediaPipe")
        self.assertTrue(wid, "window did not appear")
        deadline = time.monotonic() + 30.0
        while not any(g[1] == "/mp/frame" for g in got) and time.monotonic() < deadline:
            time.sleep(0.2)
        time.sleep(1.0)

        d = display.Display(DISPLAY)
        win = d.create_resource_object("window", int(wid))
        wm_protocols, wm_delete = d.intern_atom("WM_PROTOCOLS"), d.intern_atom("WM_DELETE_WINDOW")
        win.send_event(protocol.event.ClientMessage(window=win, client_type=wm_protocols,
                                                    data=(32, [wm_delete, X.CurrentTime, 0, 0, 0])), event_mask=0)
        d.flush()

        out, _ = app.communicate(timeout=30)          # the app must exit by itself
        self.assertEqual(app.returncode, 0, out[-1500:])
        self.assertIn("window closed", out)
        self.assertEqual(self.xdo("search", "--name", "MediaPipe"), "")   # and not re-open it
        time.sleep(0.3)
        self.assertEqual(got[-1][1:3], ("/mp/status", ("stopped", got[-1][2][1])))


if __name__ == "__main__":
    unittest.main()
