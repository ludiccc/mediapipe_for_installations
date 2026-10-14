"""Tests for the in-window control panel and runtime reconfiguration.

The panel is driven through the same entry points OpenCV would use
(``on_mouse`` with pixel coordinates, ``on_key`` with key codes); no display
is needed because rendering happens into a numpy image.
"""

import os
import sys
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
from mp_osc.config import Config, parse_args  # noqa: E402
from mp_osc.ui import ControlPanel, ListBox, PANEL_WIDTH, RadioButton, Stepper, TextField, Toggle  # noqa: E402

PORTRAIT = os.path.join(ROOT, "brother-977170_1280.jpg")


class CapturingSender:
    def __init__(self):
        self.messages = []
        self.sent = 0
        self.paused = False
        self.ip, self.port = "127.0.0.1", 5005

    def send(self, messages):
        if self.paused:
            return
        self.messages.extend(messages)
        self.sent += len(messages)

    def retarget(self, ip, port):
        self.ip, self.port = ip, int(port)

    def by_address(self, address):
        return [args for a, args in self.messages if a == address]

    def clear(self):
        self.messages.clear()


def make_app(**kwargs):
    cfg = Config(source=PORTRAIT, preview=False, quiet=True, **kwargs)
    sender = CapturingSender()
    app = App(cfg, sender=sender)
    app.panel = ControlPanel(app)
    app.panel.frame_width = 640   # as if a 640 px wide frame had been rendered
    return app, sender


class PanelWidgetsTest(unittest.TestCase):
    def setUp(self):
        self.app, self.sender = make_app(pose=True)
        self.panel = self.app.panel

    def tearDown(self):
        self.app.close()

    def test_render_places_panel_right_of_the_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self.panel.render(frame, self.app.status_lines())
        self.assertEqual(out.shape, (max(480, self.panel.height), 640 + PANEL_WIDTH, 3))
        self.assertLessEqual(self.panel.height, 720)           # fits next to a 720p preview
        self.assertGreater(int(out[:, 640:].mean()), 30)        # panel is drawn (not black)
        self.assertEqual(self.panel.frame_width, 640)
        self.assertEqual(self.panel.origin, (640, 0))

    def test_render_taller_panel_than_frame(self):
        out = self.panel.render(np.zeros((240, 320, 3), dtype=np.uint8), [])
        self.assertEqual(out.shape[1], 320 + PANEL_WIDTH)
        self.assertGreaterEqual(out.shape[0], self.panel.height)

    def test_toggle_body_switches_pose_off_and_on(self):
        self.assertTrue(self.app.analyzer.pose_enabled)
        self.panel.click("Body")
        self.assertFalse(self.app.cfg.pose)
        self.assertFalse(self.app.analyzer.pose_enabled)
        self.panel.click("Body")
        self.assertTrue(self.app.analyzer.pose_enabled)

    def test_gestures_imply_hands_and_hands_off_drops_gestures(self):
        self.panel.click("Gestures")
        self.assertTrue(self.app.cfg.hands and self.app.cfg.gesture)
        self.assertTrue(self.app.analyzer.gesture_enabled)
        self.panel.click("Hands")
        self.assertFalse(self.app.cfg.hands)
        self.assertFalse(self.app.cfg.gesture)
        self.assertFalse(self.app.analyzer.hands_enabled)

    def test_face_toggle(self):
        self.assertFalse(self.app.analyzer.face_enabled)
        self.panel.click("Face")
        self.assertTrue(self.app.analyzer.face_enabled)

    def test_image_and_output_toggles(self):
        self.panel.click("Mirror")
        self.assertTrue(self.app.mirror)
        self.panel.click("Draw")
        self.assertFalse(self.app.draw_landmarks)
        self.panel.click("Landmarks")
        self.assertFalse(self.app.cfg.send_landmarks)
        self.panel.click("Pause")
        self.assertTrue(self.sender.paused)
        self.panel.click("kinect")
        self.assertEqual(self.app.cfg.formats, ("mp", "kinect"))
        self.panel.click("mp")
        self.assertEqual(self.app.cfg.formats, ("kinect",))

    def test_blobs_toggle_and_learn_button(self):
        self.assertFalse(self.app.analyzer.blobs_enabled)
        self.panel.click("Blobs")
        self.assertTrue(self.app.cfg.blobs)
        self.assertTrue(self.app.analyzer.blobs_enabled)
        det = self.app.analyzer.blob_detector
        self.app.step()
        self.panel.click("Learn bg  (b)")
        self.assertTrue(det.learning)
        self.panel.click("Blobs")
        self.assertFalse(self.app.analyzer.blobs_enabled)

    def test_steppers(self):
        sens = self.panel.find("Blob sensitivity")
        self.assertIsInstance(sens, Stepper)
        self.panel.click("Blob sensitivity", "right")            # [+]
        self.assertEqual(self.app.cfg.blob_threshold, 30)
        self.panel.click("Blob sensitivity", "left")             # [-]
        self.panel.click("Blob sensitivity", "left")
        self.assertEqual(self.app.cfg.blob_threshold, 20)
        self.panel.click("Blob sensitivity")                     # the middle does nothing
        self.assertEqual(self.app.cfg.blob_threshold, 20)
        self.assertEqual(sens.get_value(), "20")
        area = self.panel.find("Blob min area %")
        self.panel.click("Blob min area %", "right")
        self.assertAlmostEqual(self.app.cfg.blob_min_area, 0.006)
        self.assertEqual(area.get_value(), "0.60")
        # mouse wheel over a stepper
        self.panel.on_mouse(cv2.EVENT_MOUSEWHEEL, 640 + sens.x + sens.w // 2, sens.y + 2, flags=120 << 16)
        self.assertEqual(self.app.cfg.blob_threshold, 25)                     # wheel up: +
        self.panel.on_mouse(cv2.EVENT_MOUSEWHEEL, 640 + sens.x + sens.w // 2, sens.y + 2, flags=(-120 << 16) & 0xFFFFFFFF)
        self.assertEqual(self.app.cfg.blob_threshold, 20)                     # wheel down: -

    def test_radio_buttons_and_camera_list_widgets(self):
        radios = [w for w in self.panel.widgets if isinstance(w, RadioButton)]
        self.assertEqual([r.label for r in radios], ["Cameras", "Pause", "File"])
        self.assertEqual([r.selected for r in radios], [False, False, True])
        self.panel.click("Pause")
        self.assertEqual([r.selected for r in radios], [False, True, False])
        self.assertTrue(self.sender.paused)
        self.panel.click("File")
        self.assertFalse(self.sender.paused)
        self.assertEqual(sum(r.selected for r in radios), 1)
        lst = self.panel.camera_list
        self.assertIsInstance(lst, ListBox)
        self.assertEqual(lst.get_items(), [])                     # no scan ran (headless)
        self.assertEqual(self.app.selected_camera_rows(), [])
        lst.click(lst.x + 5, lst.y + 5)                           # empty list: nothing happens
        with self.assertRaises(KeyError):
            lst.select_label("nope")

    def test_listbox_scrolling(self):
        items = [f"{i}  cam {i}" for i in range(6)]
        chosen = []
        lb = ListBox(0, 0, 200, 10, 3, "L", lambda: items, lambda: 0, chosen.append)
        self.assertEqual(lb.get_selected(), [0])                  # an int is accepted for one row
        self.assertFalse(lb.can_move)
        lb.click(5, 15)                                           # row 1
        self.assertEqual(chosen, [1])
        self.assertFalse(lb.scroll(-1))                           # already at the top
        self.assertTrue(lb.scroll(1))
        self.assertEqual(lb.offset, 1)
        lb.click(5, 25)                                           # row 2 -> item 3
        self.assertEqual(chosen[-1], 3)
        lb.click(195, 2)                                          # right strip, upper half: scroll up
        self.assertEqual(lb.offset, 0)
        lb.click(195, 28)                                         # lower half: scroll down
        self.assertEqual(lb.offset, 1)
        for _ in range(10):
            lb.scroll(1)
        self.assertEqual(lb.offset, 3)                            # clamped: 6 items, 3 visible
        img = np.zeros((40, 200, 3), dtype=np.uint8)
        lb.draw(img, self.panel.theme)
        short = ListBox(0, 0, 200, 10, 3, "S", lambda: items[:2], lambda: 1, chosen.append)
        self.assertFalse(short.scroll(1))
        short.draw(img, self.panel.theme)

    def test_listbox_multi_selection_and_move_buttons(self):
        items = ["a", "b", "c"]
        selected = [0, 2]
        moves = []
        lb = ListBox(0, 0, 200, 10, 3, "L", lambda: items, lambda: selected, lambda i: None, on_move=moves.append)
        lb.on_move = lambda i, d: moves.append((i, d))
        self.assertTrue(lb.can_move)
        self.assertEqual(lb.get_selected(), [0, 2])
        lb.click(200 - lb.MOVE_W + 3, 25)                         # "<" of row 2 (checked)
        lb.click(200 - 3, 25)                                     # ">" of row 2
        lb.click(200 - 3, 15)                                     # row 1 is not checked: a plain select
        self.assertEqual(moves, [(2, -1), (2, 1)])
        img = np.zeros((40, 200, 3), dtype=np.uint8)
        lb.draw(img, self.panel.theme)
        self.assertGreater(int(img[20:30, 170:200].mean()), 20)   # the buttons are drawn

    def test_click_outside_widgets_does_nothing(self):
        before = (self.app.cfg.pose, self.app.mirror, self.app.cfg.formats)
        self.panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 640 + PANEL_WIDTH - 2, 2)   # panel corner
        self.panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100)                   # on the image
        self.panel.on_mouse(cv2.EVENT_MOUSEMOVE, 640 + 50, 50)                 # hovering over Body
        self.assertEqual(before, (self.app.cfg.pose, self.app.mirror, self.app.cfg.formats))

    def test_quit_button(self):
        self.panel.click("Quit  (q)")
        self.assertTrue(self.app._quit)

    def test_keyboard_shortcuts_without_focus(self):
        self.assertEqual(self.panel.on_key(ord("q")), "quit")
        self.assertEqual(self.panel.on_key(27), "quit")
        self.assertIsNone(self.panel.on_key(ord("m")))     # left to the app
        self.assertIsNone(self.panel.on_key(-1))
        self.assertIsNone(self.panel.on_key(255))
        self.assertTrue(self.app.handle_key(ord("m")))
        self.assertTrue(self.app.mirror)
        self.assertFalse(self.app.handle_key(ord("q")))


class TextFieldTest(unittest.TestCase):
    def setUp(self):
        self.app, self.sender = make_app(pose=True)
        self.panel = self.app.panel
        self.ip = self.panel.find("IP / host")
        self.port = self.panel.find("Port")

    def tearDown(self):
        self.app.close()

    def test_edit_ip_and_submit(self):
        self.panel.click("IP / host")
        self.assertTrue(self.ip.focused)
        self.assertEqual(self.ip.text, "127.0.0.1")
        for _ in range(9):
            self.panel.on_key(8)                            # backspace
        self.panel.type_text("localhost")
        self.assertEqual(self.panel.on_key(13), "handled")  # enter
        self.assertFalse(self.ip.focused)
        self.assertEqual(self.ip.error, "")
        self.assertEqual(self.sender.ip, "localhost")
        self.assertEqual(self.app.cfg.ip, "localhost")

    def test_edit_port(self):
        self.panel.click("Port")
        for _ in range(4):
            self.panel.on_key(127)                          # backspace (mac/linux variant)
        self.panel.type_text("9000")
        self.panel.on_key(10)                               # enter (linux variant)
        self.assertEqual(self.sender.port, 9000)
        self.assertEqual(self.app.cfg.port, 9000)
        # receivers were told: old target got "stopped", new one "started"
        statuses = [a[0] for a in self.sender.by_address("/mp/status")]
        self.assertEqual(statuses[-2:], ["stopped", "started"])

    def test_invalid_values_show_error_and_keep_focus(self):
        self.panel.click("Port")
        self.port.text = "99999"
        self.panel.on_key(13)
        self.assertTrue(self.port.focused)
        self.assertIn("65535", self.port.error)
        self.assertEqual(self.sender.port, 5005)

        self.port.text = "abc"
        self.panel.on_key(13)
        self.assertIn("number", self.port.error)

        self.panel.click("IP / host")
        self.ip.text = "no.such.host.invalid"
        self.panel.on_key(13)
        self.assertIn("unknown host", self.ip.error)
        self.assertEqual(self.sender.ip, "127.0.0.1")

    def test_escape_cancels_edit(self):
        self.panel.click("IP / host")
        self.panel.type_text("garbage")
        self.panel.on_key(27)
        self.assertFalse(self.ip.focused)
        self.assertEqual(self.sender.ip, "127.0.0.1")

    def test_typing_q_in_a_field_does_not_quit(self):
        self.panel.click("IP / host")
        self.assertEqual(self.panel.on_key(ord("q")), "handled")
        self.assertTrue(self.app.handle_key(ord("q")))       # app keeps running
        self.assertTrue(self.app.handle_key(ord("m")))
        self.assertFalse(self.app.mirror)                     # shortcut not triggered
        self.assertTrue(self.ip.text.endswith("qm"))

    def test_clicking_elsewhere_blurs_field(self):
        self.panel.click("IP / host")
        self.panel.click("Mirror")
        self.assertFalse(self.ip.focused)
        self.assertTrue(self.app.mirror)
        self.panel.click("Port")
        self.panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)   # click on the camera image
        self.assertFalse(self.port.focused)

    def test_max_length(self):
        self.panel.click("Port")
        self.port.text = ""
        self.panel.type_text("1234567")
        self.assertEqual(self.port.text, "12345")

    def test_find_unknown_label(self):
        with self.assertRaises(KeyError):
            self.panel.find("nope")
        self.assertIsInstance(self.panel.find("Body"), Toggle)
        self.assertIsInstance(self.ip, TextField)


class RuntimeReconfigurationTest(unittest.TestCase):
    """Switching detectors while frames are flowing."""

    def test_enable_face_then_disable_pose_while_running(self):
        app, sender = make_app(pose=True)
        try:
            app.step()
            self.assertEqual(len(sender.by_address("/mp/pose/center")), 1)
            self.assertEqual(sender.by_address("/mp/face/center"), [])

            app.panel.click("Face")
            app.step()
            self.assertEqual(len(sender.by_address("/mp/face/center")), 2)   # two faces in the photo

            sender.clear()
            app.panel.click("Body")
            self.assertEqual(sender.by_address("/mp/pose/leave"), [[0]])  # receivers are told
            app.step()
            self.assertEqual(sender.by_address("/mp/pose/center"), [])
            self.assertEqual(sender.by_address("/mp/frame")[0][3:], [0, 0, 2, 0])

            app.panel.click("Body")                                       # and back on: fresh id
            app.step()
            self.assertEqual(sender.by_address("/mp/pose/enter"), [[1]])
        finally:
            app.close()

    def test_quit_request_stops_the_loop(self):
        app, _ = make_app(pose=True)
        try:
            self.assertTrue(app.step())
            app.request_quit()
            self.assertFalse(app.step())
        finally:
            app.close()

    def test_pause_stops_sending_but_not_analysis(self):
        app, sender = make_app(pose=True)
        try:
            app.panel.click("Pause")
            app.step()
            self.assertEqual(sender.by_address("/mp/frame"), [])
            self.assertEqual(app.counts[0], 1)                             # still analyzing
        finally:
            app.close()

    def test_no_panel_option(self):
        cfg = parse_args(["--no-panel", "--no-preview"])
        self.assertFalse(cfg.panel)
        app = App(Config(source=PORTRAIT, preview=False, quiet=True, panel=False), sender=CapturingSender())
        self.assertIsNone(app.panel)
        app.close()


class RetargetOverUdpTest(unittest.TestCase):
    def start_server(self):
        received = []
        d = Dispatcher()
        d.set_default_handler(lambda addr, *args: received.append((addr, list(args))))
        server = ThreadingOSCUDPServer(("127.0.0.1", 0), d)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1], received

    def test_messages_follow_the_new_port(self):
        port_a, got_a = self.start_server()
        port_b, got_b = self.start_server()
        app = App(Config(source=PORTRAIT, preview=False, quiet=True, pose=True, port=port_a))
        app.panel = ControlPanel(app)
        app.panel.frame_width = 640
        try:
            app.step()
            self.assertIsNone(app.set_osc(port=port_b))
            app.step()
        finally:
            app.close()
        time.sleep(0.3)
        self.assertIn("/mp/frame", [a for a, _ in got_a])
        self.assertEqual([a for a, _ in got_a if a == "/mp/status"][-1:], ["/mp/status"])
        self.assertEqual(got_a[-1][1][0], "stopped")
        addrs_b = [a for a, _ in got_b]
        self.assertEqual(addrs_b[0], "/mp/status")
        self.assertIn("/mp/frame", addrs_b)
        self.assertEqual(got_b[-1][1][0], "stopped")     # final shutdown goes to the new target


if __name__ == "__main__":
    unittest.main()
