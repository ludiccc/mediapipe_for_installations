"""Look & feel templates: loading, rendering the panel and the preview, switching at run time."""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc.app import App  # noqa: E402
from mp_osc.blobs import BlobDetector  # noqa: E402
from mp_osc.config import Config, parse_args  # noqa: E402
from mp_osc.themes import Theme, available_templates, load_theme, template_path, vignette  # noqa: E402
from mp_osc.tracking import Tracker  # noqa: E402
from mp_osc.ui import ControlPanel, RadioButton, Toggle  # noqa: E402
from tests.test_osc_out import fake_face, fake_hand, fake_pose  # noqa: E402

TEMPLATES = os.path.join(ROOT, "templates")
PORTRAIT = os.path.join(ROOT, "brother-977170_1280.jpg")
EXAMPLES = ["minimal", "ascii", "baroque"]


class CapturingSender:
    def __init__(self):
        self.messages, self.sent, self.paused = [], 0, False
        self.ip, self.port = "127.0.0.1", 5005

    def send(self, messages):
        self.messages.extend(messages)
        self.sent += len(messages)

    def retarget(self, ip, port):
        self.ip, self.port = ip, int(port)


def fake_tracks():
    frame = cv2.resize(cv2.imread(PORTRAIT), (640, 480))
    det = BlobDetector(mode="dark", threshold=60)
    blobs = Tracker().update(det.process(frame), 0.0).active
    poses = Tracker().update([fake_pose(0.4, 0.5)], 0.0).active
    hands = Tracker().update([fake_hand(0.7, 0.6)], 0.0).active
    faces = Tracker().update([fake_face(0.5, 0.3)], 0.0).active
    for t in poses + hands + faces + blobs:
        t.velocity[:] = (0.2, -0.1)
    return frame, poses, hands, faces, blobs, det.mask


class LoadingTest(unittest.TestCase):
    def test_available_and_paths(self):
        names = available_templates(TEMPLATES)
        self.assertEqual(names[0], "classic")
        for name in EXAMPLES:
            self.assertIn(name, names)
        self.assertIsNone(template_path("classic", TEMPLATES))
        self.assertTrue(template_path("ascii", TEMPLATES).endswith("ascii.py"))
        self.assertTrue(template_path(os.path.join(TEMPLATES, "baroque.py"), "").endswith("baroque.py"))
        with self.assertRaises(FileNotFoundError):
            template_path("rococo", TEMPLATES)
        self.assertEqual(available_templates("/no/such/dir"), ["classic"])

    def test_load_every_example(self):
        self.assertIsInstance(load_theme("classic"), Theme)
        self.assertEqual(load_theme("classic").name, "classic")
        for name in EXAMPLES:
            theme = load_theme(name, TEMPLATES)
            self.assertIsInstance(theme, Theme)
            self.assertEqual(theme.name, name)
            self.assertTrue(theme.description)
            self.assertGreater(theme.panel_width, 100)

    def test_minimal_template_file_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pink.py")
            with open(path, "w") as fh:
                fh.write("from mp_osc.themes import Theme\n"
                         "class Template(Theme):\n    bg = (200, 120, 255)\n")
            theme = load_theme(path)
            self.assertEqual(theme.bg, (200, 120, 255))
            self.assertEqual(theme.name, "classic")      # name not overridden: inherits... unless empty
            self.assertEqual(load_theme("pink", d).bg, (200, 120, 255))
            with open(os.path.join(d, "inst.py"), "w") as fh:
                fh.write("from mp_osc.themes import Theme\nTHEME = Theme()\nTHEME.name = 'inst'\n")
            self.assertEqual(load_theme("inst", d).name, "inst")
            with open(os.path.join(d, "empty.py"), "w") as fh:
                fh.write("x = 1\n")
            with self.assertRaises(ImportError):
                load_theme("empty", d)
            with open(os.path.join(d, "broken.py"), "w") as fh:
                fh.write("def (:\n")
            with self.assertRaises(SyntaxError):
                load_theme("broken", d)
            self.assertEqual(available_templates(d), ["classic", "broken", "empty", "inst", "pink"])

    def test_cli_option(self):
        cfg = parse_args(["--template", "ascii", "--no-preview"])
        self.assertEqual(cfg.template, "ascii")
        self.assertEqual(cfg.templates_dir, TEMPLATES)


class RenderingTest(unittest.TestCase):
    """Every template draws every widget and every kind of detection without error."""

    @classmethod
    def setUpClass(cls):
        cls.app = App(Config(source=PORTRAIT, preview=False, quiet=True, pose=True, blobs=True, blob_mode="dark"),
                      sender=CapturingSender())
        cls.app.file_source = PORTRAIT

    @classmethod
    def tearDownClass(cls):
        cls.app.close()

    def render_all(self, theme):
        app = self.app
        app.theme = theme
        panel = ControlPanel(app, theme)
        frame, poses, hands, faces, blobs, mask = fake_tracks()
        img = theme.draw_preview(frame.copy(), poses, hands, faces, blobs, True, True, mask=mask, status="status")
        self.assertEqual(img.shape, frame.shape)
        self.assertEqual(img.dtype, np.uint8)
        theme.draw_banner(img, "learning background 50%")
        # ids off / landmarks off paths
        theme.draw_preview(frame.copy(), poses, hands, faces, blobs, False, False)
        # focused field with cursor and an error, pressed button
        panel.find("Port").focus()
        panel.find("Port").error = "port must be a number"
        panel.find("Quit  (q)")._pressed_at = 1e12
        out = panel.render(img, app.status_lines())
        self.assertEqual(out.shape[1], frame.shape[1] + theme.panel_width)
        self.assertGreaterEqual(out.shape[0], max(frame.shape[0], panel.height))
        return panel, out

    def test_classic(self):
        panel, out = self.render_all(Theme())
        self.assertGreater(int(out[:, 640:].mean()), 30)

    def test_examples_render(self):
        for name in EXAMPLES:
            theme = load_theme(name, TEMPLATES)
            panel, out = self.render_all(theme)
            panel_img = out[:, 640:]
            self.assertGreater(int(panel_img.std()), 10, name)            # something is drawn
            self.assertIsInstance(panel.find("Body"), Toggle)
            self.assertIsInstance(panel.find("Cameras"), RadioButton)

    def test_ascii_preview_is_made_of_cells(self):
        theme = load_theme("ascii", TEMPLATES)
        frame, poses, hands, faces, blobs, mask = fake_tracks()
        img = theme.draw_preview(frame.copy(), poses, hands, faces, blobs, True, True)
        self.assertEqual(img.shape, frame.shape)
        # the picture is green-ish text on black: red channel is small, there is black between glyphs
        self.assertLess(int(img[..., 2].mean()), int(img[..., 1].mean()))
        self.assertGreater(float((img.sum(axis=2) == 0).mean()), 0.3)
        # the panel text is monospace: width depends only on the character count
        self.assertEqual(theme.text_width("iiii"), theme.text_width("WWWW"))
        self.assertEqual(theme.fit_text("abcdefghij", 5 * theme.cw), "abcd~")

    def test_baroque_preview_is_sepia(self):
        theme = load_theme("baroque", TEMPLATES)
        frame, poses, hands, faces, blobs, mask = fake_tracks()
        img = theme.draw_preview(frame.copy(), [], [], [], [], True, True)
        b, g, r = [float(img[100:380, 100:540, i].mean()) for i in range(3)]
        self.assertGreater(r, g)
        self.assertGreater(g, b)                                           # warm tone
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        vignette(white)
        self.assertLess(int(white[0, 0].mean()), 160)                    # corners darkened
        self.assertEqual(int(white[240, 320].mean()), 255)               # center untouched

    def test_minimal_panel_is_light(self):
        theme = load_theme("minimal", TEMPLATES)
        panel, out = self.render_all(theme)
        self.assertGreater(int(out[:, 640:].mean()), 200)

    def test_panel_on_the_left(self):
        class Left(Theme):
            panel_side = "left"
        theme = Left()
        panel = ControlPanel(self.app, theme)
        out = panel.render(np.zeros((480, 640, 3), dtype=np.uint8), [])
        self.assertEqual(panel.origin, (0, 0))
        self.assertGreater(int(out[:, :theme.panel_width].mean()), 30)
        self.assertEqual(int(out[:480, theme.panel_width:].mean()), 0)
        # clicking uses the panel origin: Body sits at x < panel_width
        panel.click("Body")
        self.assertFalse(self.app.cfg.pose)
        panel.click("Body")
        self.assertTrue(self.app.cfg.pose)


class SwitchingTest(unittest.TestCase):
    def setUp(self):
        self.app = App(Config(source=PORTRAIT, preview=False, quiet=True, pose=True), sender=CapturingSender())
        self.app.panel = ControlPanel(self.app)
        self.app.panel.frame_width = 640

    def tearDown(self):
        self.app.close()

    def test_cycle_with_button_key_and_reload(self):
        app, panel = self.app, self.app.panel
        self.assertEqual(app.theme.name, "classic")
        panel.click("Look  (t)")
        self.assertEqual(app.theme.name, "ascii")
        self.assertIs(panel.theme, app.theme)
        self.assertEqual(panel.width, app.theme.panel_width)
        self.assertIsInstance(panel.find("Blobs"), Toggle)                # layout rebuilt
        app.handle_key(ord("t"))
        self.assertEqual(app.theme.name, "baroque")
        app.handle_key(ord("t"))
        self.assertEqual(app.theme.name, "minimal")
        app.handle_key(ord("t"))
        self.assertEqual(app.theme.name, "classic")
        first = app.theme
        app.handle_key(ord("r"))
        self.assertIsNot(app.theme, first)                                 # reloaded: a fresh instance
        self.assertEqual(app.cfg.template, "classic")

    def test_broken_template_keeps_the_current_look(self):
        app = self.app
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.py")
            with open(path, "w") as fh:
                fh.write("raise RuntimeError('boom')\n")
            err = app.set_template(path)
        self.assertIn("boom", err)
        self.assertEqual(app.theme.name, "classic")
        self.assertIsNone(app.set_template("minimal"))
        self.assertEqual(app.theme.name, "minimal")

    def test_focus_survives_nothing_but_settings_do(self):
        app, panel = self.app, self.app.panel
        panel.click("Mirror")
        app.set_template("baroque")
        self.assertTrue(app.mirror)
        self.assertTrue(panel.find("Mirror").get())
        app.step()                                                         # draws with the new theme

    def test_unknown_template_on_the_command_line(self):
        with self.assertRaises(RuntimeError) as ctx:
            App(Config(source=PORTRAIT, preview=False, quiet=True, template="rococo"), sender=CapturingSender())
        self.assertIn("rococo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
