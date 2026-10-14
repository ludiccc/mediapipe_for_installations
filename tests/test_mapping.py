"""Crop & mapping per camera (input / output quads), the editor, and configuration files."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc import app as app_module  # noqa: E402
from mp_osc import dialogs  # noqa: E402
from mp_osc.app import App  # noqa: E402
from mp_osc.cameras import CameraInfo, CameraScanner  # noqa: E402
from mp_osc.config import Config, parse_args  # noqa: E402
from mp_osc.mapping import (  # noqa: E402
    ARROWS, KEY_RIGHT, UNIT_QUAD, CameraMapping, MappingEditor, apply_mappings,
)
from mp_osc.ui import ControlPanel  # noqa: E402
from tests.test_cameras import CAMS, CapturingSender, FakeCamera, FakeKinect, fake_open_source  # noqa: E402

PORTRAIT = os.path.join(ROOT, "test_portrait_synthetic.png")

# the example file of tpf-tools/mapping/mapping_in_mapping_out, verbatim
TPF_JSON = {
    "input": [{"x": 0.15, "y": 0.15}, {"x": 0.85, "y": 0.15}, {"x": 0.85, "y": 0.85}, {"x": 0.15, "y": 0.85}],
    "output": [{"x": 0.08, "y": 0.08}, {"x": 0.92, "y": 0.08}, {"x": 0.92, "y": 0.92}, {"x": 0.08, "y": 0.92}],
}


def test_frame(w=200, h=100):
    """Grey frame with a red square covering the middle quarter (x 25..75 %, y 25..75 %)."""
    f = np.full((h, w, 3), 80, np.uint8)
    f[h // 4:3 * h // 4, w // 4:3 * w // 4] = (0, 0, 255)
    return f


class CameraMappingTest(unittest.TestCase):
    def test_identity_passes_the_frame_through(self):
        m = CameraMapping()
        self.assertTrue(m.identity)
        f = test_frame()
        out, d = m.apply(f, None)
        self.assertIs(out, f)
        self.assertIsNone(d)

    def test_crop_stretches_the_input_quad_to_the_whole_frame(self):
        m = CameraMapping(input=np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], np.float32))
        self.assertFalse(m.identity)
        out, d = m.apply(test_frame(), np.full((100, 200), 2.0, np.float32))
        self.assertEqual(out.shape, (100, 200, 3))
        self.assertTrue(np.all(out[5:95, 5:195] == (0, 0, 255)))      # the red square now fills the picture
        self.assertEqual(d.shape, (100, 200))
        self.assertEqual(float(d[50, 100]), 2.0)

    def test_output_quad_places_the_picture_and_blackens_the_rest(self):
        m = CameraMapping(output=np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]], np.float32))
        out, _ = m.apply(test_frame(), None)
        self.assertTrue(np.all(out[:, :95] == 0))                      # left half: nothing
        self.assertEqual(tuple(out[50, 150]), (0, 0, 255))             # the centre of the picture, now at 3/4
        self.assertEqual(tuple(out[5, 105]), (80, 80, 80))

    def test_degenerate_quad_leaves_the_picture_alone(self):
        m = CameraMapping(input=np.array([[0, 0], [1, 0], [0, 0], [0, 1]], np.float32))
        f = test_frame()
        out, _ = m.apply(f, None)
        self.assertTrue(np.array_equal(out, f))

    def test_json_round_trip_and_tpf_file(self):
        m = CameraMapping.from_dict(TPF_JSON)
        self.assertAlmostEqual(float(m.input[0, 0]), 0.15)
        self.assertAlmostEqual(float(m.output[2, 1]), 0.92)
        again = CameraMapping.from_dict(json.loads(json.dumps(m.to_dict())))
        np.testing.assert_allclose(again.input, m.input)
        np.testing.assert_allclose(again.output, m.output)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            m.save(path)
            loaded = CameraMapping.load(path)
        np.testing.assert_allclose(loaded.output, m.output)
        self.assertTrue(CameraMapping.from_dict({}).identity)            # missing keys = identity
        self.assertTrue(CameraMapping.from_dict({"input": [[0, 0], [1, 0], [1, 1], [0, 1]]}).identity)  # bare pairs too
        with self.assertRaises(ValueError):
            CameraMapping.from_dict({"input": [{"x": 0, "y": 0}]})
        m.reset("input")
        self.assertTrue(np.array_equal(m.input, UNIT_QUAD))
        self.assertFalse(m.identity)
        m.reset()
        self.assertTrue(m.identity)

    def test_apply_mappings_per_slot(self):
        frame = np.hstack([test_frame(), test_frame()])                 # two 200 px slots
        depth = np.hstack([np.full((100, 200), 1.0, np.float32), np.full((100, 200), 2.0, np.float32)])
        crop = CameraMapping(input=np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], np.float32))
        out, d = apply_mappings(frame, depth, [("a", 0, 200), ("b", 200, 400)], {"b": crop, "zzz": crop})
        self.assertEqual(tuple(out[5, 5]), (80, 80, 80))                # slot a untouched
        self.assertEqual(tuple(out[5, 205]), (0, 0, 255))               # slot b cropped
        self.assertEqual(float(d[50, 300]), 2.0)


class MappingEditorTest(unittest.TestCase):
    def test_press_drag_release_and_keys(self):
        m = CameraMapping()
        ed = MappingEditor("0", m, "input")
        self.assertIs(ed.quad, m.input)
        self.assertEqual(ed.corner_at(0.5, 0.5, 640, 480), -1)
        self.assertEqual(ed.corner_at(0.99, 0.01, 640, 480), 1)          # near the top-right handle
        self.assertFalse(ed.press(0.5, 0.5, 640, 480))
        self.assertTrue(ed.press(0.99, 0.01, 640, 480))
        self.assertEqual(ed.selected, 1)
        ed.drag(0.7, 0.2)
        ed.release()
        ed.drag(0.1, 0.1)                                                 # not dragging any more
        np.testing.assert_allclose(m.input[1], [0.7, 0.2])
        ed.key(ord("3"))
        self.assertEqual(ed.selected, 2)
        ed.key(KEY_RIGHT)
        self.assertAlmostEqual(float(m.input[2, 0]), 1.0)                 # clamped at 1
        ed.key(ARROWS and 81)                                             # left
        self.assertAlmostEqual(float(m.input[2, 0]), 1.0 - 0.0025, places=5)
        ed.key(ord("o"))
        self.assertEqual(ed.mode, "output")
        self.assertIs(ed.quad, m.output)
        ed.move_to(0.3, 0.3)
        self.assertFalse(m.identity)
        ed.key(ord("r"))                                                  # reset the current (output) quad
        self.assertTrue(np.array_equal(m.output, UNIT_QUAD))
        self.assertFalse(np.array_equal(m.input, UNIT_QUAD))
        self.assertFalse(ed.key(ord("x")))
        self.assertFalse(ed.done)
        self.assertTrue(ed.key(13))
        self.assertTrue(ed.done)

    def test_drawing(self):
        ed = MappingEditor("0", CameraMapping(), "output")
        img = np.zeros((480, 640, 3), np.uint8)
        ed.draw_quad(img, 100, 0, 300, 480)
        ed.draw_info(img, "webcam 0")
        self.assertGreater(int(img[:, 100:400].mean()), 0)
        self.assertEqual(int(img[200:300, 450:640].mean()), 0)          # nothing outside the slot rectangle


class AppMappingTest(unittest.TestCase):
    def setUp(self):
        FakeCamera.opened, FakeCamera.released, FakeCamera.fail_indices = [], [], set()
        FakeCamera.open_now, FakeKinect.open_now = set(), set()
        FakeKinect.available = True
        patcher = mock.patch.object(app_module, "open_source", fake_open_source)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_app(self, source="0+2", **kw):
        kw.setdefault("pose", True)
        app = App(Config(source=source, preview=True, quiet=True, **kw), sender=CapturingSender())
        app.preview = True                                   # draw the preview even without a display
        app.cameras = CameraScanner(
            skip=app._cameras_in_use, on_done=app._on_scan_done,
            lister=lambda skip: [CameraInfo(c.index, c.name, kind=c.kind, in_use=c.spec in skip) for c in CAMS])
        app.panel = ControlPanel(app)
        app.panel.frame_width = 1280
        app.scan_cameras()
        app.cameras.wait(5)
        self.addCleanup(app.close)
        return app

    def test_gear_opens_the_editor_and_the_preview_shows_the_raw_camera(self):
        app = self.make_app()
        lst = app.panel.camera_list
        self.assertIsNone(app.editor)
        self.assertEqual(app.gear_state_row(1), "")
        right = lst.right_edge(len(CAMS))
        app.panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 1280 + right - 5, lst.y + lst.row_h + 3)   # gear of row 1 ("2")
        self.assertIsNotNone(app.editor)
        self.assertEqual(app.editor.spec, "2")
        self.assertEqual(app.editor.mode, "input")
        self.assertEqual(app.gear_state_row(1), "editing")
        app.step()
        view = app._latest_frame
        self.assertEqual(view.shape, (480, 1280, 3))                     # same window size as the composition
        raw = app._raw_slot
        self.assertEqual(raw.shape, (480, 640, 3))
        # drag the top-left handle of the input quad with the mouse (image coordinates of the view)
        app.editor_mouse(cv2.EVENT_LBUTTONDOWN, 2, 2)
        app.editor_mouse(cv2.EVENT_MOUSEMOVE, 320, 120)
        app.editor_mouse(cv2.EVENT_LBUTTONUP, 320, 120)
        np.testing.assert_allclose(app.mappings["2"].input[0], [0.25, 0.25], atol=0.01)
        self.assertEqual(app.gear_state_row(1), "editing")
        app.step()
        # the analyzed frame is now cropped: the picture of camera 2 is warped in its slot
        self.assertFalse(app.mappings["2"].identity)
        # keys go to the editor first
        self.assertTrue(app.handle_key(ord("o")))
        self.assertEqual(app.editor.mode, "output")
        app.step()
        self.assertEqual(app._latest_frame.shape, (480, 1280, 3))
        app.handle_key(ord("2"))
        app.editor_mouse(cv2.EVENT_LBUTTONDOWN, 640 + 638, 2)              # top-right handle, in slot 2
        app.editor_mouse(cv2.EVENT_MOUSEMOVE, 640 + 320, 240)
        app.editor_mouse(cv2.EVENT_LBUTTONUP, 640 + 320, 240)
        np.testing.assert_allclose(app.mappings["2"].output[1], [0.5, 0.5], atol=0.01)
        self.assertTrue(app.handle_key(13))                               # Enter closes
        self.assertIsNone(app.editor)
        self.assertEqual(app.gear_state_row(1), "mapped")
        self.assertFalse(app.handle_key(ord("q")))                        # q quits again once closed

    def test_gear_on_an_unchecked_camera_runs_it_first(self):
        app = self.make_app(source="0")
        app.toggle_mapping_editor_row(1)
        app.step()
        self.assertEqual(app.selection, ["0", "2"])
        self.assertEqual(app.editor.spec, "2")
        app.toggle_mapping_editor_row(1)                                  # the same gear closes it
        self.assertIsNone(app.editor)
        app.handle_key(ord("g"))                                          # g: edit the first camera
        self.assertEqual(app.editor.spec, "0")
        app.handle_key(27)
        self.assertIsNone(app.editor)

    def test_mirror_and_editor(self):
        app = self.make_app(source="0", mirror=True)
        app.open_mapping_editor("0", "output")
        app.step()
        app.editor_mouse(cv2.EVENT_LBUTTONDOWN, 639, 2)                  # mirrored: this is the top-LEFT corner
        self.assertEqual(app.editor.selected, 0)
        app.editor_mouse(cv2.EVENT_MOUSEMOVE, 639 - 64, 48)
        app.editor_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
        np.testing.assert_allclose(app.mappings["0"].output[0], [0.1, 0.1], atol=0.01)
        app.handle_key(ord("i"))
        app.step()
        self.assertEqual(app._latest_frame.shape, (480, 640, 3))

    def test_save_and_load_configuration(self):
        app = self.make_app()
        app.mappings["2"] = CameraMapping.from_dict(TPF_JSON)
        app.set_detectors(hands=True, blobs=True)
        app.set_blob_threshold(40)
        app.set_format("kinect", True)
        app.mirror = True
        app.set_template("minimal")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "show.json")
            self.assertIsNone(app.save_config(path))
            self.assertEqual(app.config_path, path)
            self.assertIn("saved", app.config_status()[0])
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["source"], "0+2")
            self.assertEqual(data["mappings"]["2"]["input"][0], {"x": 0.15, "y": 0.15})
            self.assertNotIn("0", data["mappings"])                        # identity mappings are not saved
            self.assertEqual(data["detectors"]["hands"], True)
            self.assertEqual(data["blobs"]["threshold"], 40)
            self.assertEqual(data["osc"]["formats"], ["mp", "kinect"])
            self.assertEqual(data["template"], "minimal")

            app.close()                                                    # frees the fake cameras
            other = self.make_app(source="0")
            self.assertIsNone(other.load_config(path))
            other.step()
            self.assertEqual(other.selection, ["0", "2"])
            self.assertEqual(other.source.specs, ["0", "2"])
            self.assertTrue(other.cfg.hands and other.cfg.blobs and other.analyzer.blobs_enabled)
            self.assertEqual(other.cfg.blob_threshold, 40)
            self.assertEqual(other.cfg.formats, ("mp", "kinect"))
            self.assertTrue(other.mirror)
            self.assertEqual(other.theme.name, "minimal")
            np.testing.assert_allclose(other.mappings["2"].output[2], [0.92, 0.92])
            self.assertIn("loaded", other.config_status()[0])

            # a broken file is reported, nothing crashes
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as fh:
                fh.write("[1, 2")
            self.assertIn("load failed", other.load_config(bad))
            self.assertTrue(other.config_status()[1])

            # --config on the command line
            cfg = parse_args(["--config", path, "--no-preview", "--source", "0"])
            self.assertEqual(cfg.config, path)
            other.close()
            third = App(Config(source="0", preview=False, quiet=True, config=path), sender=CapturingSender())
            third.cameras = CameraScanner(skip=third._cameras_in_use, lister=lambda skip: [])
            self.addCleanup(third.close)
            third.cfg.frames = 1
            third.run()
            self.assertEqual(third.cfg.blob_threshold, 40)
            self.assertEqual(third.selection, ["0", "2"])

    def test_save_without_a_path_asks_or_falls_back(self):
        app = self.make_app(source="0")
        with tempfile.TemporaryDirectory() as d:
            chosen = os.path.join(d, "chosen.json")
            with mock.patch.object(app_module, "__name__", "mp_osc.app"), \
                 mock.patch("mp_osc.dialogs.ask_save", return_value=chosen):
                self.assertIsNone(app.save_config())
            self.assertTrue(os.path.exists(chosen))
            with mock.patch("mp_osc.dialogs.ask_open", return_value=""):  # cancelled
                self.assertIsNone(app.load_config())
            with mock.patch("mp_osc.dialogs.ask_open", return_value=None):  # no tkinter: the known file
                self.assertIsNone(app.load_config())
            self.assertEqual(app.config_path, chosen)

    def test_editor_on_a_file_source(self):
        app = App(Config(source=PORTRAIT, preview=True, quiet=True), sender=CapturingSender())
        app.preview = True
        self.addCleanup(app.close)
        app.handle_key(ord("g"))
        self.assertEqual(app.editor.spec, PORTRAIT)
        self.assertEqual(app.selection, [])                               # a file is not a camera
        app.step()
        app.editor_mouse(cv2.EVENT_LBUTTONDOWN, 2, 2)
        app.editor_mouse(cv2.EVENT_MOUSEMOVE, 128, 85)
        app.editor_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
        self.assertFalse(app.mappings[PORTRAIT].identity)
        app.handle_key(13)
        app.step()
        self.assertEqual(app.to_config()["mappings"].keys(), {PORTRAIT})

    def test_dialogs_without_tkinter(self):
        with mock.patch.dict(sys.modules, {"tkinter": None}):
            self.assertIsNone(dialogs.ask_open())
            self.assertIsNone(dialogs.ask_save())


if __name__ == "__main__":
    unittest.main()
