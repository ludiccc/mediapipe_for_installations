"""Camera enumeration and the Source controls: Camera / Kinect / Pause radio group,
the camera list (several cameras side by side, ordering), depth from Kinects.

No real camera is needed: probing is replaced by a fake, and the app's
``open_source`` is patched to hand out fake camera / Kinect sources.
"""

import os
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc import app as app_module  # noqa: E402
from mp_osc import cameras  # noqa: E402
from mp_osc.app import App, main  # noqa: E402
from mp_osc.cameras import CameraInfo, CameraScanner, format_camera_list, list_cameras  # noqa: E402
from mp_osc.capture import (  # noqa: E402
    CompositeSource, FrameSource, KinectSource, open_source, parse_kinect_spec,
)
from mp_osc.config import Config  # noqa: E402
from mp_osc.ui import ControlPanel, ListBox, RadioButton, Toggle  # noqa: E402

PORTRAIT = os.path.join(ROOT, "brother-977170_1280.jpg")


class ListCamerasTest(unittest.TestCase):
    def test_contiguous_probing_stops_after_two_misses(self):
        probed = []

        def probe(i):
            probed.append(i)
            return i in (0, 1, 3)

        cams = list_cameras(probe=probe, platform="darwin", names=["FaceTime HD", "USB cam"], kinects=False)
        self.assertEqual([c.index for c in cams], [0, 1, 3])
        self.assertEqual(cams[0].name, "FaceTime HD")
        self.assertEqual(cams[1].name, "USB cam")
        self.assertEqual(cams[2].name, "Camera 3")
        self.assertEqual(probed, [0, 1, 2, 3, 4, 5])            # 4 and 5 miss -> stop
        self.assertEqual(cams[0].label, "0  FaceTime HD")
        self.assertEqual(cams[0].spec, "0")

    def test_skip_marks_the_camera_in_use_without_probing(self):
        probed = []
        cams = list_cameras(skip={"0": "webcam 0"}, probe=lambda i: probed.append(i) or False,
                            platform="win32", names=[], kinects=False)
        self.assertEqual([(c.index, c.in_use) for c in cams], [(0, True)])
        self.assertNotIn(0, probed)

    def test_linux_uses_dev_video_nodes_and_sysfs_names(self):
        nodes = {"/dev/video0": "Integrated Camera", "/dev/video1": "Integrated Camera",   # metadata node
                 "/dev/video4": "Logitech BRIO"}
        with mock.patch.object(cameras.glob, "glob", return_value=list(nodes)), \
             mock.patch.object(cameras, "_sysfs_name", side_effect=lambda d: nodes[d]):
            cams = list_cameras(probe=lambda i: i in (0, 4), platform="linux", kinects=False)
        self.assertEqual([(c.index, c.name, c.device) for c in cams],
                         [(0, "Integrated Camera", "/dev/video0"), (4, "Logitech BRIO", "/dev/video4")])
        text = format_camera_list(cams)
        self.assertIn("4  Logitech BRIO  (/dev/video4)", text)
        self.assertEqual(format_camera_list([]), "no cameras found")

    def test_kinects_are_listed_by_serial(self):
        import types
        fake = types.ModuleType("pylibfreenect2")

        class Freenect2:
            def enumerateDevices(self):
                return 2

            def getDeviceSerialNumber(self, i):
                return f"SER{i}"
        fake.Freenect2 = Freenect2
        with mock.patch.dict(sys.modules, {"pylibfreenect2": fake, "freenect": None}):
            cams = list_cameras(probe=lambda i: i == 0, platform="win32", names=[], skip={"kinect": "kinect (color)"})
        self.assertEqual([c.spec for c in cams], ["0", "kinect", "kinect:1"])
        self.assertEqual(cams[1].name, "Kinect v2 SER0")
        self.assertTrue(cams[1].in_use)
        self.assertFalse(cams[2].in_use)
        self.assertEqual(cams[1].kind, "kinect")
        # no library at all: the one in use is still listed from skip
        with mock.patch.dict(sys.modules, {"pylibfreenect2": None, "freenect": None}):
            cams = list_cameras(probe=lambda i: False, platform="win32", names=[], skip={"kinect": "kinect (depth)"})
        self.assertEqual([(c.spec, c.name, c.in_use) for c in cams], [("kinect", "kinect (depth)", True)])

    def test_linux_asks_the_kernel_and_retries_fresh_cameras(self):
        nodes = ["/dev/video0", "/dev/video1", "/dev/video4", "/dev/video5"]
        caps = {"/dev/video0": True, "/dev/video1": False, "/dev/video4": True, "/dev/video5": None}
        attempts = {}

        def probe(i):
            attempts[i] = attempts.get(i, 0) + 1
            return i == 0 or (i == 4 and attempts[i] >= 3)          # just plugged: opens on the 3rd try

        with mock.patch.object(cameras.glob, "glob", return_value=nodes), \
             mock.patch.object(cameras, "_sysfs_name", return_value="cam"):
            cams = list_cameras(probe=probe, platform="linux", kinects=False,
                                is_capture=caps.__getitem__, retry_delay=0.0)
        self.assertEqual([c.index for c in cams], [0, 4])
        self.assertNotIn(1, attempts)                                # metadata node: never opened
        self.assertEqual(attempts[4], 3)                             # retried until it opened
        self.assertEqual(attempts[5], 1)                             # unknown kind: one try only

    def test_flags_switch_kinds_off(self):
        with mock.patch.dict(sys.modules, {"pylibfreenect2": None, "freenect": None}):
            self.assertEqual(list_cameras(probe=lambda i: i == 0, platform="win32", names=[], webcams=False), [])
            cams = list_cameras(probe=lambda i: i == 0, platform="win32", names=[], kinects=False)
        self.assertEqual([c.spec for c in cams], ["0"])

    def test_real_probe_of_a_missing_index_is_false(self):
        self.assertFalse(cameras._probe(250))

    def test_scanner_runs_in_background_and_reports(self):
        done = []
        scanner = CameraScanner(skip=lambda: {"7": "mine"}, on_done=lambda: done.append(True),
                                lister=lambda skip: [CameraInfo(7, "mine", in_use="7" in skip)])
        self.assertTrue(scanner.start())
        scanner.wait(5)
        self.assertTrue(scanner.done)
        self.assertEqual(done, [True])
        self.assertEqual(scanner.cameras[0].in_use, True)
        self.assertFalse(scanner.scanning)
        self.assertEqual(scanner.by_spec("7").name, "mine")
        self.assertIsNone(scanner.by_spec("8"))
        broken = CameraScanner(lister=lambda skip: 1 / 0)
        broken.start()
        broken.wait(5)
        self.assertIn("division", broken.error)
        self.assertEqual(broken.cameras, [])

    def test_list_cameras_cli(self):
        with mock.patch.object(app_module, "list_cameras", return_value=[CameraInfo(2, "Cam")]), \
             mock.patch("sys.stdout") as out:
            self.assertEqual(main(["--list-cameras"]), 0)
        printed = "".join(str(c.args[0]) for c in out.write.call_args_list)
        self.assertIn("2  Cam", printed)

    def test_kinect_spec_parsing(self):
        self.assertEqual(parse_kinect_spec("kinect"), ("auto", 0))
        self.assertEqual(parse_kinect_spec("kinect:1"), ("auto", 1))
        self.assertEqual(parse_kinect_spec("kinect:freenect"), ("freenect", 0))
        self.assertEqual(parse_kinect_spec("KINECT:freenect2:2"), ("freenect2", 2))
        with self.assertRaises(ValueError):
            parse_kinect_spec("kinect:foo")
        with self.assertRaises(ValueError):
            parse_kinect_spec("0")


class KinectSourceTest(unittest.TestCase):
    def test_depth_to_image_near_is_bright(self):
        depth = np.array([[0, 0.5, 2.0, 4.0, 9.0]], dtype=np.float32)
        img = KinectSource.depth_to_image(depth, 4.0)
        self.assertEqual(img.shape, (1, 5, 3))
        self.assertEqual(list(img[0, :, 0]), [0, 223, 127, 0, 0])    # unknown = black, near = bright

    def test_without_hardware_a_clear_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            open_source("kinect")
        self.assertIn("no Kinect found", str(ctx.exception))
        with self.assertRaises(ValueError):
            KinectSource("xyz")


# ---------------------------------------------------------------------------
# fake live sources for the app-level tests
# ---------------------------------------------------------------------------
class FakeCamera(FrameSource):
    """Behaves like a V4L2 webcam: opening an index that is already open fails (busy)."""
    kind = "camera"
    opened = []
    released = []
    open_now = set()
    fail_indices = set()

    def __init__(self, index, width=640, height=480, name=""):
        super().__init__()
        if index in self.fail_indices:
            raise RuntimeError(f"could not open webcam {index}")
        if index in FakeCamera.open_now:
            raise RuntimeError(f"could not open webcam {index} (busy: already open)")
        FakeCamera.open_now.add(index)
        self.index = index
        self.name = f"webcam {index}" + (f" ({name})" if name else "")
        self.frame = cv2.resize(cv2.imread(PORTRAIT), (width, height))    # somebody is in front of it
        cv2.putText(self.frame, str(index), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        FakeCamera.opened.append(index)

    def read(self):
        return True, self.frame.copy()

    def release(self):
        FakeCamera.released.append(self.index)
        FakeCamera.open_now.discard(self.index)


class FakeKinect(FrameSource):
    """Like libfreenect2: opening a Kinect that is already open is fatal (it segfaults for real)."""
    kind = "kinect"
    has_depth = True
    available = False
    open_now = set()

    def __init__(self, stream="color", width=640, height=480, index=0):
        super().__init__()
        if not self.available:
            raise RuntimeError("no Kinect found (fake)")
        if index in FakeKinect.open_now:
            raise AssertionError(f"Kinect {index} opened twice - libfreenect2 would crash here")
        FakeKinect.open_now.add(index)
        self.index = index
        self.name = f"kinect ({stream}) #{index}"
        self.stream = stream
        self.frame = cv2.resize(cv2.imread(PORTRAIT), (width, height))
        self._depth = np.full((height, width), 1.5 + index, dtype=np.float32)   # 1.5 m / 2.5 m

    def read(self):
        return True, self.frame.copy()

    def depth(self):
        return self._depth if self.want_depth else None

    def release(self):
        FakeKinect.open_now.discard(self.index)


def fake_open_source(source, width=640, height=480, loop=True, kinect_stream="color", camera_name="",
                     kinect_backend="auto", kinect_pipeline="auto", camera_names=None):
    source = str(source)
    if "+" in source:
        parts = []
        for spec in source.split("+"):
            p = fake_open_source(spec, width, height, loop, kinect_stream, "", kinect_backend, kinect_pipeline,
                                 camera_names)
            p.spec = spec
            parts.append(p)
        return CompositeSource(parts)
    if source.startswith("kinect"):
        index = int(source.split(":")[1]) if ":" in source else 0
        src = FakeKinect(kinect_stream, width, height, index)
    elif source.isdigit():
        src = FakeCamera(int(source), width, height, (camera_names or {}).get(int(source), ""))
    else:
        src = open_source(source, width, height, loop)
    src.spec = source
    return src


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


CAMS = [CameraInfo(0, "Integrated"), CameraInfo(2, "USB cam"), CameraInfo(5, "Broken"),
        CameraInfo(0, "Kinect v2 A", kind="kinect"), CameraInfo(1, "Kinect v2 B", kind="kinect")]


class SourceSwitchingTest(unittest.TestCase):
    def setUp(self):
        FakeCamera.opened, FakeCamera.released, FakeCamera.fail_indices = [], [], set()
        FakeCamera.open_now, FakeKinect.open_now = set(), set()
        FakeKinect.available = False
        patcher = mock.patch.object(app_module, "open_source", fake_open_source)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_app(self, source="0", **kw):
        kw.setdefault("pose", True)
        cfg = Config(source=source, preview=False, quiet=True, **kw)
        app = App(cfg, sender=CapturingSender())
        in_use = app._cameras_in_use
        app.cameras = CameraScanner(
            skip=in_use, on_done=app._on_scan_done,
            lister=lambda skip: [CameraInfo(c.index, c.name, kind=c.kind, in_use=c.spec in skip) for c in CAMS])
        app.panel = ControlPanel(app)
        app.panel.frame_width = 640
        app.scan_cameras()
        app.cameras.wait(5)
        self.addCleanup(app.close)
        return app

    def radios(self, app):
        return [w for w in app.panel.widgets if isinstance(w, RadioButton)]

    def test_radio_group_is_exclusive(self):
        app = self.make_app()
        radios = self.radios(app)
        self.assertEqual([r.label for r in radios], ["Cameras", "Pause"])   # no File: started on a camera
        self.assertEqual([r.selected for r in radios], [True, False])
        app.panel.click("Pause")
        self.assertEqual(app.mode, "pause")
        self.assertTrue(app.sender.paused)
        self.assertEqual([r.selected for r in radios], [False, True])     # only one at a time
        app.panel.click("Cameras")
        self.assertEqual(app.mode, "camera")
        self.assertFalse(app.sender.paused)
        self.assertEqual(FakeCamera.opened, [0])                                # same camera: not reopened
        self.assertEqual(sum(r.selected for r in radios), 1)

    def test_pause_key_and_radio_agree(self):
        app = self.make_app()
        app.handle_key(ord("p"))
        self.assertEqual(app.mode, "pause")
        app.handle_key(ord("p"))
        self.assertEqual(app.mode, "camera")

    def test_camera_list_and_single_selection(self):
        app = self.make_app()
        self.assertEqual(app.camera_labels()[:3], ["1. 0  Integrated", "   2  USB cam", "   5  Broken"])
        self.assertEqual(app.selected_camera_rows(), [0])
        self.assertTrue(app.cameras.cameras[0].in_use)

        lst = app.panel.camera_list
        self.assertIsInstance(lst, ListBox)
        app.step()
        lst.select_label("2  USB cam")                  # check the USB cam -> composition of two
        app.step()                                      # the switch happens on the analysis thread (= here)
        self.assertEqual(app.selection, ["0", "2"])
        self.assertIsInstance(app.source, CompositeSource)
        self.assertEqual(app.source.name, "webcam 0 (Integrated) + webcam 2 (USB cam)")
        self.assertEqual(app.cfg.source, "0+2")
        lst.select_label("1. 0  Integrated")            # uncheck the first one -> USB cam alone
        app.step()
        self.assertEqual(app.selection, ["2"])
        self.assertEqual(app.source.index, 2)
        self.assertEqual(app.source.name, "webcam 2 (USB cam)")
        self.assertIn(0, FakeCamera.released)
        self.assertEqual(app.selected_camera_rows(), [1])
        self.assertEqual(app.source_note(), ("webcam 2 (USB cam)", False))
        self.assertEqual(app.sender.by_address("/mp/pose/leave")[0], [0])     # receivers told: everybody left
        app.step()
        self.assertGreater(app.sender.by_address("/mp/pose/enter")[-1][0], 0)  # fresh ids on the new source
        lst.select_label("1. 2  USB cam")               # the last checked camera cannot be unchecked
        self.assertEqual(app.selection, ["2"])

        # clicking a row through the mouse (a camera that cannot be opened)
        FakeCamera.fail_indices = {5}
        row = lst.y + 2 * lst.row_h + 3
        app.panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 640 + lst.x + 10, row)
        app.step()
        self.assertEqual(app.source.index, 2)                                 # failed: old camera kept
        self.assertEqual(app.selection, ["2"])                                # and the list shows it
        self.assertIn("could not open webcam 5", app.source_note()[0])
        self.assertTrue(app.source_note()[1])
        app.scan_cameras()                                                    # clears the error
        app.cameras.wait(5)
        self.assertFalse(app.source_note()[1])

    def test_side_by_side_composition_and_ordering(self):
        FakeKinect.available = True
        app = self.make_app(source="0+2")
        self.assertEqual(app.selection, ["0", "2"])
        self.assertEqual(app.mode, "camera")
        self.assertEqual([r.selected for r in self.radios(app)], [True, False])
        app.step()
        ok, frame = app.source.read()
        self.assertEqual(frame.shape, (480, 1280, 3))                        # two 640x480 side by side
        self.assertFalse(app.depth_available)                                 # webcams: no depth

        # < > buttons of a checked row: move "2" to the left. (5 cameras, 4
        # visible rows: the scroll strip takes the far right of the box.)
        lst = app.panel.camera_list
        row_y = lst.y + 1 * lst.row_h + 3
        right = lst.right_edge(len(CAMS)) - lst.GEAR_W            # the < > pair sits left of the gear
        app.panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 640 + right - lst.MOVE_W + 4, row_y)   # "<"
        app.step()
        self.assertEqual(app.selection, ["2", "0"])
        self.assertEqual(app.camera_labels()[:2], ["2. 0  Integrated", "1. 2  USB cam"])
        self.assertEqual(app.source.specs, ["2", "0"])                        # re-ordered in place...
        self.assertEqual(FakeCamera.opened, [0, 2])                           # ...without re-opening
        self.assertEqual(app.cfg.source, "2+0")
        app.panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 640 + right - 4, row_y)                       # ">"
        app.step()
        self.assertEqual(app.selection, ["0", "2"])
        app.move_camera_row(1, +1)                                            # already rightmost: no-op
        self.assertEqual(app.selection, ["0", "2"])
        app.move_camera_row(2, -1)                                            # not checked: no-op
        self.assertEqual(app.selection, ["0", "2"])

        # add a Kinect to the mix: still a composition, no depth (webcam in it).
        # The two webcams stay open as they are (the fakes would refuse a second open).
        lst.select_label("Kinect v2 A")
        app.step()
        self.assertEqual(app.selection, ["0", "2", "kinect"])
        self.assertEqual(app.mode, "camera")
        self.assertFalse(app.depth_available)
        self.assertEqual(app.z_source, "mediapipe")
        self.assertEqual(FakeCamera.opened, [0, 2])                           # never re-opened
        self.assertEqual(FakeCamera.released, [])
        self.assertEqual(FakeKinect.open_now, {0})
        # drop the first webcam: only that one is released, the Kinect keeps streaming
        lst.select_label("0  Integrated")
        app.step()
        self.assertEqual(app.selection, ["2", "kinect"])
        self.assertEqual(FakeCamera.released, [0])
        self.assertEqual(FakeKinect.open_now, {0})
        self.assertEqual(app.source.name, "webcam 2 (USB cam) + kinect (color) #0")

    def test_kinect_plus_kinect_has_depth_and_z_toggle(self):
        FakeKinect.available = True
        app = self.make_app(source="kinect+kinect:1")
        self.assertTrue(app.depth_available)
        self.assertEqual(app.source.name, "kinect (color) #0 + kinect (color) #1")
        z = app.panel.find("Z from Kinect  (z)")
        self.assertIsInstance(z, Toggle)
        self.assertTrue(z.enabled())
        self.assertEqual(app.z_source, "mediapipe")
        app.step()
        before = app.sender.by_address("/mp/pose/center")[-1]
        app.panel.click("Z from Kinect  (z)")
        self.assertTrue(app.cfg.z_from_kinect)
        self.assertEqual(app.z_source, "kinect")
        app.step()
        after = app.sender.by_address("/mp/pose/center")[-1]
        self.assertIn(round(after[3], 2), (1.5, 2.5))                         # metres: left or right Kinect
        self.assertNotIn(round(before[3], 2), (1.5, 2.5))
        self.assertEqual(app.sender.by_address("/mp/z_source"), [["mediapipe"], ["kinect"]])
        lms = app.sender.by_address("/mp/pose/landmarks")[-1]
        zs = lms[1 + 2::4]                                                    # every z of the 33 landmarks
        self.assertTrue(all(round(v, 2) in (1.5, 2.5, 0.0) for v in zs))      # 0 where outside the image
        self.assertTrue(any("z: kinect" in line for line in app.status_lines()))
        app.handle_key(ord("z"))                                              # key toggles it back
        self.assertFalse(app.cfg.z_from_kinect)
        app.step()
        self.assertEqual(app.sender.by_address("/mp/z_source")[-1], ["mediapipe"])

        # adding a webcam removes the depth: the toggle greys out, z is MediaPipe's again
        app.cfg.z_from_kinect = True
        app.panel.camera_list.select_label("0  Integrated")
        app.step()
        self.assertFalse(app.depth_available)
        self.assertFalse(z.enabled())
        self.assertEqual(app.z_source, "mediapipe")
        img = app.panel.render(np.zeros((480, 640, 3), np.uint8), app.status_lines())
        self.assertEqual(img.shape[1], 640 + app.theme.panel_width)
        z.click()                                                             # disabled: ignored
        self.assertTrue(app.cfg.z_from_kinect)

    def test_kinect_missing_keeps_camera_and_shows_error(self):
        app = self.make_app()
        lst = app.panel.camera_list
        lst.select_label("Kinect v2 A")    # (without an analysis thread the switch is immediate)
        app.step()
        self.assertEqual(app.source.kind, "camera")
        self.assertEqual(app.mode, "camera")
        self.assertIn("no Kinect found", app.source_note()[0])
        self.assertEqual(app.selection, ["0"])                                # list shows what runs
        FakeKinect.available = True
        lst.select_label("Kinect v2 A")
        app.step()
        lst.select_label("0  Integrated")                                     # uncheck the webcam
        app.step()
        self.assertEqual(app.mode, "camera")                                  # a Kinect is just a camera
        self.assertEqual(app.source.kind, "kinect")
        self.assertEqual(app.selection, ["kinect"])
        self.assertEqual(app.source.stream, "color")                          # pose is on -> colour stream
        self.assertTrue(app.depth_available)
        self.assertEqual(FakeCamera.released, [0])
        self.assertEqual([r.label for r in self.radios(app)], ["Cameras", "Pause"])
        self.assertEqual([r.selected for r in self.radios(app)], [True, False])

    def test_switching_never_opens_a_device_twice(self):
        """The bug of the field log: adding a camera re-opened the running ones (busy / crash)."""
        FakeKinect.available = True
        app = self.make_app(source="kinect")
        lst = app.panel.camera_list
        lst.select_label("0  Integrated")                                     # kinect + webcam
        app.step()
        self.assertEqual(app.selection, ["kinect", "0"])
        self.assertEqual(app.source_note()[1], False, app.source_note())      # no error
        self.assertEqual(FakeKinect.open_now, {0})
        app.move_camera_row(3, +1)                                            # re-order (row 3 = the Kinect): nothing re-opened
        app.step()
        self.assertEqual(app.selection, ["0", "kinect"])
        self.assertEqual(FakeCamera.opened, [0])
        self.assertEqual(app.source.specs, ["0", "kinect"])
        lst.select_label("2  USB cam")
        app.step()
        self.assertEqual(app.source.specs, ["0", "kinect", "2"])
        # a failing new part leaves everything as it was (and releases nothing)
        FakeCamera.fail_indices = {5}
        lst.select_label("5  Broken")
        app.step()
        self.assertEqual(app.source.specs, ["0", "kinect", "2"])
        self.assertEqual(app.selection, ["0", "kinect", "2"])
        self.assertEqual(FakeCamera.released, [])
        self.assertTrue(app.source_note()[1])
        app.close()
        self.assertEqual(FakeKinect.open_now, set())
        self.assertEqual(FakeCamera.open_now, set())

    def test_kinect_depth_stream_when_only_blobs(self):
        FakeKinect.available = True
        app = self.make_app(source="kinect", pose=False, blobs=True)
        self.assertEqual(app.mode, "camera")
        self.assertEqual(app.source.stream, "depth")
        self.assertEqual([r.label for r in self.radios(app)], ["Cameras", "Pause"])
        self.assertEqual(app.selected_camera_rows(), [3])
        self.assertEqual(app.camera_labels()[3], "1. kinect  Kinect v2 A")

    def test_no_webcams_and_no_kinects_flags(self):
        FakeKinect.available = True
        app = self.make_app(source="", no_webcams=True)
        self.assertEqual(app.cfg.source, "kinect")                            # the default follows the flag
        self.assertEqual(app.source.kind, "kinect")
        with self.assertRaises(ValueError):
            Config(no_webcams=True, no_kinects=True)
        self.assertEqual(Config(preview=False).source, "0")
        self.assertEqual(Config(preview=False, no_kinects=True).source, "0")

    def test_file_source_gets_a_file_option_and_can_return(self):
        app = self.make_app(source=PORTRAIT)
        self.assertEqual(app.mode, "file")
        self.assertEqual([r.label for r in self.radios(app)], ["Cameras", "Pause", "File"])
        self.assertEqual(app.selected_camera_rows(), [])
        app.step()
        t1 = app._last_t
        app.panel.click("Cameras")
        app.step()
        self.assertEqual(app.source.kind, "camera")
        self.assertEqual(app.selection, ["0"])
        self.assertGreaterEqual(app._last_t, t1)                              # time keeps increasing
        app.panel.click("File")
        app.step()
        self.assertEqual(app.mode, "file")
        self.assertEqual(app.source.kind, "image")
        app.panel.click("Pause")
        app.handle_key(ord("p"))                                              # un-pause returns to the file
        self.assertEqual(app.mode, "file")

    def test_status_mentions_source_and_blobs(self):
        app = self.make_app()
        lines = app.status_lines()
        self.assertTrue(any("source: webcam 0" in line for line in lines))
        self.assertTrue(any("blobs" in line for line in lines))
        self.assertTrue(any("z: mediapipe" in line for line in lines))


class CompositeSourceTest(unittest.TestCase):
    class Part(FrameSource):
        kind = "camera"

        def __init__(self, spec, w, h, value, depth=None):
            super().__init__()
            self.spec, self.name = spec, f"part {spec}"
            self.frame = np.full((h, w, 3), value, np.uint8)
            self.has_depth = depth is not None
            self._depth = None if depth is None else np.full((h, w), depth, np.float32)
            self.released = False

        def read(self):
            return True, self.frame.copy()

        def depth(self):
            return self._depth if self.want_depth else None

        def release(self):
            self.released = True

    def test_parts_are_scaled_to_the_first_height_and_stacked(self):
        a = self.Part("a", 640, 480, 10)
        b = self.Part("b", 320, 480, 20)
        c = self.Part("c", 1920, 1080, 30)
        comp = CompositeSource([a, b, c])
        ok, frame = comp.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (480, 640 + 320 + 853, 3))
        self.assertEqual(int(frame[10, 10, 0]), 10)
        self.assertEqual(int(frame[10, 700, 0]), 20)
        self.assertEqual(int(frame[10, 1200, 0]), 30)
        self.assertEqual(comp.name, "part a + part b + part c")
        self.assertEqual(comp.specs, ["a", "b", "c"])
        self.assertFalse(comp.has_depth)
        self.assertIsNone(comp.depth())
        self.assertTrue(comp.set_order(["c", "a", "b"]))
        self.assertEqual(comp.specs, ["c", "a", "b"])
        self.assertFalse(comp.set_order(["c", "a"]))
        ok, frame = comp.read()
        self.assertEqual(frame.shape, (1080, 1920 + 1440 + 720, 3))          # first part sets the height now
        comp.release()
        self.assertTrue(a.released and b.released and c.released)
        with self.assertRaises(ValueError):
            CompositeSource([])

    def test_depth_only_when_every_part_has_it(self):
        k1 = self.Part("k1", 512, 424, 10, depth=1.0)
        k2 = self.Part("k2", 1024, 848, 20, depth=2.0)
        comp = CompositeSource([k1, k2])
        self.assertTrue(comp.has_depth)
        comp.want_depth = True
        ok, frame = comp.read()
        self.assertEqual(frame.shape, (424, 1024, 3))
        depth = comp.depth()
        self.assertEqual(depth.shape, (424, 1024))
        self.assertEqual(float(depth[0, 0]), 1.0)
        self.assertEqual(float(depth[0, 1000]), 2.0)
        comp.want_depth = False
        comp.read()
        self.assertIsNone(comp.depth())
        mixed = CompositeSource([k1, self.Part("w", 640, 480, 5)])
        mixed.want_depth = True
        mixed.read()
        self.assertFalse(mixed.has_depth)
        self.assertIsNone(mixed.depth())

    def test_open_source_builds_a_composite_from_a_plus_spec(self):
        comp = open_source(f"{PORTRAIT}+{PORTRAIT}", loop=False)
        self.assertIsInstance(comp, CompositeSource)
        ok, frame = comp.read()
        self.assertEqual(frame.shape[1], 2 * cv2.imread(PORTRAIT).shape[1])
        self.assertEqual(comp.spec, f"{PORTRAIT}+{PORTRAIT}")
        comp.release()
        single = open_source(PORTRAIT + "+")                                  # a stray plus is ignored
        self.assertEqual(single.kind, "image")


if __name__ == "__main__":
    unittest.main()
