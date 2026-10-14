"""Kinect v2 through pylibfreenect2 - against a fake module that mirrors the real API.

The real library needs libfreenect2 and a sensor, so the tests install a
stand-in ``pylibfreenect2`` in ``sys.modules`` with the same names and calling
conventions (``Freenect2().enumerateDevices()``, ``getDeviceSerialNumber``,
``openDevice(serial, pipeline=...)``, ``SyncMultiFrameListener``,
``FrameType``, ``waitForNewFrame`` / ``release``, ``Frame.asarray``).
"""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_osc.app import App  # noqa: E402
from mp_osc.capture import KinectSource, open_source  # noqa: E402
from mp_osc.config import Config, parse_args  # noqa: E402


class FakeFrame:
    def __init__(self, array):
        self._array = array

    def asarray(self, dtype=np.float32):
        if self._array.dtype == np.uint8 and dtype is not np.uint8:
            raise ValueError("color frames must be read as uint8")     # like the real binding
        return self._array


class FakeListener:
    instances = []

    def __init__(self, frame_types):
        self.frame_types = frame_types
        self.released = []
        self.new_frames = True
        FakeListener.instances.append(self)

    def hasNewFrame(self):
        return self.new_frames

    def waitForNewFrame(self):
        color = np.zeros((1080, 1920, 4), dtype=np.uint8)
        color[:, :, 0] = 10; color[:, :, 1] = 20; color[:, :, 2] = 30; color[:, :, 3] = 255
        depth = np.zeros((424, 512), dtype=np.float32)
        depth[100:200, 100:200] = 1000.0       # something 1 m away
        depth[300:400, 300:400] = 4000.0       # something far
        return {"color": FakeFrame(color), "depth": FakeFrame(depth), "ir": FakeFrame(depth)}

    def release(self, frames):
        self.released.append(frames)


class FakeFrameBuffer:
    """pylibfreenect2.Frame(width, height, bytes_per_pixel) as used for undistorted/registered/bigdepth."""

    def __init__(self, width, height, bpp):
        self.width, self.height, self.bpp = width, height, bpp
        self.data = np.zeros((height, width), dtype=np.float32)

    def asarray(self, dtype=None):
        if dtype is None:                                  # exactly what pylibfreenect2 does
            raise ValueError("Cannot determine type of data. Specify dtype explicitly.")
        return self.data


class FakeRegistration:
    """Fills bigdepth (1920x1082) from the 512x424 depth: here simply an upscale + 1 padding row."""

    calls = 0

    def __init__(self, ir_params, color_params):
        assert ir_params == "ir" and color_params == "color"

    def apply(self, color, depth, undistorted, registered, enable_filter=True, bigdepth=None, color_depth_map=None):
        FakeRegistration.calls += 1
        import cv2
        d = cv2.resize(depth.asarray(), (1920, 1080), interpolation=cv2.INTER_NEAREST)
        bigdepth.data[:] = np.inf                         # libfreenect2 marks unknown pixels with inf
        bigdepth.data[1:-1, :] = np.where(d > 0, d, np.inf)


class FakeDevice:
    def __init__(self, serial, pipeline):
        self.serial, self.pipeline = serial, pipeline
        self.listeners = []
        self.started = self.stopped = self.closed = False

    def getIrCameraParams(self):
        return "ir"

    def getColorCameraParams(self):
        return "color"

    def setColorFrameListener(self, listener):
        self.listeners.append(("color", listener))

    def setIrAndDepthFrameListener(self, listener):
        self.listeners.append(("depth", listener))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def make_fake_module(num_devices=1, pipelines=("cpu",), opengl_fails=False):
    m = types.ModuleType("pylibfreenect2")
    devices = []

    class Freenect2:
        def enumerateDevices(self):
            return num_devices

        def getDeviceSerialNumber(self, index):
            return f"SERIAL{index}"

        def openDevice(self, serial, pipeline=None):
            d = FakeDevice(serial, pipeline)
            devices.append(d)
            return d

    class FrameType:
        Color, Ir, Depth = 1, 2, 4

    m.Freenect2 = Freenect2
    m.FrameType = FrameType
    m.SyncMultiFrameListener = FakeListener
    m.Registration = FakeRegistration
    m.Frame = FakeFrameBuffer
    m.devices = devices
    for name, cls_name in (("opengl", "OpenGLPacketPipeline"), ("opencl", "OpenCLPacketPipeline"),
                           ("cuda", "CudaPacketPipeline"), ("cpu", "CpuPacketPipeline")):
        if name in pipelines:
            if name == "opengl" and opengl_fails:
                def boom(*a, **k):
                    raise RuntimeError("no OpenGL context")
                setattr(m, cls_name, boom)
            else:
                setattr(m, cls_name, type(cls_name, (), {"kind": name}))
    return m


class Freenect2BackendTest(unittest.TestCase):
    def setUp(self):
        FakeListener.instances = []

    def with_module(self, module):
        patcher = mock.patch.dict(sys.modules, {"pylibfreenect2": module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_color_stream_opens_like_the_reference_code(self):
        m = make_fake_module(pipelines=("opengl", "opencl", "cpu"))
        self.with_module(m)
        src = open_source("kinect", width=640, height=480, kinect_stream="color")
        self.assertEqual(src.kind, "kinect")
        self.assertEqual(src.backend, "freenect2")
        self.assertIn("SERIAL0", src.name)
        self.assertIn("opengl pipeline", src.name)                      # first of the auto order
        dev = m.devices[0]
        self.assertEqual(dev.pipeline.kind, "opengl")
        self.assertEqual([kind for kind, _ in dev.listeners], ["color", "depth"])
        self.assertTrue(dev.started)
        listener = FakeListener.instances[0]
        self.assertEqual(listener.frame_types, m.FrameType.Color | m.FrameType.Depth)   # depth for "Z from Kinect"

        ok, frame = src.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (360, 640, 3))                    # BGRX -> BGR, scaled to the width
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(list(frame[0, 0]), [10, 20, 30])
        self.assertEqual(len(listener.released), 1)                     # every frame map is released
        src.read()
        self.assertEqual(len(listener.released), 2)
        src.release()
        self.assertTrue(dev.stopped and dev.closed)

    def test_aligned_depth_for_z_from_kinect(self):
        FakeRegistration.calls = 0
        self.with_module(make_fake_module())
        src = open_source("kinect", width=640, kinect_stream="color")
        ok, frame = src.read()
        self.assertIsNone(src.depth())                                  # not asked for: no registration
        self.assertEqual(FakeRegistration.calls, 0)
        src.want_depth = True
        ok, frame = src.read()
        depth = src.depth()
        self.assertEqual(FakeRegistration.calls, 1)
        self.assertEqual(frame.shape, (360, 640, 3))
        self.assertEqual(depth.shape, (360, 640))                      # scaled like the colour frame
        self.assertEqual(depth.dtype, np.float32)
        # fake depth: 1000 mm in rows 100..200 / cols 100..200 of the 512x424 map -> 1.0 m there
        self.assertAlmostEqual(float(depth[int(150 / 424 * 360), int(150 / 512 * 640)]), 1.0, places=3)
        self.assertEqual(float(depth[5, 5]), 0.0)                       # unknown (inf) -> 0
        self.assertTrue(src.has_depth)
        src.release()

    def test_depth_failure_does_not_kill_the_colour_stream(self):
        self.with_module(make_fake_module())
        src = open_source("kinect", width=640, kinect_stream="color")
        src.want_depth = True
        with mock.patch.object(FakeRegistration, "apply", side_effect=RuntimeError("registration exploded")):
            ok, frame = src.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (360, 640, 3))
        self.assertIsNone(src.depth())
        self.assertIn("registration exploded", src.depth_error)
        ok, frame = src.read()                                          # keeps working, depth stays off
        self.assertTrue(ok)
        self.assertIsNone(src.depth())
        src.release()

    def test_depth_stream_near_is_bright(self):
        self.with_module(make_fake_module())
        src = open_source("kinect", width=0, kinect_stream="depth")
        self.assertEqual(FakeListener.instances[0].frame_types, 4)      # Depth only
        ok, frame = src.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (424, 512, 3))                    # native size when width = 0
        self.assertEqual(int(frame[0, 0, 0]), 0)                        # unknown = black
        self.assertGreater(int(frame[150, 150, 0]), int(frame[350, 350, 0]))   # 1 m brighter than 4 m
        self.assertGreater(int(frame[350, 350, 0]), 0)
        self.assertEqual(float(src.depth()[150, 150]), 1.0)             # the depth stream always has depth()
        src.release()

    def test_pipeline_fallback_and_forcing(self):
        m = make_fake_module(pipelines=("opengl", "cpu"), opengl_fails=True)
        self.with_module(m)
        src = KinectSource("color", pipeline="auto")
        self.assertIn("cpu pipeline", src.name)                         # opengl failed, opencl/cuda absent
        src.release()
        src = KinectSource("color", pipeline="cpu")
        self.assertEqual(m.devices[-1].pipeline.kind, "cpu")
        src.release()
        with self.assertRaises(RuntimeError) as ctx:
            KinectSource("color", pipeline="opencl")                    # forced but not available
        self.assertIn("opencl", str(ctx.exception))
        self.with_module(make_fake_module(pipelines=()))
        src = KinectSource("color")
        self.assertIn("default pipeline", src.name)                     # library chooses
        self.assertIsNone(src._backend.device.pipeline)
        src.release()

    def test_no_device_gives_a_clear_error_and_other_backends_are_tried(self):
        self.with_module(make_fake_module(num_devices=0))
        with self.assertRaises(RuntimeError) as ctx:
            open_source("kinect")
        msg = str(ctx.exception)
        self.assertIn("no Kinect found", msg)
        self.assertIn("found no Kinect v2", msg)
        self.assertIn("USB 3", msg)
        self.assertIn("freenect", msg)                                  # v1 and OpenNI were tried too
        with self.assertRaises(RuntimeError):
            KinectSource("color", backend="freenect2")
        with self.assertRaises(ValueError):
            KinectSource("color", backend="kinectron")

    def test_read_times_out_instead_of_hanging(self):
        self.with_module(make_fake_module())
        src = KinectSource("color")
        src._backend.timeout = 0.05
        FakeListener.instances[0].new_frames = False
        self.assertEqual(src.read(), (False, None))
        src.release()

    def test_preload_hint(self):
        self.with_module(make_fake_module())
        with mock.patch.dict(os.environ, {"LIBFREENECT2_LIBRARY": "/no/such/libfreenect2.so"}):
            src = KinectSource("color")                                 # the module imports anyway: fine
            src.release()
        with mock.patch.dict(sys.modules, {"pylibfreenect2": None}), \
             mock.patch.dict(os.environ, {"LIBFREENECT2_LIBRARY": "/no/such/libfreenect2.so"}):
            with self.assertRaises(RuntimeError) as ctx:
                KinectSource("color", backend="freenect2")
        self.assertIn("LIBFREENECT2_LIBRARY", str(ctx.exception))

    def test_cli_options(self):
        cfg = parse_args(["--source", "kinect", "--kinect-backend", "freenect2", "--kinect-pipeline", "cpu"])
        self.assertEqual((cfg.kinect_backend, cfg.kinect_pipeline), ("freenect2", "cpu"))
        with self.assertRaises(ValueError):
            Config(kinect_backend="x")
        with self.assertRaises(ValueError):
            Config(kinect_pipeline="x")

    def test_app_runs_on_the_kinect_depth_stream(self):
        self.with_module(make_fake_module())
        sender = mock.Mock(ip="127.0.0.1", port=5005, paused=False, sent=0)
        app = App(Config(source="kinect", blobs=True, blob_mode="bright", blob_threshold=20, blob_min_area=0.01,
                         preview=False, quiet=True, width=512, kinect_pipeline="cpu"), sender=sender)
        try:
            self.assertEqual(app.mode, "camera")
            self.assertEqual(app.source.stream, "depth")               # only Blobs -> depth
            self.assertTrue(app.step())
            self.assertEqual(app.counts[3], 2)                          # the two objects of the fake depth
        finally:
            app.close()
        self.assertTrue(sys.modules["pylibfreenect2"].devices[0].closed)


class WindowCloseTest(unittest.TestCase):
    def test_window_closed_on_hidden_or_destroyed_window(self):
        import cv2
        with mock.patch.object(cv2, "getWindowProperty", side_effect=cv2.error("NULL window")):
            self.assertTrue(App.window_closed())                 # GTK/Win32 destroy the window on [x]
        with mock.patch.object(cv2, "getWindowProperty", return_value=0.0):
            self.assertTrue(App.window_closed())
        with mock.patch.object(cv2, "getWindowProperty", return_value=1.0):
            self.assertFalse(App.window_closed())


if __name__ == "__main__":
    unittest.main()
