"""Frame sources: webcams, Kinects, several cameras side by side, video files, images.

Everything the app analyzes comes from a :class:`FrameSource`. Whatever the
hardware, a source is used the same way::

    source = open_source("0")            # webcam 0 - or "kinect", "0+2", "clip.mp4", "photo.jpg"
    ok, frame = source.read()            # frame: numpy array, height x width x 3, BGR (OpenCV order)
    t = source.timestamp()               # seconds, of the frame just read
    depth = source.depth()               # metres per pixel (Kinect) or None
    source.release()

The *source specification* strings understood by :func:`open_source`:

==================  =====================================================
``"0"``, ``"2"``    webcam by index (what ``cv2.VideoCapture`` takes)
``"kinect"``        the first Kinect (v2 via pylibfreenect2, v1 via
                    libfreenect, or any OpenNI device)
``"kinect:1"``      the second Kinect v2; ``"kinect:freenect"`` forces a
                    backend; ``"kinect:freenect2:1"`` both
``"0+kinect"``      several sources side by side, left to right
                    (:class:`CompositeSource`)
``"clip.mp4"``      a video file (its own frame rate is the clock)
``"photo.jpg"``     a still image, repeated forever (handy for tests)
==================  =====================================================

Every source has:

* ``kind`` - ``"camera"``, ``"kinect"``, ``"video"`` or ``"image"``; the
  control panel's *Source* radio group shows it.
* ``name`` - human readable, for the status area.
* ``has_depth`` / ``want_depth`` / ``depth()`` - Kinects can hand out a depth
  map *aligned with the colour image* (same size, metres, 0 = unknown). It is
  only computed when the app sets ``want_depth`` (the *Z from Kinect* toggle),
  because the alignment costs a few milliseconds per frame.

Timestamps of live sources come from the monotonic clock; video files use
their own frame rate, so velocities are right even when the file is processed
faster or slower than real time.

The webcam source grabs frames in a background thread and always hands out
the most recent one, so analysis latency does not pile up when MediaPipe is
slower than the camera.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================================
# the common interface
# ============================================================================
class FrameSource:
    """Base class: one ``read()`` per analyzed frame.

    Subclasses implement ``read()`` (and ``release()``); the rest is shared.
    """

    name = "source"
    kind = "video"
    live = True          # True: timestamps follow the wall clock (cameras)
    has_depth = False    # True when ``depth()`` can return something (Kinects)

    def __init__(self):
        self._t0 = time.monotonic()
        self.want_depth = False   # set by the app: compute the aligned depth map on every read

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Next frame as ``(ok, bgr_image)``; ``(False, None)`` when the source is finished / broken."""
        raise NotImplementedError

    def depth(self) -> np.ndarray | None:
        """Depth of the last frame: float32 metres per pixel, same height/width, 0 = unknown. None if unavailable."""
        return None

    def timestamp(self) -> float:
        """Time in seconds of the frame returned by the last ``read()``."""
        return time.monotonic() - self._t0

    def release(self) -> None:
        """Free the camera / file. Safe to call twice."""

    @property
    def fps(self) -> float:
        return 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# ============================================================================
# files
# ============================================================================
class ImageSource(FrameSource):
    """Repeats a still image forever (useful for testing without a camera)."""

    kind = "image"

    def __init__(self, path: str, width: int = 0, height: int = 0):
        super().__init__()
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"could not read image {path!r}")
        if width and height:
            image = cv2.resize(image, (width, height))
        self._image = image
        self.name = os.path.basename(path)

    def read(self):
        return True, self._image.copy()


class VideoFileSource(FrameSource):
    """Reads a video file sequentially, optionally looping at the end."""

    kind = "video"
    live = False

    def __init__(self, path: str, loop: bool = True):
        super().__init__()
        if not os.path.exists(path):
            raise RuntimeError(f"video file not found: {path!r}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video {path!r}")
        self._loop = loop
        self.name = os.path.basename(path)
        self._frames = 0
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= self._fps <= 240.0:
            self._fps = 30.0

    @property
    def fps(self) -> float:
        return self._fps

    def read(self):
        ok, frame = self._cap.read()
        if not ok and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if ok:
            self._frames += 1
        return ok, frame

    def timestamp(self) -> float:
        # Frame count keeps growing across loops, so timestamps never go back.
        return (self._frames - 1) / self._fps

    def release(self):
        self._cap.release()


# ============================================================================
# webcam
# ============================================================================
class WebcamSource(FrameSource):
    """Threaded webcam reader that always returns the latest frame.

    Why a thread: ``cv2.VideoCapture.read()`` returns the *next buffered*
    frame. If MediaPipe is slower than the camera, the buffer fills and the
    app falls behind reality (latency grows, nothing is "live" any more). So
    a background thread keeps reading and stores only the newest frame;
    ``read()`` blocks until a frame newer than the previously returned one
    is available, so each frame is analyzed at most once and never queued.
    """

    kind = "camera"

    def __init__(self, index: int, width: int = 640, height: int = 480, timeout: float = 5.0,
                 name: str = ""):
        super().__init__()
        self.index = index
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open webcam {index}. Try another index with --source 1, "
                "check that no other program is using the camera, or pass a video file."
            )
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.name = f"webcam {index}" + (f" ({name})" if name else "")
        self._timeout = timeout
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0          # number of frames grabbed so far
        self._last_seq = 0     # sequence of the last frame handed out
        self._running = True
        self._failed = False
        self._thread = threading.Thread(target=self._loop, name="webcam", daemon=True)
        self._thread.start()

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def _loop(self) -> None:
        """Background thread: grab frames as fast as the camera delivers them."""
        failures = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures > 30:          # unplugged / broken: give up, read() reports failure
                    with self._cond:
                        self._failed = True
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            failures = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self):
        with self._cond:
            deadline = time.monotonic() + self._timeout
            while self._seq == self._last_seq and not self._failed and self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._cond.wait(remaining)
            if self._failed or not self._running or self._frame is None:
                return False, None
            self._last_seq = self._seq
            return True, self._frame

    def release(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._cap.release()


# ============================================================================
# Kinect
# ============================================================================
KINECT_STREAMS = ("auto", "color", "depth")
KINECT_BACKENDS = ("auto", "freenect2", "freenect", "openni")
KINECT_PIPELINES = ("auto", "opengl", "opencl", "cuda", "cpu")


def depth_to_image(depth_m: np.ndarray, max_m: float) -> np.ndarray:
    """Depth in metres (0 = unknown) -> BGR grey image: near = bright, far / unknown = black.

    This is what the ``depth`` stream shows and what the Blobs detector sees
    with a ceiling-mounted Kinect: a person is a bright shape on a dark floor.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = depth_m > 0
    grey = np.zeros(depth_m.shape, dtype=np.uint8)
    scaled = 255.0 * (1.0 - np.clip(depth_m / max_m, 0.0, 1.0))
    grey[valid] = scaled[valid].astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def clean_depth_m(depth: np.ndarray, scale_to_m: float, max_m: float = 10.0) -> np.ndarray:
    """Raw depth array -> float32 metres with 0 for unknown (NaN, inf, negative, too far)."""
    d = np.asarray(depth, dtype=np.float32) * np.float32(scale_to_m)
    bad = ~np.isfinite(d) | (d <= 0) | (d > max_m)
    d[bad] = 0.0
    return d


def _preload_libfreenect2() -> str:
    """Load the libfreenect2 shared library by hand when asked to.

    Some pylibfreenect2 builds (Windows, Homebrew, a library installed in a
    non-standard folder) only import when libfreenect2 is already loaded in
    the process. Set ``LIBFREENECT2_LIBRARY`` to the full path of
    ``freenect2.dll`` / ``libfreenect2.dylib`` / ``libfreenect2.so``.
    Returns an error text ("" when fine or not requested).
    """
    path = os.environ.get("LIBFREENECT2_LIBRARY", "")
    if not path:
        return ""
    import ctypes
    try:
        ctypes.CDLL(path)
        return ""
    except OSError as exc:
        return f"LIBFREENECT2_LIBRARY={path!r} could not be loaded: {exc}"


def parse_kinect_spec(spec: str) -> tuple[str, int]:
    """``"kinect"`` / ``"kinect:1"`` / ``"kinect:freenect"`` / ``"kinect:freenect2:1"`` -> (backend, index)."""
    parts = spec.strip().lower().split(":")
    if parts[0] != "kinect":
        raise ValueError(f"not a kinect spec: {spec!r}")
    backend, index = "auto", 0
    for part in parts[1:]:
        if not part:
            continue
        if part.isdigit():
            index = int(part)
        elif part in KINECT_BACKENDS:
            backend = part
        else:
            raise ValueError(f"bad kinect spec {spec!r}: use kinect, kinect:<n>, kinect:<backend>[:<n>]")
    return backend, index


def is_kinect_spec(spec: str) -> bool:
    return str(spec).strip().lower().split(":")[0] == "kinect"


class _Freenect2Backend:
    """Kinect v2 through pylibfreenect2 (libfreenect2).

    Colour frames are 1920x1080 BGRX, depth frames 512x424 float32 millimetres
    (0 = unknown, usable up to ~4.5 m). The two cameras of the sensor do not
    see the same pixels, so to know the distance of a *colour* pixel the depth
    must be re-projected into the colour image: libfreenect2's ``Registration``
    does that and fills ``bigdepth``, a 1920x1082 depth image aligned with the
    colour frame (one padding row above and below).
    """

    name = "freenect2"
    MAX_DEPTH_M = 4.5
    PIPELINE_CLASSES = {"opengl": "OpenGLPacketPipeline", "opencl": "OpenCLPacketPipeline",
                        "cuda": "CudaPacketPipeline", "cpu": "CpuPacketPipeline"}
    AUTO_ORDER = ("opengl", "opencl", "cuda", "cpu")    # the order of pylibfreenect2's own example

    def __init__(self, stream: str, pipeline: str = "auto", device_index: int = 0, timeout: float = 5.0):
        preload_error = _preload_libfreenect2()
        try:
            import pylibfreenect2 as k  # type: ignore
        except ImportError as exc:
            raise RuntimeError("python module 'pylibfreenect2' (Kinect v2) is not installed"
                               + (f"; {preload_error}" if preload_error else f" ({exc})")) from exc
        from .cameras import quiet_libfreenect2
        quiet_libfreenect2(k)
        self.k = k
        self.stream = stream
        self.timeout = timeout
        self.pipeline_name, pipe = self._select_pipeline(k, pipeline)
        self.fn = k.Freenect2()
        num = self.fn.enumerateDevices()
        if num == 0:
            raise RuntimeError("pylibfreenect2 found no Kinect v2 (check the USB 3 connection and the "
                               "libfreenect2 udev rules / driver)")
        if device_index >= num:
            raise RuntimeError(f"Kinect v2 #{device_index} not found ({num} device(s))")
        self.serial = self.fn.getDeviceSerialNumber(device_index)
        self.device = self.fn.openDevice(self.serial, pipeline=pipe) if pipe is not None else \
            self.fn.openDevice(self.serial)
        if self.device is None:
            raise RuntimeError(f"Kinect v2 {self.serial!r} could not be opened (in use by another program?)")
        # Colour mode also subscribes to depth so that the aligned depth map
        # can be produced when the app asks for it (want_depth).
        types = (k.FrameType.Color | k.FrameType.Depth) if stream == "color" else k.FrameType.Depth
        self.listener = k.SyncMultiFrameListener(types)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()
        self._registration = None
        self.want_depth = False       # set by KinectSource before every read
        self.name = f"freenect2 {self.serial} ({self.pipeline_name} pipeline)"

    @classmethod
    def _select_pipeline(cls, k, wanted: str):
        """(name, pipeline object) - None when the library should choose."""
        order = cls.AUTO_ORDER if wanted == "auto" else (wanted,)
        errors = []
        for name in order:
            cls_name = cls.PIPELINE_CLASSES[name]
            factory = getattr(k, cls_name, None)
            if factory is None:
                errors.append(f"{name}: not in this pylibfreenect2 build")
                continue
            try:
                return name, factory()
            except Exception as exc:      # a pipeline that cannot initialise (no GPU/OpenCL...)
                errors.append(f"{name}: {exc}")
        if wanted != "auto":
            raise RuntimeError("Kinect v2 pipeline " + "; ".join(errors))
        return "default", None

    def _wait(self) -> bool:
        """Poll ``hasNewFrame`` with a timeout so a dead sensor cannot hang the app."""
        deadline = time.monotonic() + self.timeout
        has = getattr(self.listener, "hasNewFrame", None)
        if has is None:
            return True
        while not has():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.002)
        return True

    @staticmethod
    def frame_as_array(frame, dtype=None) -> np.ndarray:
        """pylibfreenect2 frames -> numpy copy.

        Frames coming from the device know their type (``asarray()`` works,
        colour wants ``asarray(np.uint8)``); frames created from Python - the
        ``bigdepth`` buffer - do not, and ``asarray()`` raises "Cannot
        determine type of data": they need the dtype explicitly.
        """
        if dtype is not None:
            try:
                return np.array(frame.asarray(dtype), copy=True)
            except (TypeError, ValueError):
                pass
        try:
            return np.array(frame.asarray(), copy=True)
        except (TypeError, ValueError):
            return np.array(frame.asarray(np.float32), copy=True)

    def _aligned_depth_m(self, frames) -> np.ndarray:
        """Depth re-projected onto the colour image (1080x1920, metres) via Registration + bigdepth."""
        k = self.k
        if self._registration is None:
            self._registration = k.Registration(self.device.getIrCameraParams(),
                                                self.device.getColorCameraParams())
            self._undistorted = k.Frame(512, 424, 4)
            self._registered = k.Frame(512, 424, 4)
            self._bigdepth = k.Frame(1920, 1082, 4)
        self._registration.apply(frames["color"], frames["depth"], self._undistorted, self._registered,
                                 bigdepth=self._bigdepth)
        big = self.frame_as_array(self._bigdepth, np.float32)   # (1082, 1920) mm, inf where unknown
        if big.shape[0] == 1082:
            big = big[1:-1, :]
        return clean_depth_m(big, 0.001, self.MAX_DEPTH_M)

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """-> (ok, bgr image, depth in metres or None)."""
        if not self._wait():
            return False, None, None
        frames = self.listener.waitForNewFrame()
        try:
            if self.stream == "color":
                color = self.frame_as_array(frames["color"], np.uint8)
                if color.ndim == 3 and color.shape[2] == 4:
                    color = np.ascontiguousarray(color[:, :, :3])   # BGRX -> BGR
                elif color.ndim == 2:
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
                depth = self._aligned_depth_m(frames) if self.want_depth else None
                return True, color, depth
            depth = clean_depth_m(self.frame_as_array(frames["depth"], np.float32), 0.001, self.MAX_DEPTH_M)
            return True, depth_to_image(depth, self.MAX_DEPTH_M), depth
        finally:
            self.listener.release(frames)

    def close(self):
        try:
            self.device.stop()
        finally:
            self.device.close()


class _FreenectBackend:
    """Kinect v1 through the python bindings of libfreenect (``freenect`` module).

    ``DEPTH_REGISTERED`` asks the driver for depth in millimetres aligned with
    the RGB image; older bindings only have the raw 11-bit values, which are
    converted with the usual approximation.
    """

    name = "freenect"
    MAX_DEPTH_M = 4.5

    def __init__(self, stream: str):
        try:
            import freenect  # type: ignore
        except ImportError as exc:
            raise RuntimeError("python module 'freenect' (Kinect v1, libfreenect) is not installed") from exc
        self.freenect = freenect
        self.stream = stream
        self.want_depth = False
        if freenect.sync_get_video() is None:
            raise RuntimeError("libfreenect found no Kinect v1")

    def _depth_m(self) -> np.ndarray | None:
        fn = self.freenect
        fmt = getattr(fn, "DEPTH_REGISTERED", None)
        if fmt is not None:
            result = fn.sync_get_depth(format=fmt)
            if result is None:
                return None
            return clean_depth_m(result[0], 0.001, self.MAX_DEPTH_M)
        result = fn.sync_get_depth()
        if result is None:
            return None
        raw = np.asarray(result[0], dtype=np.float32)
        metres = 1.0 / (raw * -0.0030711016 + 3.3309495161)      # standard raw -> m approximation
        metres[raw >= 2047] = 0.0
        return clean_depth_m(metres, 1.0, self.MAX_DEPTH_M)

    def read(self):
        if self.stream == "color":
            result = self.freenect.sync_get_video()
            if result is None:
                return False, None, None
            color = cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)
            return True, color, self._depth_m() if self.want_depth else None
        depth = self._depth_m()
        if depth is None:
            return False, None, None
        return True, depth_to_image(depth, self.MAX_DEPTH_M), depth

    def close(self):
        try:
            self.freenect.sync_stop()
        except Exception:
            pass


class _OpenNIBackend:
    """Any OpenNI device through OpenCV's OpenNI2 / OpenNI video backend."""

    name = "openni"
    MAX_DEPTH_M = 4.0

    def __init__(self, stream: str):
        self.stream = stream
        self.want_depth = False
        self._cap = None
        errors = []
        for backend, label in ((getattr(cv2, "CAP_OPENNI2", None), "OpenNI2"),
                               (getattr(cv2, "CAP_OPENNI", None), "OpenNI")):
            if backend is None:
                continue
            cap = cv2.VideoCapture(backend)
            if cap.isOpened():
                self._cap = cap
                self.name = f"openni ({label})"
                if hasattr(cv2, "CAP_PROP_OPENNI_REGISTRATION"):
                    cap.set(cv2.CAP_PROP_OPENNI_REGISTRATION, 1)    # depth aligned with the colour image
                return
            cap.release()
            errors.append(f"OpenCV {label} backend: no device")
        raise RuntimeError("; ".join(errors) or "OpenCV was built without OpenNI")

    def _depth_m(self) -> np.ndarray | None:
        ok, depth = self._cap.retrieve(None, cv2.CAP_OPENNI_DEPTH_MAP)     # uint16 millimetres
        if not ok or depth is None:
            return None
        return clean_depth_m(depth, 0.001, self.MAX_DEPTH_M)

    def read(self):
        cap = self._cap
        if not cap.grab():
            return False, None, None
        if self.stream == "color":
            ok, frame = cap.retrieve(None, cv2.CAP_OPENNI_BGR_IMAGE)
            if not ok or frame is None:
                return False, None, None
            return True, frame, self._depth_m() if self.want_depth else None
        depth = self._depth_m()
        if depth is None:
            return False, None, None
        return True, depth_to_image(depth, self.MAX_DEPTH_M), depth

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


KINECT_BACKEND_CLASSES = {"freenect2": _Freenect2Backend, "freenect": _FreenectBackend, "openni": _OpenNIBackend}


class KinectSource(FrameSource):
    """A Kinect (v2 or v1) or any OpenNI depth sensor.

    ``stream="color"`` hands out the colour camera (for MediaPipe); ``"depth"``
    hands out the depth map as an 8-bit grey image (near = bright, far/unknown
    = black), which is the ideal input for the Blobs detector with a
    ceiling-mounted sensor - the same signal the original Processing sketch of
    ``Sonido_Cuadrafonico`` used.

    In both modes ``depth()`` returns the distance of every pixel of the frame
    in metres once ``want_depth`` is set (the *Z from Kinect* toggle).

    Backends (``backend="auto"`` tries them in this order):

    * ``freenect2`` - Kinect **v2** through ``pylibfreenect2`` (libfreenect2).
      ``pipeline`` picks the depth processing: opengl / opencl / cuda / cpu or
      auto (first one that initialises).
    * ``freenect`` - Kinect **v1** through the python bindings of libfreenect.
    * ``openni`` - any OpenNI device, when OpenCV was built with OpenNI2.

    ``width`` scales the frames (and the depth map) to that width, aspect
    kept; 0 = native (1920x1080 colour / 512x424 depth on a Kinect v2).
    """

    kind = "kinect"
    has_depth = True

    def __init__(self, stream: str = "color", width: int = 0, height: int = 0,
                 backend: str = "auto", pipeline: str = "auto", device_index: int = 0):
        super().__init__()
        if stream not in ("color", "depth"):
            raise ValueError(f"kinect stream must be 'color' or 'depth', not {stream!r}")
        if backend not in KINECT_BACKENDS:
            raise ValueError(f"kinect backend must be one of {KINECT_BACKENDS}, not {backend!r}")
        self.stream = stream
        self.width = int(width or 0)
        self.index = device_index
        self._backend = None
        self._depth: np.ndarray | None = None
        self.depth_error = ""        # set when the aligned depth cannot be produced
        errors = []
        for name in (KINECT_BACKENDS[1:] if backend == "auto" else (backend,)):
            cls = KINECT_BACKEND_CLASSES[name]
            try:
                self._backend = cls(stream, pipeline, device_index) if name == "freenect2" else cls(stream)
                self.backend = name
                break
            except RuntimeError as exc:
                errors.append(str(exc))
        if self._backend is None:
            raise RuntimeError("no Kinect found (" + "; ".join(errors) + "). Kinect v2: pip install "
                               "pylibfreenect2 (needs libfreenect2); Kinect v1: the 'freenect' module of "
                               "libfreenect; or an OpenCV build with OpenNI2. Plug the sensor in first.")
        self.name = f"kinect ({stream}) via {self._backend.name}"

    def read(self):
        self._backend.want_depth = bool(self.want_depth) and not self.depth_error
        try:
            ok, frame, depth = self._backend.read()
        except Exception as exc:
            if self._backend.want_depth:
                # The colour stream must survive a depth-alignment problem:
                # report once, carry on without depth.
                self.depth_error = f"{type(exc).__name__}: {exc}"
                print(f"\nkinect: depth alignment failed ({self.depth_error}); z stays MediaPipe's",
                      file=sys.stderr)
                self._backend.want_depth = False
                ok, frame, depth = self._backend.read()
            else:
                raise
        if not ok or frame is None:
            self._depth = None
            return False, None
        if self.width and frame.shape[1] != self.width:
            h = max(1, round(frame.shape[0] * self.width / frame.shape[1]))
            frame = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_AREA)
            if depth is not None:
                depth = cv2.resize(depth, (self.width, h), interpolation=cv2.INTER_NEAREST)
        self._depth = depth
        return True, frame

    def depth(self):
        return self._depth

    @staticmethod
    def depth_to_image(depth_m: np.ndarray, max_m: float) -> np.ndarray:
        return depth_to_image(depth_m, max_m)

    def release(self):
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None


# ============================================================================
# several cameras side by side
# ============================================================================
class CompositeSource(FrameSource):
    """Several sources shown next to each other in one wide frame.

    Every part is scaled to the height of the first one (aspect kept) and the
    parts are stacked left to right in ``parts`` order. MediaPipe then
    analyzes that wide image as if it were one camera, so coordinates are
    normalized to the *whole* composition (a person in the right-hand camera
    has x > 0.5 with two equal cameras).

    Depth: available only when **every** part has depth (Kinect + Kinect);
    with a webcam in the mix ``has_depth`` is False and the z stays
    MediaPipe's. ``set_order`` re-arranges the parts without re-opening them.
    """

    kind = "camera"

    def __init__(self, parts: list[FrameSource]):
        super().__init__()
        if not parts:
            raise ValueError("a composite source needs at least one part")
        self.parts = list(parts)
        self._depth = None
        self.slots: list[tuple[str, int, int]] = []    # (spec, x0, x1) of every part in the last frame
        self._update_name()

    def _update_name(self) -> None:
        self.name = " + ".join(p.name for p in self.parts)
        self.has_depth = all(p.has_depth for p in self.parts)

    @property
    def specs(self) -> list[str]:
        return [getattr(p, "spec", "") for p in self.parts]

    def set_order(self, specs: list[str]) -> bool:
        """Re-order the parts to match ``specs`` (left to right). False if they do not match."""
        by_spec = {getattr(p, "spec", ""): p for p in self.parts}
        if sorted(by_spec) != sorted(specs):
            return False
        self.parts = [by_spec[s] for s in specs]
        self._update_name()
        return True

    def read(self):
        frames, depths = [], []
        for part in self.parts:
            part.want_depth = self.want_depth and self.has_depth
            ok, frame = part.read()
            if not ok or frame is None:
                self._depth = None
                return False, None
            frames.append(frame)
            depths.append(part.depth())
        height = frames[0].shape[0]
        scaled, scaled_depths = [], []
        for frame, depth in zip(frames, depths):
            if frame.shape[0] != height:
                w = max(1, round(frame.shape[1] * height / frame.shape[0]))
                frame = cv2.resize(frame, (w, height), interpolation=cv2.INTER_AREA)
                if depth is not None:
                    depth = cv2.resize(depth, (w, height), interpolation=cv2.INTER_NEAREST)
            scaled.append(frame)
            scaled_depths.append(depth)
        if self.want_depth and self.has_depth and all(d is not None for d in scaled_depths):
            self._depth = np.hstack(scaled_depths)
        else:
            self._depth = None
        slots, x = [], 0
        for part, frame in zip(self.parts, scaled):
            slots.append((getattr(part, "spec", ""), x, x + frame.shape[1]))
            x += frame.shape[1]
        self.slots = slots
        return True, np.hstack(scaled)

    def depth(self):
        return self._depth

    def release(self):
        for part in self.parts:
            try:
                part.release()
            except Exception:
                pass


# ============================================================================
# the factory
# ============================================================================
def open_source(source: str, width: int = 640, height: int = 480, loop: bool = True,
                kinect_stream: str = "color", camera_name: str = "", kinect_backend: str = "auto",
                kinect_pipeline: str = "auto", camera_names: dict | None = None) -> FrameSource:
    """Open the source described by ``source`` (see the module docstring for the spec strings).

    ``camera_name`` / ``camera_names`` only decorate the ``name`` of webcams
    (the panel shows "webcam 2 (Logitech BRIO)").
    """
    source = str(source).strip()
    if "+" in source:
        specs = [s.strip() for s in source.split("+") if s.strip()]
        if len(specs) > 1:
            parts: list[FrameSource] = []
            try:
                for spec in specs:
                    part = open_source(spec, width, height, loop, kinect_stream, "", kinect_backend,
                                       kinect_pipeline, camera_names)
                    part.spec = spec  # type: ignore[attr-defined]
                    parts.append(part)
            except Exception:
                for part in parts:
                    part.release()
                raise
            comp = CompositeSource(parts)
            comp.spec = "+".join(specs)  # type: ignore[attr-defined]
            return comp
        source = specs[0] if specs else source
    if is_kinect_spec(source):
        backend, index = parse_kinect_spec(source)
        if backend == "auto":
            backend = kinect_backend
        src = KinectSource("depth" if kinect_stream == "depth" else "color", width, height,
                           backend=backend, pipeline=kinect_pipeline, device_index=index)
    elif source.lstrip("-").isdigit():
        index = int(source)
        name = camera_name or (camera_names or {}).get(index, "")
        src = WebcamSource(index, width, height, name=name)
    else:
        ext = os.path.splitext(source)[1].lower()
        src = ImageSource(source) if ext in IMAGE_EXTENSIONS else VideoFileSource(source, loop=loop)
    src.spec = source  # type: ignore[attr-defined]
    return src
