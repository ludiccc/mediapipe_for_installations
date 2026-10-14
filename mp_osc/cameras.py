"""Enumeration of the cameras (webcams and Kinects) attached to the computer.

OpenCV can open a camera by index but cannot tell you how many there are or
what they are called, so this module looks around by itself:

* Linux: the ``/dev/video*`` nodes, with the names from sysfs. The kernel is
  asked (``VIDIOC_QUERYCAP``) which nodes are real video capture devices - a
  webcam usually also creates a *metadata* node that is not one - and only
  those are opened once to check that they are free. A camera that was just
  plugged in can refuse the first open while its driver settles, so capture
  nodes are retried a few times.
* macOS / Windows: indices 0, 1, 2 ... are probed until two in a row fail
  (indices are contiguous on those platforms). Names come from
  ``system_profiler`` (macOS) or ``pygrabber`` (Windows, optional) when
  available, otherwise "Camera <n>".
* Kinects: Kinect v2 sensors are enumerated through pylibfreenect2 (by serial
  number, without opening them); a Kinect v1 through libfreenect.

Probing means opening every webcam for a moment, which can take a second per
device, so the app runs :func:`list_cameras` in a background thread
(:class:`CameraScanner`) and offers a *Rescan* button.

    >>> for cam in list_cameras():
    ...     print(cam.spec, cam.name)          # "0  Integrated Camera", "kinect  Kinect v2 012345"

``cam.spec`` is the string ``open_source`` understands (see ``capture.py``);
several specs joined with ``+`` make a side-by-side composition.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import cv2


@dataclass(frozen=True)
class CameraInfo:
    index: int                 # webcam: what cv2.VideoCapture takes; kinect: device number
    name: str                  # human readable, best effort
    device: str = ""           # /dev/videoN on Linux, "" elsewhere
    in_use: bool = False       # currently opened by this app (not probed)
    kind: str = "webcam"       # "webcam" or "kinect"
    spec: str = ""             # source spec for open_source(); filled in automatically

    def __post_init__(self):
        if not self.spec:
            spec = str(self.index) if self.kind == "webcam" else ("kinect" if self.index == 0 else f"kinect:{self.index}")
            object.__setattr__(self, "spec", spec)

    @property
    def label(self) -> str:
        return f"{self.spec}  {self.name}"


# Opening a camera that another program is streaming from, or probing an index
# that does not exist, makes some backends print to stderr. Keep it quiet.
_QUIET = {"OPENCV_LOG_LEVEL": "ERROR", "OPENCV_VIDEOIO_DEBUG": "0"}


def _probe(index: int, backend: int | None = None) -> bool:
    """True if ``cv2.VideoCapture(index)`` opens (the camera is there and free)."""
    old = {k: os.environ.get(k) for k in _QUIET}
    os.environ.update(_QUIET)
    try:
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        try:
            return bool(cap.isOpened())
        finally:
            cap.release()
    except cv2.error:
        return False
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------ Linux
# struct v4l2_capability: driver[16] card[32] bus_info[32] version caps device_caps reserved[3]
_VIDIOC_QUERYCAP = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_META_CAPTURE = 0x00800000
_V4L2_CAP_DEVICE_CAPS = 0x80000000


def _v4l2_is_capture(device: str) -> bool | None:
    """True/False if the kernel says the node can/cannot capture video; None if it cannot be asked."""
    try:
        import fcntl
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except (OSError, ImportError):
        return None
    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
        version, caps, device_caps = struct.unpack_from("<III", buf, 80)
        effective = device_caps if caps & _V4L2_CAP_DEVICE_CAPS else caps
        if effective & _V4L2_CAP_META_CAPTURE and not effective & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE):
            return False
        return bool(effective & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE))
    except OSError:
        return None
    finally:
        os.close(fd)


def _sysfs_name(device: str) -> str:
    node = os.path.basename(device)
    try:
        with open(f"/sys/class/video4linux/{node}/name", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _linux_cameras(skip: Mapping[str, str], probe: Callable[[int], bool],
                   is_capture: Callable[[str], bool | None] = _v4l2_is_capture,
                   retries: int = 3, retry_delay: float = 0.4) -> list[CameraInfo]:
    out = []
    for device in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        m = re.search(r"(\d+)$", device)
        if not m:
            continue
        index = int(m.group(1))
        name = _sysfs_name(device) or f"Camera {index}"
        if str(index) in skip:
            out.append(CameraInfo(index, name, device, in_use=True))
            continue
        capture = is_capture(device)
        if capture is False:
            continue                       # a metadata node: not a camera
        # A real capture node (or unknown): check that it opens. A camera that
        # was plugged in a moment ago may need a second or two.
        attempts = retries if capture else 1
        for attempt in range(attempts):
            if probe(index):
                out.append(CameraInfo(index, name, device))
                break
            if attempt + 1 < attempts:
                time.sleep(retry_delay)
    return out


# ------------------------------------------------------------------ macOS
def _macos_names(timeout: float = 8.0) -> list[str]:
    try:
        res = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=timeout)
        data = json.loads(res.stdout or "{}")
        return [str(c.get("_name", "")) for c in data.get("SPCameraDataType", [])]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


# ------------------------------------------------------------------ Windows
def _windows_names() -> list[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        return [str(n) for n in FilterGraph().get_input_devices()]
    except Exception:
        return []


def _probe_contiguous(skip: Mapping[str, str], probe: Callable[[int], bool], names: list[str],
                      max_index: int = 10, max_misses: int = 2) -> list[CameraInfo]:
    out = []
    misses = 0
    for index in range(max_index + 1):
        name = names[index] if index < len(names) else f"Camera {index}"
        if str(index) in skip:
            out.append(CameraInfo(index, name, in_use=True))
            misses = 0
        elif probe(index):
            out.append(CameraInfo(index, name))
            misses = 0
        else:
            misses += 1
            if misses >= max_misses and index >= len(names):
                break
    return out


# ------------------------------------------------------------------ Kinects
def quiet_libfreenect2(k) -> None:
    """Keep libfreenect2's [Info] chatter off the terminal (errors are still shown)."""
    try:
        level = getattr(k, "LoggerLevel", None)
        if level is not None and hasattr(k, "createConsoleLogger") and hasattr(k, "setGlobalLogger"):
            k.setGlobalLogger(k.createConsoleLogger(level.Error))
    except Exception:
        pass


def list_kinects(skip: Mapping[str, str] = {}) -> list[CameraInfo]:
    """Kinect sensors: v2 by serial through pylibfreenect2, v1 through libfreenect.

    Enumerating a Kinect v2 does not open it, so it is safe while the app uses
    one. A Kinect v1 is not probed while a Kinect is in use (libfreenect does
    not like two contexts); the one in use is listed from ``skip``.
    """
    out: list[CameraInfo] = []
    kinect_in_use = {spec: name for spec, name in skip.items() if spec.split(":")[0] == "kinect"}
    try:
        import pylibfreenect2 as k  # type: ignore
        quiet_libfreenect2(k)
        fn = k.Freenect2()
        for i in range(int(fn.enumerateDevices())):
            serial = fn.getDeviceSerialNumber(i)
            info = CameraInfo(i, f"Kinect v2 {serial}", kind="kinect")
            out.append(CameraInfo(i, info.name, kind="kinect", in_use=info.spec in kinect_in_use))
    except Exception:
        pass
    if not out and not kinect_in_use:
        try:
            import freenect  # type: ignore
            ctx = freenect.init()
            try:
                n = int(freenect.num_devices(ctx))
            finally:
                freenect.shutdown(ctx)
            for i in range(n):
                out.append(CameraInfo(i, "Kinect v1", kind="kinect"))
        except Exception:
            pass
    for spec, name in kinect_in_use.items():
        if not any(c.spec == spec for c in out):
            out.append(CameraInfo(0, name or "Kinect", kind="kinect", in_use=True, spec=spec))
    return out


# ------------------------------------------------------------------ public
def list_cameras(skip: Iterable[str] | Mapping[str, str] = (), max_index: int = 10,
                 probe: Callable[[int], bool] = _probe, platform: str | None = None,
                 names: list[str] | None = None, kinects: bool = True, webcams: bool = True,
                 **linux_options) -> list[CameraInfo]:
    """Cameras that can be opened right now: webcams sorted by index, then Kinects.

    ``skip``: source specs not to probe (the cameras this app already has
    open; they are listed as ``in_use``), optionally mapped to a display
    name. ``webcams`` / ``kinects`` switch a kind off (``--no-webcams`` /
    ``--no-kinects``). ``probe`` / ``platform`` / ``names`` exist for tests.
    """
    platform = platform or sys.platform
    skip = dict(skip) if isinstance(skip, Mapping) else {str(s): "" for s in skip}
    found: list[CameraInfo] = []
    if webcams:
        if platform.startswith("linux"):
            found += _linux_cameras(skip, probe, **linux_options)
        else:
            if names is None:
                names = _macos_names() if platform == "darwin" else _windows_names() if platform.startswith("win") else []
            found += _probe_contiguous(skip, probe, names, max_index=max_index)
    if kinects:
        found += list_kinects(skip)
    return found


def format_camera_list(cameras: list[CameraInfo]) -> str:
    if not cameras:
        return "no cameras found"
    width = max(len(c.spec) for c in cameras)
    lines = []
    for c in cameras:
        extra = f"  ({c.device})" if c.device else ""
        extra += "  [in use]" if c.in_use else ""
        lines.append(f"  {c.spec:<{width}}  {c.name}{extra}")
    return "\n".join(lines)


class CameraScanner:
    """Runs ``list_cameras`` in a background thread; the result is read later.

        scanner = CameraScanner(skip=lambda: {"0": "webcam 0"})
        scanner.start()
        ...
        if scanner.done: print(scanner.cameras)

    ``lock`` is held while probing: the app takes it before opening a camera
    so that a probe and a real open never collide on the same device.
    """

    def __init__(self, skip: Callable[[], Mapping[str, str] | Iterable[str]] = lambda: {},
                 on_done: Callable[[], None] | None = None,
                 lister: Callable[..., list[CameraInfo]] = list_cameras):
        self._skip = skip
        self._on_done = on_done
        self._lister = lister
        self._thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.cameras: list[CameraInfo] = []
        self.done = False
        self.error = ""

    @property
    def scanning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.scanning:
            return False
        self.done = False
        self._thread = threading.Thread(target=self._run, name="camera-scan", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            with self.lock:
                skip = self._skip()
                cams = self._lister(skip=dict(skip) if isinstance(skip, Mapping) else list(skip))
            self.cameras = cams
            self.error = ""
        except Exception as exc:          # a broken backend must not kill the app
            self.error = str(exc)
        finally:
            self.done = True
            if self._on_done:
                self._on_done()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def by_spec(self, spec: str) -> CameraInfo | None:
        return next((c for c in self.cameras if c.spec == spec), None)
