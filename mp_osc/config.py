"""All the options of the app, in one place.

:class:`Config` is a dataclass: one attribute per option with its default,
grouped like the ``--help`` output (video input, detectors, OSC output,
tracking, UI). The whole app reads ``cfg.<option>``; the control panel
*writes* some of them at run time (``cfg.pose``, ``cfg.blob_threshold`` ...).
``__post_init__`` validates values and applies the "default to --pose when
nothing is selected" rule.

:func:`build_parser` describes the same options for ``argparse`` (every
``--flag`` maps to an attribute of the same name, ``-`` becoming ``_``) and
:func:`parse_args` turns the command line into a ``Config``.

To add an option: add the attribute here, add the ``--flag`` to the parser
in the matching group, and read ``cfg.<option>`` where it is needed.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.dirname(HERE)  # the repository root holds the .task files

DEFAULT_TEMPLATES_DIR = os.path.join(DEFAULT_MODELS_DIR, "templates")

FORMATS = ("mp", "legacy", "kinect")
ID_STYLES = ("arg", "address")
KINECT_Y_MODES = ("image", "size")
KINECT_FROM = ("auto", "pose", "blobs")
KINECT_STREAMS = ("auto", "color", "depth")
KINECT_BACKENDS = ("auto", "freenect2", "freenect", "openni")
KINECT_PIPELINES = ("auto", "opengl", "opencl", "cuda", "cpu")
BLOB_MODES = ("background", "motion", "bright", "dark")


@dataclass
class Config:
    # --- video input -------------------------------------------------------
    source: str = ""           # webcam index, "kinect", "0+kinect" (side by side), video file or image
                               # ("" = the first webcam, or the first Kinect with --no-webcams)
    no_webcams: bool = False   # do not list / use webcams
    no_kinects: bool = False   # do not list / use Kinects (no libfreenect2 at all)
    kinect_stream: str = "auto"  # kinect: "color", "depth" or auto (depth when only Blobs run)
    z_from_kinect: bool = False  # replace MediaPipe's z by the Kinect distance (metres)
    kinect_backend: str = "auto"   # freenect2 (Kinect v2), freenect (v1), openni
    kinect_pipeline: str = "auto"  # Kinect v2 depth processing: opengl / opencl / cuda / cpu
    width: int = 640
    height: int = 480
    mirror: bool = False       # flip horizontally (x grows to the viewer's right)
    loop: bool = True          # loop video files
    frames: int = 0            # stop after N frames (0 = run forever)
    max_fps: float = 0.0       # cap the analysis rate (0 = as fast as possible)

    # --- detectors ---------------------------------------------------------
    pose: bool = False
    hands: bool = False
    face: bool = False
    gesture: bool = False      # hands + gesture classification
    blobs: bool = False        # OpenCV blob detection (shapes, e.g. ceiling camera)
    blob_mode: str = "background"
    blob_threshold: int = 25   # sensitivity: grey-level difference (1..255)
    blob_min_area: float = 0.004   # fraction of the image (0.004 ~ 1200 px at 640x480)
    blob_max_area: float = 0.5
    blob_max: int = 8          # largest N blobs are reported
    blob_blur: int = 5         # gaussian blur kernel (0 = none)
    blob_morph: int = 5        # open/close kernel (0 = none)
    blob_learn_frames: int = 30    # frames averaged for the background
    blob_adapt: float = 0.0    # 0 = static background, 0.01 = follows slow changes
    blob_roi: tuple[float, float, float, float] | None = None   # x0 y0 x1 y1 (0..1)
    max_poses: int = 4
    max_hands: int = 4
    max_faces: int = 4
    min_confidence: float = 0.5
    models_dir: str = DEFAULT_MODELS_DIR

    # --- OSC output --------------------------------------------------------
    ip: str = "127.0.0.1"
    port: int = 5005
    formats: tuple[str, ...] = ("mp",)
    prefix: str = "/mp"
    id_style: str = "arg"
    bundle: bool = False
    send_landmarks: bool = True     # pose & hand landmarks
    send_world: bool = False        # pose world landmarks (meters)
    send_face_landmarks: bool = False  # 478 face points, large message
    send_blendshapes: bool = True
    kinect_y: str = "image"
    kinect_from: str = "auto"  # what feeds /kinect/person: poses or blobs (auto: blobs if enabled)
    invert_x: bool = False

    # --- tracking ----------------------------------------------------------
    track_distance: float = 0.25  # max normalized distance to keep an ID
    track_timeout: float = 0.5    # seconds without detection before "leave"
    smooth: float = 0.0           # 0 = raw, 0.9 = very smooth (EMA weight)
    max_speed: float = 2.0        # normalized units/s mapped to speed = 1.0

    # --- UI / logging ------------------------------------------------------
    preview: bool = True
    panel: bool = True         # control panel inside the preview window
    template: str = "classic"  # look & feel: name in templates/ or a path to a .py file
    templates_dir: str = DEFAULT_TEMPLATES_DIR
    show_ids: bool = True
    verbose: bool = False
    quiet: bool = False
    list_cameras: bool = False  # print the cameras and exit
    config: str = ""           # configuration file (JSON) to load at start-up

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            self.source = "kinect" if self.no_webcams else "0"
        if self.no_webcams and self.no_kinects:
            raise ValueError("--no-webcams and --no-kinects together leave no camera to use")
        # Default to pose detection when nothing was selected.
        if not (self.pose or self.hands or self.face or self.gesture or self.blobs):
            self.pose = True
        if self.gesture:
            self.hands = True
        self.formats = tuple(self.formats)
        for fmt in self.formats:
            if fmt not in FORMATS:
                raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
        if self.id_style not in ID_STYLES:
            raise ValueError(f"unknown id style {self.id_style!r}; choose from {ID_STYLES}")
        if self.kinect_y not in KINECT_Y_MODES:
            raise ValueError(f"unknown kinect_y {self.kinect_y!r}; choose from {KINECT_Y_MODES}")
        if self.kinect_from not in KINECT_FROM:
            raise ValueError(f"unknown kinect_from {self.kinect_from!r}; choose from {KINECT_FROM}")
        if self.kinect_stream not in KINECT_STREAMS:
            raise ValueError(f"unknown kinect_stream {self.kinect_stream!r}; choose from {KINECT_STREAMS}")
        if self.kinect_backend not in KINECT_BACKENDS:
            raise ValueError(f"unknown kinect_backend {self.kinect_backend!r}; choose from {KINECT_BACKENDS}")
        if self.kinect_pipeline not in KINECT_PIPELINES:
            raise ValueError(f"unknown kinect_pipeline {self.kinect_pipeline!r}; choose from {KINECT_PIPELINES}")
        if self.blob_mode not in BLOB_MODES:
            raise ValueError(f"unknown blob mode {self.blob_mode!r}; choose from {BLOB_MODES}")
        if not 1 <= self.blob_threshold <= 255:
            raise ValueError("blob threshold must be in 1..255")
        if not 0.0 <= self.blob_min_area <= self.blob_max_area <= 1.0:
            raise ValueError("blob areas must satisfy 0 <= min <= max <= 1 (fractions of the image)")
        if self.blob_roi is not None:
            self.blob_roi = tuple(float(v) for v in self.blob_roi)
            if len(self.blob_roi) != 4 or not all(0.0 <= v <= 1.0 for v in self.blob_roi):
                raise ValueError("blob roi must be four numbers in 0..1: x0 y0 x1 y1")
        if not 0.0 <= self.smooth < 1.0:
            raise ValueError("smooth must be in [0, 1)")
        if not self.prefix.startswith("/"):
            self.prefix = "/" + self.prefix
        self.prefix = self.prefix.rstrip("/")

    def model_path(self, name: str) -> str:
        return os.path.join(self.models_dir, name)

    @property
    def effective_kinect_stream(self) -> str:
        """'depth' when the Blobs detector is the only one (a ceiling Kinect), else 'color'."""
        if self.kinect_stream != "auto":
            return self.kinect_stream
        only_blobs = self.blobs and not (self.pose or self.hands or self.face or self.gesture)
        return "depth" if only_blobs else "color"

    @property
    def kinect_source_is_blobs(self) -> bool:
        """Whether /kinect/person is derived from blobs (else from poses)."""
        if self.kinect_from == "auto":
            return self.blobs
        return self.kinect_from == "blobs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mediapipe_osc.py",
        description=("Analiza una cámara en vivo con MediaPipe y envía los resultados por OSC. "
                     "Analyze a live camera with MediaPipe and send the results as OSC. "
                     "Manual en castellano: README_mediapipe_osc_es.md; English manual: README_mediapipe_osc.md."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("entrada de video / video input")
    g.add_argument("--source", default="",
                   help="webcam index (0, 1, ...), 'kinect' (kinect:1 = second one), several cameras "
                        "side by side joined with + (e.g. 0+kinect), a video file or a still image "
                        "(default: webcam 0, or the first Kinect with --no-webcams)")
    g.add_argument("--no-webcams", action="store_true",
                   help="ignore webcams: only Kinects are listed and used")
    g.add_argument("--no-kinects", action="store_true",
                   help="ignore Kinects: only webcams are listed and used (libfreenect2 is never loaded)")
    g.add_argument("--cam", type=int, default=None,
                   help="webcam index (same as --source, kept for the older scripts)")
    g.add_argument("--list-cameras", action="store_true",
                   help="print the cameras found on this computer and exit")
    g.add_argument("--kinect-stream", choices=KINECT_STREAMS, default="auto",
                   help="with --source kinect: the colour camera or the depth map (as a grey "
                        "image, near = bright). 'auto' = depth when only --blobs runs")
    g.add_argument("--z-from-kinect", action="store_true",
                   help="replace MediaPipe's relative z by the distance measured by the Kinect "
                        "(metres, 0 = no reading) for every landmark; needs a Kinect source "
                        "(or several Kinects side by side, not a Kinect + webcam)")
    g.add_argument("--kinect-backend", choices=KINECT_BACKENDS, default="auto",
                   help="'freenect2' = Kinect v2 via pylibfreenect2, 'freenect' = Kinect v1 via "
                        "libfreenect, 'openni' = OpenCV's OpenNI backend; 'auto' tries them in that order")
    g.add_argument("--kinect-pipeline", choices=KINECT_PIPELINES, default="auto",
                   help="Kinect v2 depth processing pipeline ('auto' = first that works: opengl, "
                        "opencl, cuda, cpu)")
    g.add_argument("--width", "--cam_width", dest="width", type=int, default=640)
    g.add_argument("--height", "--cam_height", dest="height", type=int, default=480)
    g.add_argument("--mirror", action="store_true",
                   help="flip the image horizontally so x grows to the viewer's right")
    g.add_argument("--no-loop", dest="loop", action="store_false",
                   help="do not loop video files")
    g.add_argument("--frames", type=int, default=0,
                   help="stop after this many frames (0 = run forever)")
    g.add_argument("--max-fps", type=float, default=0.0,
                   help="cap the analysis/sending rate (0 = as fast as possible)")

    g = p.add_argument_group("detectores / detectors (predeterminado / default: --pose)")
    g.add_argument("--pose", action="store_true", help="body landmarks (33 points per person)")
    g.add_argument("--hands", action="store_true", help="hand landmarks (21 points per hand)")
    g.add_argument("--face", action="store_true", help="face landmarks + blendshapes")
    g.add_argument("--gesture", action="store_true",
                   help="hand landmarks + gesture classification (implies --hands)")
    g.add_argument("--blobs", action="store_true",
                   help="OpenCV blob detection: shapes that stand out from the background "
                        "(ceiling camera, silhouettes, objects, Kinect depth). No MediaPipe model.")
    g.add_argument("--blob-mode", choices=BLOB_MODES, default="background",
                   help="'background': difference to a learned empty scene (b / 'Learn bg' "
                        "re-captures it); 'motion': adaptive MOG2; 'bright'/'dark': fixed "
                        "threshold on the grey level")
    g.add_argument("--blob-threshold", type=int, default=25,
                   help="sensitivity, grey-level difference 1..255 (lower = more sensitive)")
    g.add_argument("--blob-min-area", type=float, default=0.004,
                   help="smallest blob, as a fraction of the image (0.004 = 1200 px at 640x480)")
    g.add_argument("--blob-max-area", type=float, default=0.5,
                   help="largest blob, as a fraction of the image")
    g.add_argument("--blob-max", type=int, default=8, help="report at most this many blobs (largest first)")
    g.add_argument("--blob-blur", type=int, default=5, help="gaussian blur kernel size, 0 = off")
    g.add_argument("--blob-morph", type=int, default=5, help="morphological open/close kernel, 0 = off")
    g.add_argument("--blob-learn-frames", type=int, default=30,
                   help="frames averaged when learning the background")
    g.add_argument("--blob-adapt", type=float, default=0.0,
                   help="background adaptation rate where nothing is detected (0 = static)")
    g.add_argument("--blob-roi", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None,
                   help="only detect inside this normalized rectangle")
    g.add_argument("--max-poses", type=int, default=4)
    g.add_argument("--max-hands", type=int, default=4)
    g.add_argument("--max-faces", type=int, default=4)
    g.add_argument("--min-confidence", type=float, default=0.5,
                   help="detection / presence / tracking confidence threshold")
    g.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                   help="folder containing the .task model files")

    g = p.add_argument_group("salida OSC / OSC output")
    g.add_argument("--ip", default="127.0.0.1", help="OSC receiver address")
    g.add_argument("--port", type=int, default=5005, help="OSC receiver port")
    g.add_argument("--format", dest="formats", nargs="+", choices=FORMATS, default=["mp"],
                   help="'mp' = numeric /mp/... messages; 'legacy' = string messages of the "
                        "original ejemplo-*.py scripts; 'kinect' = /kinect/person id x y speed "
                        "for the Sonido_Cuadrafonico scripts. Several can be combined.")
    g.add_argument("--prefix", default="/mp", help="address prefix for the 'mp' format")
    g.add_argument("--id-style", choices=ID_STYLES, default="arg",
                   help="'arg': /mp/pose/center <id> x y z  -  'address': /mp/pose/<id>/center x y z")
    g.add_argument("--bundle", action="store_true",
                   help="pack all messages of a frame in one OSC bundle")
    g.add_argument("--no-landmarks", dest="send_landmarks", action="store_false",
                   help="do not send pose/hand landmark lists (only centers, velocity, ...)")
    g.add_argument("--world", dest="send_world", action="store_true",
                   help="also send pose world landmarks (meters, hip-centered)")
    g.add_argument("--face-landmarks", dest="send_face_landmarks", action="store_true",
                   help="send the 478 face landmarks (large message)")
    g.add_argument("--no-blendshapes", dest="send_blendshapes", action="store_false")
    g.add_argument("--kinect-y", choices=KINECT_Y_MODES, default="image",
                   help="'kinect' format: y from the image (overhead camera) or from the "
                        "person's size (frontal camera: bigger = closer = smaller y)")
    g.add_argument("--kinect-from", choices=KINECT_FROM, default="auto",
                   help="'kinect' format: derive /kinect/person from body poses or from blobs "
                        "('auto' = blobs whenever the Blobs detector is on)")
    g.add_argument("--invert-x", action="store_true",
                   help="'kinect' format: send 1-x instead of x")

    g = p.add_argument_group("seguimiento / tracking")
    g.add_argument("--track-distance", type=float, default=0.25,
                   help="max normalized distance between frames to keep the same ID")
    g.add_argument("--track-timeout", type=float, default=0.5,
                   help="seconds without detection before a 'leave' event")
    g.add_argument("--smooth", type=float, default=0.0,
                   help="exponential smoothing of landmarks, 0 = none, 0.9 = very smooth")
    g.add_argument("--max-speed", type=float, default=2.0,
                   help="speed (normalized units per second) that maps to 1.0")

    g = p.add_argument_group("vista previa y registro / preview and logging")
    g.add_argument("--no-preview", dest="preview", action="store_false",
                   help="run headless (no window)")
    g.add_argument("--no-panel", dest="panel", action="store_false",
                   help="preview without the control panel (keyboard shortcuts only)")
    g.add_argument("--template", default="classic",
                   help="look & feel of the window: a name from the templates/ folder "
                        "(classic, minimal, ascii, baroque) or the path to a template .py file")
    g.add_argument("--templates-dir", default=DEFAULT_TEMPLATES_DIR,
                   help="folder with the template files")
    g.add_argument("--config", default="",
                   help="load this configuration file (JSON saved with the 'Save config' button) at start-up")
    g.add_argument("--no-ids", dest="show_ids", action="store_false",
                   help="do not draw IDs on the preview")
    g.add_argument("--verbose", action="store_true", help="print every OSC message")
    g.add_argument("--quiet", action="store_true", help="print only errors")
    return p


def parse_args(argv: list[str] | None = None) -> Config:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.cam is not None:
        ns.source = str(ns.cam)
    kwargs = {k: v for k, v in vars(ns).items() if k != "cam"}
    try:
        return Config(**kwargs)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # unreachable, keeps type checkers happy
