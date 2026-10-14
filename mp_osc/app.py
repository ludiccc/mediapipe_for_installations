"""The application: capture -> detectors -> tracking -> OSC, plus the window.

If you read one file of this package, read this one. The data flow of every
frame is in :meth:`App.step`; everything else is plumbing around it.

The big picture
---------------
::

    FrameSource ──frame──▶ Analyzer ──detections──▶ Tracker(s) ──tracks──▶ MessageBuilder ──▶ OscSender ──UDP──▶ receivers
    (capture.py)           (detectors.py,            (tracking.py)          (osc_out.py)        (osc_out.py)
                            blobs.py)                      │
                                                           └──▶ Theme.draw_preview ──▶ window + ControlPanel (ui.py)

* **FrameSource** (``capture.py``): where images come from - a webcam, a
  Kinect, several of them side by side, a video file or a still image. All
  look the same to the rest of the app (``read()`` gives a BGR frame).
* **Analyzer** (``detectors.py``): runs the detectors that are switched on -
  MediaPipe's body pose / hands / gestures / face, and the OpenCV *Blobs*
  detector (``blobs.py``) - and returns plain numpy arrays.
* **Tracker** (``tracking.py``): MediaPipe does not say *who* is who from one
  frame to the next. One tracker per kind (poses, hands, faces, blobs) gives
  every detection a stable id, reports ``enter`` / ``leave`` and a velocity.
* **MessageBuilder / OscSender** (``osc_out.py``): turn the tracks into OSC
  messages (``/mp/pose/center 3 0.5 0.4 0.0`` ...) and send them over UDP.
* **Theme / ControlPanel** (``themes.py``, ``ui.py``): the window. The panel
  is drawn by the app itself inside the OpenCV window - there is no GUI
  toolkit - and its buttons call the ``set_*`` methods of :class:`App`.

Threads
-------
With a preview there are two threads (see :meth:`App.run`):

* the **analysis thread** runs :meth:`App.step` in a loop: read, analyze,
  track, send, draw;
* the **main thread** runs :meth:`App._window_loop`: shows the latest drawn
  frame, polls the keyboard and the mouse (``cv2.waitKey``), so the window
  stays responsive however slow the models are.

MediaPipe objects must only be touched by the thread that created them, so
changes coming from the panel (switch a detector, change the camera) are
*queued* with :meth:`App._on_analysis_thread` and applied by the analysis
thread between two frames. Plain attribute writes (``mirror``, ``paused``)
need no such care. Without a preview (``--no-preview``) there is a single
thread and the queue is executed immediately.

Sources and the *Source* section of the panel
---------------------------------------------
``self.source`` is the running :class:`FrameSource`. The panel's camera list
holds everything :mod:`mp_osc.cameras` found (webcams and Kinects); the
checked entries form ``self.selection`` - one spec, or several joined with
``+`` which :func:`open_source` turns into a :class:`CompositeSource` (the
pictures side by side, in the order of the list). Webcams and Kinects live in
the same list. The radio group shows the *mode*: ``camera`` (the selection),
``pause`` (no OSC is sent) or ``file`` (the video/image of the command line).

Switching sources never re-opens a device that is already running: the parts
of the new composition that are open are kept, only the new ones are opened
and only the dropped ones are released (opening a camera twice fails, and
opening a Kinect v2 twice crashes libfreenect2).

Crop & mapping
--------------
Every camera can have a :class:`CameraMapping` (``self.mappings[spec]``): an
input quad (which part of the picture) and an output quad (where it goes in
the camera's slot), applied to the frame right after ``read()`` - see
``mapping.py``. The gear of a list row opens the :class:`MappingEditor` for
that camera; while it is open the preview shows the raw picture (input
mode) or the mapped one (output mode) with draggable handles.

Configuration files
-------------------
``save_config`` / ``load_config`` write and read a JSON file with everything
the panel can change (cameras, detectors, blob settings, OSC target, formats,
image options, template, mappings) - the *Save config* / *Load config*
buttons, ``--config`` on the command line.

Depth
-----
A Kinect source can provide, for every pixel, the distance in metres
(``source.depth()``). With *Z from Kinect* on (``cfg.z_from_kinect``) the
app replaces MediaPipe's relative z of every landmark by that distance
(:mod:`mp_osc.depth`), and tells the receivers with ``/mp/z_source``. A
composition of several Kinects has depth too; one with a webcam in it has
not (the z stays MediaPipe's).
"""

from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time
from typing import Callable

import cv2

from . import __version__
from .cameras import CameraScanner, format_camera_list, list_cameras
from .capture import CompositeSource, FrameSource, is_kinect_spec, open_source
from .config import Config, parse_args
from .depth import apply_depth
from .detectors import Analyzer
from .mapping import CameraMapping, MappingEditor, apply_mappings
from .osc_out import MessageBuilder, OscSender
from .themes import Theme, available_templates, load_theme
from .tracking import Tracker

WINDOW = "MediaPipe -> OSC"          # title of the preview window (tests look for it)
MODES = ("camera", "pause", "file")   # the Source radio group: live cameras / OSC paused / the file


class FpsMeter:
    """Smoothed frames-per-second estimate (exponential moving average)."""

    def __init__(self, alpha: float = 0.9):
        self._t = None
        self._fps = 0.0
        self._alpha = alpha

    def tick(self, now: float) -> float:
        if self._t is not None:
            dt = now - self._t
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else self._alpha * self._fps + (1 - self._alpha) * inst
        self._t = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class App:
    """Everything wired together. ``App(cfg).run()`` is the whole program.

    Attributes the control panel reads or writes (see ``ui.py``):

    ``cfg``           the :class:`Config` (detectors on/off, blob settings, formats ...)
    ``source``        the running frame source; ``mode`` / ``set_mode`` for the radio group
    ``selection``     specs of the cameras checked in the list, left to right
    ``cameras``       the :class:`CameraScanner` with the list of cameras found
    ``sender``        the OSC sender (``ip``, ``port``, ``paused``, ``sent``)
    ``theme``         the active look (``set_template`` / ``next_template``)
    ``mirror``, ``draw_landmarks``, ``show_ids``   preview options
    """

    def __init__(self, cfg: Config, sender: OscSender | None = None):
        self.cfg = cfg
        self.log = (lambda *a, **k: None) if cfg.quiet else print
        self.log(f"MediaPipe -> OSC {__version__}")

        # ---- look & feel (loaded before anything is drawn; cheap, no window needed)
        try:
            self.theme: Theme = load_theme(cfg.template, cfg.templates_dir)
        except (FileNotFoundError, ImportError, SyntaxError) as exc:
            raise RuntimeError(f"template: {exc}") from exc

        # ---- cameras: the list is filled by a background scan (every webcam
        # is opened for a moment), started with the panel or by scan_cameras().
        self.cameras = CameraScanner(skip=self._cameras_in_use, on_done=self._on_scan_done,
                                     lister=lambda skip: list_cameras(skip=skip, webcams=not cfg.no_webcams,
                                                                      kinects=not cfg.no_kinects))
        self.panel = None

        # ---- video source
        self.source: FrameSource = self._open(cfg.source)
        self.source.want_depth = cfg.z_from_kinect
        self.log(f"video source: {self.source.name}")
        # The file/image given on the command line (so the 'File' option can return to it).
        self.file_source = cfg.source if self.source.kind in ("video", "image") else ""
        # The cameras checked in the panel's list, left to right (empty for a file).
        self.selection: list[str] = self._live_specs(cfg.source)
        self._time_offset = 0.0      # keeps app time monotonic across source switches
        self._last_t = 0.0
        self.source_error = ""       # shown under the Source radio buttons
        # Crop & mapping per camera spec, and the editor when one is being edited.
        self.mappings: dict[str, CameraMapping] = {}
        self.editor: MappingEditor | None = None
        self._raw_slot = None        # raw picture of the camera being edited (input mode)
        self.config_path = cfg.config or ""
        self.config_note = ""        # result of the last save / load

        # ---- detectors
        try:
            self.analyzer = Analyzer(cfg)
        except Exception:
            self.source.release()   # do not leave the camera thread running
            raise
        self.log(f"detectors: {', '.join(self.analyzer_names())}")

        # ---- tracking: one independent id counter per kind (pose ids, hand
        # ids, face ids and blob ids are unrelated to each other).
        def tracker() -> Tracker:
            return Tracker(cfg.track_distance, cfg.track_timeout, cfg.smooth)

        self.pose_tracker = tracker()
        self.hand_tracker = tracker()
        self.face_tracker = tracker()
        self.blob_tracker = tracker()

        # ---- OSC
        self.builder = MessageBuilder(cfg)
        self.sender = sender or OscSender(cfg.ip, cfg.port, bundle=cfg.bundle, verbose=cfg.verbose)
        self.log(f"OSC -> {cfg.ip}:{cfg.port}  formats={','.join(cfg.formats)}  "
                 f"prefix={cfg.prefix}  id_style={cfg.id_style}  bundle={cfg.bundle}")
        self._z_source_sent = ""     # last /mp/z_source value told to the receivers

        # ---- preview window
        self.preview = cfg.preview
        if self.preview and sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            # Qt aborts the process (not catchable) when there is no display.
            print("no display found (DISPLAY is not set): running without preview, "
                  "use --no-preview to silence this", file=sys.stderr)
            self.preview = False
        self.mirror = cfg.mirror
        self.show_ids = cfg.show_ids
        self.draw_landmarks = True
        self.frame_index = 0
        self.fps = FpsMeter()
        self.counts = (0, 0, 0, 0)    # poses, hands, faces, blobs in the last frame
        self._quit = False
        self._window_ready = False
        self._worker: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._commands: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._latest_frame = None
        if self.preview and cfg.panel:
            from .ui import ControlPanel
            self.panel = ControlPanel(self)
            self.scan_cameras()

    # ====================================================================
    # plumbing between the two threads
    # ====================================================================
    def _on_analysis_thread(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the analysis thread (immediately if there is no such thread)."""
        if self._worker is None or threading.current_thread() is self._worker:
            fn()
        else:
            self._commands.put(fn)

    def _apply_pending(self) -> None:
        """Analysis thread, between frames: execute what the panel queued."""
        while True:
            try:
                fn = self._commands.get_nowait()
            except queue.Empty:
                return
            fn()

    def _mark_dirty(self) -> None:
        """Ask the window to redraw the panel (something it shows has changed)."""
        if self.panel is not None:
            self.panel.dirty = True

    # ====================================================================
    # detectors
    # ====================================================================
    def analyzer_names(self) -> list[str]:
        a = self.analyzer
        names = []
        if a.pose_enabled: names.append("pose")
        if a.gesture_enabled: names.append("gesture")
        elif a.hands_enabled: names.append("hands")
        if a.face_enabled: names.append("face")
        if a.blobs_enabled: names.append("blobs")
        return names

    def set_detectors(self, pose: bool | None = None, hands: bool | None = None,
                      face: bool | None = None, gesture: bool | None = None,
                      blobs: bool | None = None) -> None:
        """Switch detectors on/off (None = leave as is). Gestures need hands."""
        cfg = self.cfg
        if pose is not None: cfg.pose = bool(pose)
        if hands is not None: cfg.hands = bool(hands)
        if face is not None: cfg.face = bool(face)
        if gesture is not None: cfg.gesture = bool(gesture)
        if blobs is not None: cfg.blobs = bool(blobs)
        if gesture:
            cfg.hands = True            # gestures need the hand detector
        if hands is False:
            cfg.gesture = False
        self._on_analysis_thread(self._apply_detectors)

    def _apply_detectors(self) -> None:
        cfg = self.cfg
        self.analyzer.configure(cfg.pose, cfg.hands, cfg.face, cfg.gesture, cfg.blobs)

        # Forget the tracks of detectors that were switched off and tell the receivers.
        gone = self.builder.leave_all(
            self.pose_tracker.reset() if not cfg.pose else [],
            self.hand_tracker.reset() if not cfg.hands else [],
            self.face_tracker.reset() if not cfg.face else [],
            self.blob_tracker.reset() if not cfg.blobs else [],
        )
        self.sender.send(gone)
        self.log(f"\ndetectors: {', '.join(self.analyzer_names()) or 'none'}")

    # ---- blobs ------------------------------------------------------------
    def learn_background(self) -> None:
        """Capture the (empty) scene again as the blob background ('b' key / button)."""
        def apply():
            det = self.analyzer.blob_detector
            if det is not None:
                det.learn()
                self.sender.send(self.builder.leave_all([], [], [], self.blob_tracker.reset()))
                self.log("\nblobs: learning the background")
        self._on_analysis_thread(apply)

    def step_blob_threshold(self, direction: int) -> None:
        """+/- button: steps of 5, of 1 below 10 (where it matters)."""
        current = self.cfg.blob_threshold
        step = 5 if (current >= 10 if direction > 0 else current > 10) else 1
        self.set_blob_threshold(current + step * (1 if direction > 0 else -1))

    def set_blob_threshold(self, value: int) -> None:
        value = int(max(1, min(255, value)))
        self.cfg.blob_threshold = value
        det = self.analyzer.blob_detector
        if det is not None:
            det.threshold = value       # plain int, safe to set from any thread

    def step_blob_min_area(self, direction: int) -> None:
        factor = 1.5 if direction > 0 else 1 / 1.5
        self.set_blob_min_area(self.cfg.blob_min_area * factor)

    def set_blob_min_area(self, value: float) -> None:
        value = float(max(0.0001, min(self.cfg.blob_max_area, value)))
        self.cfg.blob_min_area = value
        det = self.analyzer.blob_detector
        if det is not None:
            det.min_area = value

    # ====================================================================
    # sources: camera(s) / kinect / pause / file
    # ====================================================================
    @property
    def mode(self) -> str:
        """'pause' while OSC sending is paused, 'file' on a video/image, else 'camera' (webcams and/or Kinects)."""
        if self.sender.paused:
            return "pause"
        return "file" if self.source.kind in ("video", "image") else "camera"

    def set_mode(self, mode: str) -> None:
        """Radio group 'Source': exactly one of camera / pause (/ file).

        ``camera`` runs the cameras checked in the list (webcams and Kinects,
        side by side if several), ``file`` the video/image of the command
        line. ``pause`` only stops the OSC output.
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "pause":
            self.sender.paused = True
            self._mark_dirty()
            return
        self.sender.paused = False
        if mode == "camera":
            if not self.selection:
                # nothing checked (we were on a file): the first camera of the list
                first = [c.spec for c in self.cameras.cameras]
                self.selection = first[:1] or [self._default_live_spec()]
            self._apply_selection()
        elif mode == "file" and self.file_source and self.source.kind not in ("video", "image"):
            self._switch_source(self.file_source)
        self._mark_dirty()

    def _default_live_spec(self) -> str:
        return "kinect" if self.cfg.no_webcams else "0"

    # ---- the camera list ---------------------------------------------------
    @staticmethod
    def _live_specs(spec: str) -> list[str]:
        """The camera/kinect specs inside a source spec ("0+kinect" -> ["0", "kinect"]; files -> [])."""
        parts = [s.strip() for s in str(spec).split("+") if s.strip()]
        return [p for p in parts if p.lstrip("-").isdigit() or is_kinect_spec(p)]

    def _cameras_in_use(self) -> dict[str, str]:
        """spec -> name of the live sources running now (the scanner must not probe them)."""
        source = getattr(self, "source", None)
        if source is None:
            return {}
        parts = getattr(source, "parts", [source])
        return {getattr(p, "spec", ""): p.name for p in parts if p.kind in ("camera", "kinect")}

    def camera_labels(self) -> list[str]:
        """Rows of the list: '1. 0  Integrated Camera' when checked (1 = leftmost), '   ...' otherwise."""
        labels = []
        for c in self.cameras.cameras:
            if c.spec in self.selection:
                labels.append(f"{self.selection.index(c.spec) + 1}. {c.label}")
            else:
                labels.append(f"   {c.label}")
        return labels

    def selected_camera_rows(self) -> list[int]:
        """Rows of the list that are checked (in list order)."""
        return [i for i, c in enumerate(self.cameras.cameras) if c.spec in self.selection]

    def toggle_camera_row(self, row: int) -> None:
        """Click on a row: add that camera to the composition (at the right), or remove it.

        The last checked camera cannot be removed (there would be nothing to
        show); at least one source always runs.
        """
        cams = self.cameras.cameras
        if not 0 <= row < len(cams):
            return
        spec = cams[row].spec
        if spec in self.selection:
            if len(self.selection) == 1:
                return
            self.selection.remove(spec)
        else:
            self.selection.append(spec)
        self.sender.paused = False
        self._apply_selection()

    def move_camera_row(self, row: int, direction: int) -> None:
        """The < > buttons of a checked row: move that camera left (-1) or right (+1) in the composition."""
        cams = self.cameras.cameras
        if not 0 <= row < len(cams) or cams[row].spec not in self.selection:
            return
        i = self.selection.index(cams[row].spec)
        j = i + (1 if direction > 0 else -1)
        if not 0 <= j < len(self.selection):
            return
        self.selection[i], self.selection[j] = self.selection[j], self.selection[i]
        self._apply_selection()

    def select_camera(self, spec: str) -> None:
        """Run exactly that camera (by spec, e.g. "2" or "kinect"); un-pauses."""
        self.selection = [str(spec)]
        self.sender.paused = False
        self._apply_selection()

    def _apply_selection(self) -> None:
        """Make the running source match ``self.selection`` (already-open cameras are kept)."""
        wanted = "+".join(self.selection)
        if getattr(self.source, "spec", None) != wanted:
            self._switch_source(wanted)
        self._mark_dirty()

    def scan_cameras(self) -> None:
        self.source_error = ""
        self.cameras.start()
        self._mark_dirty()

    def _on_scan_done(self) -> None:
        """Scanner thread: give the running webcams the names just found, redraw the list."""
        names = {c.index: c.name for c in self.cameras.cameras if c.kind == "webcam"}
        for part in self._parts(self.source):
            if part.kind == "camera" and getattr(part, "index", None) in names and "(" not in part.name:
                part.name = f"webcam {part.index} ({names[part.index]})"
        update = getattr(self.source, "_update_name", None)
        if update is not None:
            update()
        self._mark_dirty()

    def source_note(self) -> tuple[str, bool]:
        """Text under the Source controls: error, scanning, or the running source."""
        if self.source_error:
            return self.source_error, True
        if self.cameras.scanning:
            return "scanning cameras...", False
        if self.cameras.error:
            return f"scan failed: {self.cameras.error}", True
        return self.source.name, False

    # ---- opening / switching -----------------------------------------------
    def _open(self, spec: str) -> FrameSource:
        cfg = self.cfg
        names = {c.index: c.name for c in self.cameras.cameras if c.kind == "webcam"}
        return open_source(spec, cfg.width, cfg.height, cfg.loop, cfg.effective_kinect_stream,
                           kinect_backend=cfg.kinect_backend, kinect_pipeline=cfg.kinect_pipeline,
                           camera_names=names)

    def _switch_source(self, spec: str) -> None:
        self.source_error = f"opening {spec}..."
        self._mark_dirty()
        self._on_analysis_thread(lambda: self._apply_source(spec))

    @staticmethod
    def _parts(source: FrameSource) -> list[FrameSource]:
        """The sources inside ``source`` (itself, or the parts of a composition)."""
        return list(getattr(source, "parts", [source]))

    def _apply_source(self, spec: str) -> None:
        """Analysis thread: make ``spec`` the running source, keeping the old one on failure.

        Parts of the new composition that are already open (same spec) are
        reused as they are; only the new parts are opened, and the parts that
        are no longer wanted are released *after* the switch. A device is
        therefore never opened twice.
        """
        wanted = [s.strip() for s in str(spec).split("+") if s.strip()]
        current = {getattr(p, "spec", None): p for p in self._parts(self.source)}
        opened: list[FrameSource] = []
        parts: list[FrameSource] = []
        try:
            with self.cameras.lock:      # not while the scanner is probing cameras
                for one in wanted:
                    if one in current:
                        parts.append(current[one])
                    else:
                        part = self._open(one)
                        opened.append(part)
                        parts.append(part)
        except (RuntimeError, ValueError) as exc:
            for part in opened:
                part.release()
            # the panel gets the gist ("no Kinect found"), the terminal the details
            self.source_error = str(exc).split("(")[0].split(". ")[0].strip() or str(exc)
            self.selection = self._live_specs(getattr(self.source, "spec", "")) or self.selection
            self.log(f"\nsource: {exc}")
            self._mark_dirty()
            return
        if not parts:
            return
        new = parts[0] if len(parts) == 1 else CompositeSource(parts)
        new.spec = "+".join(wanted)  # type: ignore[attr-defined]
        self._time_offset = self._last_t        # a new source's clock starts at 0
        new.want_depth = self.cfg.z_from_kinect
        self.source = new
        for one, part in current.items():
            if part not in parts:
                part.release()
        self.cfg.source = new.spec
        self.source_error = ""
        # Everybody 'leaves' the old picture; blobs need a new background.
        self.sender.send(self.builder.leave_all(
            self.pose_tracker.reset(), self.hand_tracker.reset(), self.face_tracker.reset(),
            self.blob_tracker.reset()))
        det = self.analyzer.blob_detector
        if det is not None:
            det.learn()
        self.log(f"\nvideo source: {new.name}")
        self._mark_dirty()

    # ---- crop & mapping --------------------------------------------------------
    def slots(self, frame) -> list[tuple[str, int, int]]:
        """``(spec, x0, x1)`` of every camera in the composed frame (one slot for a single source)."""
        slots = getattr(self.source, "slots", None)
        if slots:
            return list(slots)
        return [(getattr(self.source, "spec", ""), 0, frame.shape[1])]

    def mapping_for(self, spec: str) -> CameraMapping:
        """The mapping of a camera (created as identity when it has none yet)."""
        mapping = self.mappings.get(spec)
        if mapping is None:
            mapping = self.mappings[spec] = CameraMapping()
        return mapping

    def has_mapping(self, spec: str) -> bool:
        m = self.mappings.get(spec)
        return m is not None and not m.identity

    def open_mapping_editor(self, spec: str, mode: str = "input") -> None:
        """The gear of a list row: edit the crop & mapping of that camera (runs it first if needed)."""
        if self.editor is not None and self.editor.spec == spec:
            self.close_mapping_editor()
            return
        if spec not in self.selection and self._live_specs(spec):
            self.selection.append(spec)
            self.sender.paused = False
            self._apply_selection()
        self.editor = MappingEditor(spec, self.mapping_for(spec), mode)
        self._mark_dirty()

    def close_mapping_editor(self) -> None:
        self.editor = None
        self._raw_slot = None
        self._mark_dirty()

    def toggle_mapping_editor_row(self, row: int) -> None:
        cams = self.cameras.cameras
        if 0 <= row < len(cams):
            self.open_mapping_editor(cams[row].spec)

    def gear_state_row(self, row: int) -> str:
        """How the gear of a list row is drawn: 'editing', 'mapped' (a mapping is active) or ''."""
        cams = self.cameras.cameras
        if not 0 <= row < len(cams):
            return ""
        spec = cams[row].spec
        if self.editor is not None and self.editor.spec == spec:
            return "editing"
        return "mapped" if self.has_mapping(spec) else ""

    def editor_name(self, editor: MappingEditor | None = None) -> str:
        editor = editor or self.editor
        if editor is None:
            return ""
        cam = self.cameras.by_spec(editor.spec)
        return cam.name if cam is not None else editor.spec

    @staticmethod
    def _editor_slot(slots, editor: MappingEditor) -> tuple[str, int, int] | None:
        return next((sl for sl in slots if sl[0] == editor.spec), None)

    def _editor_view(self, frame, slots, editor: MappingEditor):
        """What the preview shows while editing: raw picture + input quad, or mapped + output quad.

        ``editor`` is the analysis thread's snapshot: the main thread may
        close the editor (set ``self.editor = None``) at any moment.
        """
        h, w = frame.shape[:2]
        if editor.mode == "input":
            raw = self._raw_slot
            if raw is None:
                return frame
            view = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
            editor.draw_quad(view)                # quads are defined on the unmirrored picture
            if self.mirror:
                view = cv2.flip(view, 1)
            editor.draw_info(view, self.editor_name(editor))
            return view
        slot = self._editor_slot(slots, editor)
        if slot is not None:
            x0, x1 = slot[1], slot[2]
            if self.mirror:
                x0, x1 = w - x1, w - x0
            editor.draw_quad(frame, x0, 0, x1 - x0, h)
        editor.draw_info(frame, self.editor_name(editor))
        return frame

    def editor_mouse(self, event: int, x: int, y: int) -> None:
        """Mouse events on the camera image while the editor is open (image pixel coordinates)."""
        editor = self.editor
        if editor is None or self._latest_frame is None:
            return
        h, w = self._latest_frame.shape[:2]
        if self.mirror:
            x = w - 1 - x
        if editor.mode == "input":
            rect = (0, w)
        else:
            slot = self._editor_slot(self.slots(self._latest_frame), editor)
            rect = (slot[1], slot[2]) if slot else (0, w)
        x0, x1 = rect
        if x1 <= x0:
            return
        nx, ny = (x - x0) / (x1 - x0), y / h
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.press(nx, ny, x1 - x0, h)
        elif event == cv2.EVENT_MOUSEMOVE:
            editor.drag(nx, ny)
        elif event == cv2.EVENT_LBUTTONUP:
            editor.release()
        self._mark_dirty()

    # ---- configuration files ---------------------------------------------------
    CONFIG_VERSION = 1

    def to_config(self) -> dict:
        """Everything the panel can change, as a JSON-able dict."""
        cfg = self.cfg
        return {
            "version": self.CONFIG_VERSION,
            "source": "+".join(self.selection) if self.selection else cfg.source,
            "detectors": {"pose": cfg.pose, "hands": cfg.hands, "gesture": cfg.gesture, "face": cfg.face,
                          "blobs": cfg.blobs},
            "blobs": {"mode": cfg.blob_mode, "threshold": cfg.blob_threshold, "min_area": cfg.blob_min_area,
                      "max_area": cfg.blob_max_area, "max": cfg.blob_max},
            "osc": {"ip": self.sender.ip, "port": self.sender.port, "formats": list(cfg.formats),
                    "send_landmarks": cfg.send_landmarks, "prefix": cfg.prefix, "id_style": cfg.id_style},
            "image": {"mirror": self.mirror, "draw": self.draw_landmarks, "show_ids": self.show_ids,
                      "z_from_kinect": cfg.z_from_kinect, "kinect_stream": cfg.kinect_stream},
            "template": cfg.template,
            "mappings": {spec: m.to_dict() for spec, m in self.mappings.items() if not m.identity},
        }

    def apply_config(self, data: dict) -> None:
        """Apply a dict made by :meth:`to_config` to the running app (sources are switched live)."""
        cfg = self.cfg
        det = data.get("detectors", {})
        if det:
            self.set_detectors(pose=det.get("pose"), hands=det.get("hands"), face=det.get("face"),
                               gesture=det.get("gesture"), blobs=det.get("blobs"))
        blobs = data.get("blobs", {})
        if blobs:
            if blobs.get("mode") in ("background", "motion", "bright", "dark"):
                cfg.blob_mode = blobs["mode"]
                det_ = self.analyzer.blob_detector
                if det_ is not None:
                    self._on_analysis_thread(lambda: det_.set_mode(cfg.blob_mode))
            if "threshold" in blobs:
                self.set_blob_threshold(int(blobs["threshold"]))
            if "max_area" in blobs:
                cfg.blob_max_area = float(blobs["max_area"])
            if "min_area" in blobs:
                self.set_blob_min_area(float(blobs["min_area"]))
            if "max" in blobs:
                cfg.blob_max = int(blobs["max"])
                if self.analyzer.blob_detector is not None:
                    self.analyzer.blob_detector.max_blobs = cfg.blob_max
        osc = data.get("osc", {})
        if osc:
            if "formats" in osc:
                cfg.formats = tuple(f for f in osc["formats"] if f in ("mp", "legacy", "kinect")) or cfg.formats
            if "send_landmarks" in osc:
                cfg.send_landmarks = bool(osc["send_landmarks"])
            if "prefix" in osc:
                cfg.prefix = osc["prefix"] if str(osc["prefix"]).startswith("/") else "/" + str(osc["prefix"])
                self.builder.prefix = cfg.prefix
            if "id_style" in osc and osc["id_style"] in ("arg", "address"):
                cfg.id_style = osc["id_style"]
            if "ip" in osc or "port" in osc:
                self.set_osc(ip=osc.get("ip"), port=osc.get("port"))
        image = data.get("image", {})
        if image:
            if "mirror" in image: self.mirror = bool(image["mirror"])
            if "draw" in image: self.draw_landmarks = bool(image["draw"])
            if "show_ids" in image: self.show_ids = bool(image["show_ids"])
            if image.get("kinect_stream") in ("auto", "color", "depth"):
                cfg.kinect_stream = image["kinect_stream"]
            if "z_from_kinect" in image:
                self.set_z_from_kinect(bool(image["z_from_kinect"]))
        if data.get("template") and data["template"] != cfg.template:
            self.set_template(data["template"])
        mappings = data.get("mappings", {})
        self.mappings = {spec: CameraMapping.from_dict(m) for spec, m in mappings.items()}
        if self.editor is not None:
            self.editor.mapping = self.mapping_for(self.editor.spec)
        source = data.get("source")
        if source and not self.file_source:
            specs = self._live_specs(source)
            if specs:
                self.selection = specs
                self.sender.paused = False
                self._apply_selection()
        self._mark_dirty()

    def save_config(self, path: str | None = None) -> str | None:
        """Write the configuration to ``path`` (or the known one, or ask). Returns an error or None."""
        if path is None:
            path = self.config_path
            if not path:
                from .dialogs import ask_save
                chosen = ask_save(initial=self.default_config_path())
                path = chosen if chosen is not None else self.default_config_path()
            if not path:
                return None                      # cancelled
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_config(), fh, indent=2)
        except OSError as exc:
            self.config_note = f"save failed: {exc}"
            self._mark_dirty()
            return self.config_note
        self.config_path = path
        self.config_note = f"saved {os.path.basename(path)}"
        self.log(f"\nconfiguration saved to {path}")
        self._mark_dirty()
        return None

    def load_config(self, path: str | None = None) -> str | None:
        """Read and apply a configuration file (``path``, or ask with a file dialog). Returns an error or None."""
        if path is None:
            from .dialogs import ask_open
            chosen = ask_open(initial=self.config_path or self.default_config_path())
            if chosen is None:                   # no tkinter: fall back to the default file
                chosen = self.config_path or self.default_config_path()
            path = chosen
            if not path:
                return None                      # cancelled
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("not a configuration object")
            self.apply_config(data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.config_note = f"load failed: {exc}"
            self.log(f"\nconfiguration {path!r}: {exc}")
            self._mark_dirty()
            return self.config_note
        self.config_path = path
        self.config_note = f"loaded {os.path.basename(path)}"
        self.log(f"\nconfiguration loaded from {path}")
        self._mark_dirty()
        return None

    def default_config_path(self) -> str:
        return os.path.join(os.path.dirname(self.cfg.templates_dir), "mediapipe_osc_config.json")

    def config_status(self) -> tuple[str, bool]:
        return self.config_note, self.config_note.endswith("failed") or "failed:" in self.config_note

    # ---- depth ---------------------------------------------------------------
    @property
    def depth_available(self) -> bool:
        """True when the running source can measure distances (a Kinect, or only Kinects side by side)."""
        return bool(getattr(self.source, "has_depth", False))

    @property
    def z_source(self) -> str:
        """What the z of the landmarks currently means: 'kinect' (metres) or 'mediapipe' (relative)."""
        return "kinect" if self.cfg.z_from_kinect and self.depth_available else "mediapipe"

    def set_z_from_kinect(self, enabled: bool) -> None:
        """The *Z from Kinect* toggle. Takes effect when the source has depth."""
        self.cfg.z_from_kinect = bool(enabled)
        self.source.want_depth = bool(enabled)
        self._mark_dirty()

    def _announce_z_source(self) -> None:
        """Tell the receivers when the meaning of z changes (called from the analysis thread)."""
        current = self.z_source
        if current != self._z_source_sent:
            self.sender.send(self.builder.z_source(current))
            self._z_source_sent = current

    # ====================================================================
    # OSC target / formats
    # ====================================================================
    def set_osc(self, ip: str | None = None, port: int | str | None = None) -> str | None:
        """Retarget the OSC output. Returns an error message, or None on success."""
        new_ip = self.sender.ip if ip is None else str(ip).strip()
        new_port = self.sender.port if port is None else port
        try:
            new_port = int(new_port)
        except (TypeError, ValueError):
            return "port must be a number"
        if not 1 <= new_port <= 65535:
            return "port must be 1..65535"
        if not new_ip:
            return "ip is empty"
        try:
            socket.getaddrinfo(new_ip, new_port, socket.AF_INET, socket.SOCK_DGRAM)
        except socket.gaierror:
            return "unknown host"
        if (new_ip, new_port) != (self.sender.ip, self.sender.port):
            self.cfg.ip, self.cfg.port = new_ip, new_port
            self._on_analysis_thread(lambda: self._apply_osc(new_ip, new_port))
        return None

    def _apply_osc(self, ip: str, port: int) -> None:
        # Let the old receiver reset before we move on.
        self.sender.send(self.builder.status("stopped"))
        self.sender.retarget(ip, port)
        self.sender.send(self.builder.status("started", self.z_source))
        self._z_source_sent = self.z_source
        self.log(f"\nOSC -> {ip}:{port}")

    def set_format(self, name: str, enabled: bool) -> None:
        formats = [f for f in self.cfg.formats if f != name]
        if enabled:
            formats.append(name)
        self.cfg.formats = tuple(formats)

    # ====================================================================
    # look & feel
    # ====================================================================
    def set_template(self, name: str) -> str | None:
        """Load a template by name/path. Returns an error message or None."""
        try:
            theme = load_theme(name, self.cfg.templates_dir)
        except Exception as exc:          # a broken template must not kill a running show
            self.log(f"\ntemplate {name!r}: {exc}")
            return str(exc)
        self.cfg.template = name
        self.theme = theme
        if self.panel is not None:
            self.panel.set_theme(theme)
        self.log(f"\nlook: {theme.name}")
        return None

    def next_template(self) -> None:
        names = available_templates(self.cfg.templates_dir)
        current = self.cfg.template
        if current not in names:
            stem = os.path.splitext(os.path.basename(current))[0]
            current = stem if stem in names else names[0]
        self.set_template(names[(names.index(current) + 1) % len(names)])

    def reload_template(self) -> None:
        self.set_template(self.cfg.template)

    def request_quit(self) -> None:
        self._quit = True

    # ====================================================================
    # one frame
    # ====================================================================
    def step(self) -> bool:
        """Grab, analyze, track, send and draw one frame. Returns False when the app should stop.

        This is the heart of the program; everything happens in this order:

        1. apply the settings changed from the panel since the last frame;
        2. read a frame (and mirror it if asked);
        3. run the detectors -> ``Detections`` (numpy arrays, no ids yet);
        4. if *Z from Kinect* is on, overwrite every z with the measured distance;
        5. match detections to the tracks of the previous frame -> ids, enter/leave, velocity;
        6. build the OSC messages of this frame and send them;
        7. draw the preview (only with a window).
        """
        cfg = self.cfg
        self._apply_pending()                                    # 1
        ok, frame = self.source.read()                           # 2
        if not ok or frame is None:
            return False
        depth = self.source.depth() if cfg.z_from_kinect else None
        # crop & mapping of every camera (slot = its columns in the composed frame)
        slots = self.slots(frame)
        editor = self.editor
        if editor is not None and editor.mode == "input":
            slot = next((sl for sl in slots if sl[0] == editor.spec), None)
            self._raw_slot = frame[:, slot[1]:slot[2]].copy() if slot else None
        if self.mappings:
            frame, depth = apply_mappings(frame, depth, slots, self.mappings)
        if self.mirror:
            frame = cv2.flip(frame, 1)
            if depth is not None:
                depth = cv2.flip(depth, 1)

        now = time.monotonic()
        # Video files use their own frame rate as clock, live sources the wall clock.
        t = self.source.timestamp() + self._time_offset
        self._last_t = t
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             # MediaPipe wants RGB, OpenCV gives BGR
        det = self.analyzer.analyze(rgb, int(t * 1000))          # 3
        if depth is not None and depth.shape[:2] == frame.shape[:2]:
            apply_depth(det, depth)                              # 4
        self._announce_z_source()

        poses = self.pose_tracker.update(det.poses, t)           # 5
        hands = self.hand_tracker.update(det.hands, t)
        faces = self.face_tracker.update(det.faces, t)
        blobs = self.blob_tracker.update(det.blobs, t)
        self.counts = (len(poses.active), len(hands.active), len(faces.active), len(blobs.active))

        fps = self.fps.tick(now)
        messages = self.builder.frame(                           # 6
            self.frame_index, t, fps,
            poses.active, poses.entered, poses.left,
            hands.active, hands.entered, hands.left,
            faces.active, faces.entered, faces.left,
            blobs.active, blobs.entered, blobs.left,
        )
        self.sender.send(messages)

        if not cfg.quiet and not cfg.verbose and self.frame_index % 30 == 0:
            self.log(f"\rframe {self.frame_index:6d}  {fps:5.1f} fps  "
                     f"poses={len(poses.active)} hands={len(hands.active)} faces={len(faces.active)} "
                     f"blobs={len(blobs.active)}  msgs={self.sender.sent}", end="", flush=True)

        if self.preview:                                         # 7
            theme = self.theme
            blob_det = self.analyzer.blob_detector
            mask = blob_det.mask if blob_det is not None else None
            frame = theme.draw_preview(frame, poses.active, hands.active, faces.active, blobs.active,
                                       self.show_ids, self.draw_landmarks, mask=mask,
                                       status="" if self.panel else self.status_line())
            if blob_det is not None and blob_det.learning:
                theme.draw_banner(frame, f"learning background {blob_det.learn_progress * 100:.0f}% "
                                         "- keep the scene empty")
            if editor is not None:
                frame = self._editor_view(frame, slots, editor)
            self._latest_frame = frame

        self.frame_index += 1
        if self._quit:
            return False
        if cfg.frames and self.frame_index >= cfg.frames:
            return False
        if cfg.max_fps > 0:
            budget = 1.0 / cfg.max_fps - (time.monotonic() - now)
            if budget > 0:
                time.sleep(budget)
        return True

    def status_line(self) -> str:
        p, h, f, b = self.counts
        return (f"{self.fps.fps:4.1f} fps  P:{p} H:{h} F:{f} B:{b}  -> {self.sender.ip}:{self.sender.port}"
                + ("  [mirror]" if self.mirror else "") + ("  [paused]" if self.sender.paused else "")
                + ("  [z: kinect]" if self.z_source == "kinect" else ""))

    def status_lines(self) -> list[str]:
        p, h, f, b = self.counts
        return [
            f"{self.fps.fps:4.1f} fps   frame {self.frame_index}",
            f"bodies {p}  hands {h}  faces {f}  blobs {b}",
            f"sent {self.sender.sent} msgs" + ("   PAUSED" if self.sender.paused else ""),
            f"-> {self.sender.ip}:{self.sender.port}   z: {self.z_source}",
            f"source: {self.source.name}",
        ]

    # ====================================================================
    # the window (main thread)
    # ====================================================================
    def _window_loop(self) -> None:
        """Main-thread loop: show the latest analyzed frame + panel, handle input.

        Keys are polled every few milliseconds (OpenCV only remembers the last
        key pressed between two waitKey calls); the image is re-rendered only
        when a new frame arrived, the panel changed, or for the cursor blink.
        """
        import numpy as np
        placeholder = np.zeros((self.cfg.height or 480, self.cfg.width or 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "waiting for the first frame...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        shown_frame = None
        last_render = 0.0
        last_key = 0.0
        while not self._quit:
            key = cv2.waitKey(5) if self._window_ready else -1
            if not self.handle_key(key):
                self._quit = True
                break
            if self._window_ready and self.window_closed():
                # The [x] of the window: imshow would silently re-create it.
                self.log("\nwindow closed")
                self._quit = True
                break
            now = time.monotonic()
            if key is not None and key >= 0:
                last_key = now
            frame = self._latest_frame
            panel_dirty = self.panel is not None and self.panel.dirty
            # Rendering is rate limited: the backend only remembers the last
            # key pressed between two waitKey calls, and imshow of a big frame
            # can block for tens of milliseconds, so a re-render on every key
            # press would lose keys while typing. While keys are coming in we
            # render at most ~7 times per second, otherwise ~30.
            min_gap = 0.15 if now - last_key < 0.3 else 0.03
            if now - last_render < min_gap or not (
                frame is not shown_frame or panel_dirty or now - last_render > 0.25
            ):
                continue
            shown_frame = frame
            image = placeholder if frame is None else frame
            try:
                if not self._window_ready:
                    # WINDOW_GUI_NORMAL: no Qt toolbar/status bar around the image
                    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
                    if self.panel is not None:
                        cv2.setMouseCallback(WINDOW, self.panel.on_mouse)
                    self._window_ready = True
                if self.panel is not None:
                    image = self.panel.render(image, self.status_lines())
                    self.panel.dirty = False
                # imshow on a window the user just closed would re-open it
                # (OpenCV's Qt GUI thread hides it on [x] and shows it again on
                # the next image), so look *immediately* before showing.
                if self.window_closed():
                    self.log("\nwindow closed")
                    self._quit = True
                    break
                cv2.imshow(WINDOW, image)
            except cv2.error as exc:
                # opencv-python-headless or no display: keep sending OSC without a window.
                print(f"\npreview disabled ({str(exc).splitlines()[0][:80]}...). "
                      "Use --no-preview to silence this, or install opencv-python (not headless).",
                      file=sys.stderr)
                self.preview = False
                self._worker.join()
                return
            last_render = now

    @staticmethod
    def window_closed() -> bool:
        """True once the user closed the preview window (its [x] button).

        Backends differ: Qt hides the window (``WND_PROP_VISIBLE`` becomes 0),
        GTK / Win32 destroy it (``getWindowProperty`` then raises). Both mean
        "closed" - this is only called once the window has been created.
        """
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _analysis_loop(self) -> None:
        try:
            while not self._quit and self.step():
                pass
        except BaseException as exc:  # reported from the main thread
            self._worker_error = exc
        finally:
            self._quit = True

    def handle_key(self, key: int) -> bool:
        """Keyboard shortcuts. False = quit.

        Priority: a focused text field gets every key; then the crop & mapping
        editor (while open); then the global shortcuts.
        """
        if key is None or key < 0:
            return True
        if self.panel is not None and self.panel.focused_field is not None:
            self.panel.on_key(key)
            return True
        code = key & 0xFF
        if self.editor is not None and code != 255:
            # the crop & mapping editor takes the keys (i/o, 1-4, arrows, r, Enter/Esc)
            if self.editor.key(code):
                if self.editor.done:
                    self.close_mapping_editor()
                self._mark_dirty()
                return True
        if self.panel is not None:
            result = self.panel.on_key(key)
            if result == "quit":
                return False
            if result == "handled":
                return True
        key = code
        if key in (27, ord("q")):
            return False
        elif key == ord("g"):
            # crop & mapping of the first camera (or of the video/image being played)
            spec = self.selection[0] if self.selection else getattr(self.source, "spec", "")
            if spec:
                self.open_mapping_editor(spec)
        elif key == ord("m"):
            self.mirror = not self.mirror
        elif key == ord("l"):
            self.draw_landmarks = not self.draw_landmarks
        elif key == ord("i"):
            self.show_ids = not self.show_ids
        elif key == ord("p"):
            self.set_mode("pause" if not self.sender.paused else
                          ("file" if self.source.kind in ("video", "image") else "camera"))
        elif key == ord("b"):
            self.learn_background()
        elif key == ord("t"):
            self.next_template()
        elif key == ord("r"):
            self.reload_template()
        elif key == ord("c"):
            self.scan_cameras()
        elif key == ord("z"):
            self.set_z_from_kinect(not self.cfg.z_from_kinect)
        self._mark_dirty()
        return True

    # ====================================================================
    # run / close
    # ====================================================================
    def run(self) -> int:
        """Start the threads, run until quit, clean up. Returns the exit code."""
        if self.cfg.config:
            self.load_config(self.cfg.config)
        self.sender.send(self.builder.status("started", self.z_source))
        self._z_source_sent = self.z_source
        try:
            if self.preview:
                self._worker = threading.Thread(target=self._analysis_loop, name="analysis", daemon=True)
                self._worker.start()
                self._window_loop()
            else:
                while self.step():
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            self._quit = True
            if self._worker is not None:
                self._worker.join(timeout=10.0)
            self.close()
        if self._worker_error is not None and not isinstance(self._worker_error, KeyboardInterrupt):
            raise self._worker_error
        return 0

    def close(self) -> None:
        """Release everything (idempotent): final OSC messages, cameras, models, window."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Tell receivers that everybody left, then that we are stopping
        # (even when the stream was paused: receivers must be able to reset).
        self.sender.paused = False
        try:
            self.sender.send(self.builder.leave_all(
                self.pose_tracker.reset(), self.hand_tracker.reset(), self.face_tracker.reset(),
                self.blob_tracker.reset()))
            self.sender.send(self.builder.status("stopped"))
        finally:
            self.source.release()
            self.analyzer.close()
            if self._window_ready:
                cv2.destroyAllWindows()
        self.log(f"\nstopped after {self.frame_index} frames, {self.sender.sent} OSC messages")


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``mediapipe_osc.py``: parse the options, build the App, run it."""
    cfg = parse_args(argv)
    if cfg.list_cameras:
        print("cameras found (use --source <spec>, join several with +):")
        print(format_camera_list(list_cameras(webcams=not cfg.no_webcams, kinects=not cfg.no_kinects)))
        return 0
    try:
        app = App(cfg)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return app.run()
