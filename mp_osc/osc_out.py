"""OSC message schema and sending.

Three output formats can be enabled at the same time (``--format``):

``mp`` (default, numeric arguments; ``<id>`` is a stable tracking id)::

    /mp/status            s:"started"|"stopped"  s:version
    /mp/z_source          s:"mediapipe"|"kinect"  what the z of landmarks means (at start and on change)
    /mp/frame             i:frame  f:time_s  f:fps  i:n_poses  i:n_hands  i:n_faces  i:n_blobs
    /mp/pose/enter        i:id
    /mp/pose/leave        i:id
    /mp/pose/center       i:id  f:x f:y f:z             (visibility-weighted mean of the body)
    /mp/pose/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
    /mp/pose/velocity     i:id  f:vx f:vy f:speed       (normalized units per second)
    /mp/pose/landmarks    i:id  33 x (f:x f:y f:z f:visibility)
    /mp/pose/world        i:id  33 x (f:x f:y f:z)      meters, only with --world
    /mp/hand/enter|leave  i:id
    /mp/hand/info         i:id  s:handedness f:score
    /mp/hand/center       i:id  f:x f:y f:z
    /mp/hand/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
    /mp/hand/velocity     i:id  f:vx f:vy f:speed
    /mp/hand/pinch        i:id  f:distance              (thumb tip - index tip, meters)
    /mp/hand/gesture      i:id  s:name f:score          only with --gesture
    /mp/hand/landmarks    i:id  21 x (f:x f:y f:z)
    /mp/face/enter|leave  i:id
    /mp/face/center       i:id  f:x f:y f:z             (nose tip)
    /mp/face/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
    /mp/face/velocity     i:id  f:vx f:vy f:speed
    /mp/face/blendshapes  i:id  52 x f:value            order: /mp/face/blendshape_names
    /mp/face/blendshape_names  52 x s:name              sent at start-up
    /mp/face/landmarks    i:id  478 x (f:x f:y f:z)     only with --face-landmarks
    /mp/blob/enter|leave  i:id                            (--blobs: OpenCV shapes)
    /mp/blob/center       i:id  f:x f:y f:0
    /mp/blob/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
    /mp/blob/velocity     i:id  f:vx f:vy f:speed
    /mp/blob/info         i:id  f:area f:angle f:w f:h   area = fraction of the image, angle in degrees
    /mp/blob/contour      i:id  16 x (f:x f:y)           outline, 16 points from the topmost one
    /mp/blobs             i:count  count x i:id          ids of the blobs on screen, biggest first

With ``--id-style address`` the id moves into the address:
``/mp/pose/3/center f:x f:y f:z`` (enter/leave keep the id as argument).

``legacy`` reproduces the string messages of the original ``ejemplo-*.py``
scripts so the Processing sketches in this repository keep working.

``kinect`` sends ``/kinect/person i:id f:x f:y f:speed`` as expected by the
``Sonido_Cuadrafonico`` scripts (a webcam replaces the Kinect + Processing).
It is derived from the blobs when the Blobs detector is on (a ceiling camera,
like the original Kinect sketch), otherwise from the body poses
(``--kinect-from`` overrides this).
"""

from __future__ import annotations

import numbers
from typing import Any, Sequence

import numpy as np
from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

from . import __version__
from .config import Config
from .detectors import BLENDSHAPE_NAMES
from .tracking import Track

Message = tuple[str, list[Any]]


def _f(values) -> list[float]:
    return [float(v) for v in values]


class MessageBuilder:
    """Turns tracker output into a list of ``(address, args)`` tuples."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prefix = cfg.prefix

    # ----------------------------------------------------------- helpers
    def _addr(self, kind: str, what: str, track_id: int | None = None) -> tuple[str, list[Any]]:
        """Return (address, leading args) honouring the id style."""
        if track_id is None:
            return f"{self.prefix}/{kind}/{what}", []
        if self.cfg.id_style == "address":
            return f"{self.prefix}/{kind}/{track_id}/{what}", []
        return f"{self.prefix}/{kind}/{what}", [int(track_id)]

    def _msg(self, kind: str, what: str, track_id: int | None, args: Sequence[Any]) -> Message:
        addr, lead = self._addr(kind, what, track_id)
        return addr, lead + list(args)

    # ----------------------------------------------------------- status
    def status(self, text: str, z_source: str | None = None) -> list[Message]:
        """``/mp/status started|stopped`` (+ blendshape names and the z source at start-up)."""
        out: list[Message] = []
        if "mp" in self.cfg.formats:
            out.append((f"{self.prefix}/status", [text, __version__]))
            if text == "started":
                if self.cfg.face and self.cfg.send_blendshapes:
                    out.append((f"{self.prefix}/face/blendshape_names", list(BLENDSHAPE_NAMES)))
                if z_source:
                    out += self.z_source(z_source)
        return out

    def z_source(self, name: str) -> list[Message]:
        """``/mp/z_source mediapipe|kinect``: whether z is MediaPipe's relative depth or metres."""
        if "mp" not in self.cfg.formats:
            return []
        return [(f"{self.prefix}/z_source", [name])]

    # ----------------------------------------------------------- per frame
    def frame(
        self,
        frame_index: int,
        time_s: float,
        fps: float,
        poses: list[Track], pose_enter: list[int], pose_leave: list[int],
        hands: list[Track], hand_enter: list[int], hand_leave: list[int],
        faces: list[Track], face_enter: list[int], face_leave: list[int],
        blobs: Sequence[Track] = (), blob_enter: Sequence[int] = (), blob_leave: Sequence[int] = (),
    ) -> list[Message]:
        out: list[Message] = []
        blobs = list(blobs)
        if "mp" in self.cfg.formats:
            out += self._mp_frame(frame_index, time_s, fps, poses, pose_enter, pose_leave,
                                  hands, hand_enter, hand_leave, faces, face_enter, face_leave,
                                  blobs, list(blob_enter), list(blob_leave))
        if "legacy" in self.cfg.formats:
            out += self._legacy(poses, hands, faces)
        if "kinect" in self.cfg.formats:
            out += self._kinect(blobs if self.cfg.kinect_source_is_blobs else poses)
        return out

    def leave_all(self, pose_ids: list[int], hand_ids: list[int], face_ids: list[int],
                  blob_ids: Sequence[int] = ()) -> list[Message]:
        out: list[Message] = []
        if "mp" in self.cfg.formats:
            out += [self._msg("pose", "leave", None, [i]) for i in pose_ids]
            out += [self._msg("hand", "leave", None, [i]) for i in hand_ids]
            out += [self._msg("face", "leave", None, [i]) for i in face_ids]
            out += [self._msg("blob", "leave", None, [i]) for i in blob_ids]
        return out

    # ----------------------------------------------------------- 'mp' format
    def _common(self, kind: str, track: Track) -> list[Message]:
        c = track.center
        box = track.detection.bbox()
        return [
            self._msg(kind, "center", track.id, _f(c[:3])),
            self._msg(kind, "bbox", track.id, _f(box)),
            self._msg(kind, "velocity", track.id, _f(track.velocity[:2]) + [track.speed]),
        ]

    def _mp_frame(self, frame_index, time_s, fps, poses, pose_enter, pose_leave,
                  hands, hand_enter, hand_leave, faces, face_enter, face_leave,
                  blobs, blob_enter, blob_leave) -> list[Message]:
        cfg = self.cfg
        out: list[Message] = [
            (f"{self.prefix}/frame",
             [int(frame_index), float(time_s), float(fps), len(poses), len(hands), len(faces), len(blobs)])
        ]

        out += [self._msg("pose", "enter", None, [i]) for i in pose_enter]
        for t in poses:
            out += self._common("pose", t)
            if cfg.send_landmarks:
                out.append(self._msg("pose", "landmarks", t.id, _f(t.landmarks[:, :4].ravel())))
            if cfg.send_world:
                out.append(self._msg("pose", "world", t.id, _f(t.detection.world.ravel())))
        out += [self._msg("pose", "leave", None, [i]) for i in pose_leave]

        out += [self._msg("hand", "enter", None, [i]) for i in hand_enter]
        for t in hands:
            d = t.detection
            out.append(self._msg("hand", "info", t.id, [d.handedness, float(d.handedness_score)]))
            out += self._common("hand", t)
            out.append(self._msg("hand", "pinch", t.id, [d.pinch]))
            if cfg.gesture:
                out.append(self._msg("hand", "gesture", t.id, [d.gesture, float(d.gesture_score)]))
            if cfg.send_landmarks:
                out.append(self._msg("hand", "landmarks", t.id, _f(t.landmarks[:, :3].ravel())))
        out += [self._msg("hand", "leave", None, [i]) for i in hand_leave]

        out += [self._msg("face", "enter", None, [i]) for i in face_enter]
        for t in faces:
            out += self._common("face", t)
            if cfg.send_blendshapes:
                out.append(self._msg("face", "blendshapes", t.id, _f(t.detection.blendshapes)))
            if cfg.send_face_landmarks:
                out.append(self._msg("face", "landmarks", t.id, _f(t.landmarks[:, :3].ravel())))
        out += [self._msg("face", "leave", None, [i]) for i in face_leave]

        out += [self._msg("blob", "enter", None, [i]) for i in blob_enter]
        if blobs or blob_enter or blob_leave or cfg.blobs:
            # biggest first: the detector sorts by area and the tracker keeps the
            # detection order in ``Track.detection``; ids here are sorted by size.
            by_size = sorted(blobs, key=lambda t: -t.detection.area)
            out.append((f"{self.prefix}/blobs", [len(blobs)] + [int(t.id) for t in by_size]))
        for t in blobs:
            d = t.detection
            out += self._common("blob", t)
            out.append(self._msg("blob", "info", t.id, [float(d.area), float(d.angle), float(d.size[0]), float(d.size[1])]))
            if cfg.send_landmarks:
                out.append(self._msg("blob", "contour", t.id, _f(t.landmarks[:, :2].ravel())))
        out += [self._msg("blob", "leave", None, [i]) for i in blob_leave]
        return out

    # ----------------------------------------------------------- 'legacy' format
    @staticmethod
    def _legacy(poses: list[Track], hands: list[Track], faces: list[Track]) -> list[Message]:
        """Same strings as ejemplo-pose-landmark-osc.py / ejemplo-hand-landmarks.py /
        ejemplo-face-landmark-osc.py / ejemplo-gesture-recognizer-osc.py."""
        out: list[Message] = []
        for idx, t in enumerate(poses):
            args = ["{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(*lm[:5]) for lm in t.landmarks]
            out.append((f"pose_{idx}", args))
        for idx, t in enumerate(hands):
            args = [f"{idx} {i} {lm[0]:.3f} {lm[1]:.3f} {lm[2]:.3f}" for i, lm in enumerate(t.landmarks)]
            out.append(("hand_landmarks", args))
            out.append(("distancia_pulgar_indice", [f"{idx} {t.detection.pinch}"]))
            if t.detection.gesture:
                out.append(("/gesture", [f"{t.detection.gesture} {t.detection.gesture_score:.2f}"]))
        for idx, t in enumerate(faces):
            args = ["{:.3f} {:.3f} {:.3f}".format(*lm[:3]) for lm in t.landmarks]
            out.append((f"face_{idx}", args))
        return out

    # ----------------------------------------------------------- 'kinect' format
    def _kinect(self, tracks: list[Track]) -> list[Message]:
        """/kinect/person id x y speed, all floats in 0..1 (see Sonido_Cuadrafonico).

        ``tracks`` are body poses or blobs; both have a center and a bbox.
        """
        cfg = self.cfg
        out: list[Message] = []
        for t in tracks:
            x = float(t.center[0])
            if cfg.invert_x:
                x = 1.0 - x
            if cfg.kinect_y == "size":
                # Frontal camera: a taller silhouette is closer to the camera
                # (front of the room = small y).
                box = t.detection.bbox()
                height = float(box[3] - box[1])
                y = 1.0 - height
            else:
                y = float(t.center[1])
            speed = t.speed / cfg.max_speed if cfg.max_speed > 0 else 0.0
            out.append(("/kinect/person", [int(t.id), _clip01(x), _clip01(y), _clip01(speed)]))
        return out


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


# ---------------------------------------------------------------------------
def encode(address: str, args: Sequence[Any]) -> osc_message_builder.OscMessage:
    """Build a python-osc message with explicit argument types."""
    mb = osc_message_builder.OscMessageBuilder(address=address)
    for a in args:
        if isinstance(a, (bool, np.bool_)):
            mb.add_arg(bool(a), arg_type="T" if a else "F")
        elif isinstance(a, numbers.Integral):
            mb.add_arg(int(a), arg_type="i")
        elif isinstance(a, numbers.Real):
            mb.add_arg(float(a), arg_type="f")
        elif isinstance(a, str):
            mb.add_arg(a, arg_type="s")
        else:
            mb.add_arg(a)
    return mb.build()


class OscSender:
    """Sends ``(address, args)`` messages over UDP, individually or bundled."""

    def __init__(self, ip: str, port: int, bundle: bool = False, verbose: bool = False):
        self.bundle = bundle
        self.verbose = verbose
        self.sent = 0
        self.paused = False
        self.retarget(ip, port)

    def retarget(self, ip: str, port: int) -> None:
        """Point the sender at another receiver (can be called while running)."""
        self._client = udp_client.SimpleUDPClient(ip, int(port))
        self.ip, self.port = ip, int(port)

    def send(self, messages: list[Message]) -> None:
        if not messages or self.paused:
            return
        if self.verbose:
            for addr, args in messages:
                print(addr, " ".join(_fmt(a) for a in args))
        if self.bundle:
            bb = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            for addr, args in messages:
                bb.add_content(encode(addr, args))
            self._client.send(bb.build())
        else:
            for addr, args in messages:
                self._client.send(encode(addr, args))
        self.sent += len(messages)


def _fmt(a: Any) -> str:
    return f"{a:.3f}" if isinstance(a, float) else str(a)
