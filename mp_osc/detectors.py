"""The detectors: MediaPipe Tasks (pose, hands, gestures, face) + the OpenCV Blobs detector.

What this module does
---------------------
:class:`Analyzer` owns the detectors that are switched on and runs them on
every frame (``analyze()``), returning a :class:`Detections` object: four
lists (``poses``, ``hands``, ``faces``, ``blobs``) of small dataclasses that
hold **plain numpy arrays**. Converting MediaPipe's result objects to arrays
right here means nothing else in the app depends on MediaPipe's classes - the
tracker, the OSC builder, the drawing and the tests all work on arrays.

MediaPipe Tasks in two sentences
--------------------------------
A *Task* (``PoseLandmarker``, ``HandLandmarker``, ``GestureRecognizer``,
``FaceLandmarker``) is created from a ``.task`` model file (they are in the
repository folder) with options such as how many people to look for. In
``VIDEO`` running mode the task expects frames with strictly increasing
timestamps in milliseconds and uses them for its own temporal tracking - that
is why ``analyze()`` takes ``timestamp_ms`` and bumps it when the caller
repeats one.

The dataclasses
---------------
``PoseDetection.landmarks`` is a ``(33, 5)`` array of ``x y z visibility
presence``; ``HandDetection.landmarks`` ``(21, 3)``; ``FaceDetection.landmarks``
``(478, 3)``; ``BlobDetection.landmarks`` ``(16, 2)`` contour points. All have a
``center`` property (where the thing is, for the tracker and the messages)
and a ``bbox()`` method. The ``Blobs`` detector (``mp_osc.blobs``) needs no
model at all.

Coordinates: ``x`` and ``y`` are normalized to the image (0..1, origin at the
top-left corner). ``z`` is MediaPipe's relative depth (roughly the same scale
as x, smaller = closer to the camera); with *Z from Kinect* the app
overwrites it with metres (see ``mp_osc.depth``). World landmarks are in
metres, centred on the hips / wrist.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .blobs import BlobDetection, BlobDetector
from .config import Config


POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"
GESTURE_MODEL = "gesture_recognizer.task"

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
FACE_LANDMARK_COUNT = 478
BLENDSHAPE_COUNT = 52

# Indices used to summarize a detection with a single point.
POSE_NOSE = 0
POSE_LEFT_HIP, POSE_RIGHT_HIP = 23, 24
HAND_WRIST = 0
HAND_THUMB_TIP, HAND_INDEX_TIP = 4, 8
FACE_NOSE_TIP = 1


@dataclass
class PoseDetection:
    landmarks: np.ndarray            # (33, 5): x y z visibility presence
    world: np.ndarray                # (33, 3): meters, hip centered

    # Landmarks with visibility below VIS_LOW get no weight in the center, full
    # weight from VIS_HIGH on; in between the weight ramps linearly (continuous,
    # so the center does not jump when a limb appears). Coordinates are clipped
    # to CLIP_MARGIN around the image because MediaPipe extrapolates hidden
    # limbs several screen heights outside the frame.
    VIS_LOW, VIS_HIGH = 0.3, 0.7
    CLIP_MARGIN = 0.25
    # Core body points that define "where the person is" (limbs are excluded:
    # a swinging arm should not move the person).
    CORE = [POSE_NOSE, 11, 12, POSE_LEFT_HIP, POSE_RIGHT_HIP]  # nose, shoulders, hips

    def _weighted_mean(self, idx) -> tuple[np.ndarray, float]:
        lms = self.landmarks[idx]
        pts = lms[:, :3].copy()
        pts[:, :2] = np.clip(pts[:, :2], -self.CLIP_MARGIN, 1.0 + self.CLIP_MARGIN)
        w = np.clip((lms[:, 3:4] - self.VIS_LOW) / (self.VIS_HIGH - self.VIS_LOW), 0.0, 1.0)
        total = float(w.sum())
        if total < 1e-6:
            return pts.mean(axis=0), 0.0
        return (pts * w).sum(axis=0) / total, total

    @property
    def center(self) -> np.ndarray:
        """Visibility-weighted mean of nose, shoulders and hips (x, y, z).

        Falls back to the whole body when none of those is visible. Unlike the
        hip midpoint this stays inside the image when people are seen from the
        waist up, and it moves smoothly as body parts appear/disappear.
        """
        center, weight = self._weighted_mean(self.CORE)
        if weight < 1e-6:
            center, _ = self._weighted_mean(slice(None))
        return center

    @property
    def hips(self) -> np.ndarray:
        """Hip midpoint (x, y, z); may fall outside the image."""
        return self.landmarks[[POSE_LEFT_HIP, POSE_RIGHT_HIP], :3].mean(axis=0)

    def bbox(self, min_visibility: float = 0.5) -> np.ndarray:
        """(xmin, ymin, xmax, ymax) of the visible landmarks."""
        pts = self.landmarks[self.landmarks[:, 3] >= min_visibility, :2]
        if len(pts) == 0:
            pts = self.landmarks[:, :2]
        return np.concatenate([pts.min(axis=0), pts.max(axis=0)])


@dataclass
class HandDetection:
    landmarks: np.ndarray            # (21, 3)
    world: np.ndarray                # (21, 3) meters, wrist centered
    handedness: str                  # "Left" / "Right" (as seen by the camera)
    handedness_score: float
    gesture: str = ""                # only with the gesture recognizer
    gesture_score: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return self.landmarks.mean(axis=0)

    @property
    def pinch(self) -> float:
        """Thumb tip - index tip distance in (approximate) meters."""
        return float(np.linalg.norm(self.world[HAND_THUMB_TIP] - self.world[HAND_INDEX_TIP]))

    def bbox(self) -> np.ndarray:
        pts = self.landmarks[:, :2]
        return np.concatenate([pts.min(axis=0), pts.max(axis=0)])


@dataclass
class FaceDetection:
    landmarks: np.ndarray            # (478, 3)
    blendshapes: np.ndarray          # (52,) 0..1, order in BLENDSHAPE_NAMES

    @property
    def center(self) -> np.ndarray:
        return self.landmarks[FACE_NOSE_TIP]

    def bbox(self) -> np.ndarray:
        pts = self.landmarks[:, :2]
        return np.concatenate([pts.min(axis=0), pts.max(axis=0)])


@dataclass
class Detections:
    poses: list[PoseDetection] = field(default_factory=list)
    hands: list[HandDetection] = field(default_factory=list)
    faces: list[FaceDetection] = field(default_factory=list)
    blobs: list[BlobDetection] = field(default_factory=list)


# Blendshape order produced by face_landmarker.task (filled in on first detection,
# this is the documented order of the model).
BLENDSHAPE_NAMES: list[str] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


def _landmarks_to_array(landmarks, with_visibility: bool = False) -> np.ndarray:
    if with_visibility:
        return np.array(
            [(lm.x, lm.y, lm.z, lm.visibility or 0.0, lm.presence or 0.0) for lm in landmarks],
            dtype=np.float32,
        )
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)


class Analyzer:
    """Creates the requested detectors and runs them on each frame.

    Detectors can be switched on and off at runtime with ``configure()``;
    landmarkers are created lazily (loading a model takes about a second)
    and closed when no longer needed. Only one thread may use an Analyzer:
    the MediaPipe tasks are not thread-safe, which is why the app queues
    ``configure()`` calls to its analysis thread.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_ts = -1
        self._pose = self._hand = self._face = self._gesture = None
        self._blobs: BlobDetector | None = None
        self.configure(cfg.pose, cfg.hands, cfg.face, cfg.gesture, cfg.blobs)

    # ------------------------------------------------------------------
    def configure(self, pose: bool, hands: bool, face: bool, gesture: bool, blobs: bool = False) -> None:
        """Create / close detectors so that exactly the requested ones run."""
        cfg = self.cfg
        conf = cfg.min_confidence
        mode = vision.RunningMode.VIDEO
        hands = hands or gesture

        if blobs and self._blobs is None:
            self._blobs = BlobDetector(
                mode=cfg.blob_mode, threshold=cfg.blob_threshold, min_area=cfg.blob_min_area,
                max_area=cfg.blob_max_area, max_blobs=cfg.blob_max, blur=cfg.blob_blur,
                morph=cfg.blob_morph, learn_frames=cfg.blob_learn_frames, adapt=cfg.blob_adapt,
                roi=cfg.blob_roi,
            )
        elif not blobs and self._blobs is not None:
            self._blobs = None

        if pose and self._pose is None:
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=self._base(POSE_MODEL),
                    running_mode=mode,
                    num_poses=cfg.max_poses,
                    min_pose_detection_confidence=conf,
                    min_pose_presence_confidence=conf,
                    min_tracking_confidence=conf,
                )
            )
        elif not pose and self._pose is not None:
            self._pose.close()
            self._pose = None

        # The gesture recognizer also outputs hand landmarks, so it replaces
        # the plain hand landmarker when gestures are wanted.
        want_gesture = hands and gesture
        want_hand = hands and not gesture
        if want_gesture and self._gesture is None:
            self._gesture = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(
                    base_options=self._base(GESTURE_MODEL),
                    running_mode=mode,
                    num_hands=cfg.max_hands,
                    min_hand_detection_confidence=conf,
                    min_hand_presence_confidence=conf,
                    min_tracking_confidence=conf,
                )
            )
        elif not want_gesture and self._gesture is not None:
            self._gesture.close()
            self._gesture = None
        if want_hand and self._hand is None:
            self._hand = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=self._base(HAND_MODEL),
                    running_mode=mode,
                    num_hands=cfg.max_hands,
                    min_hand_detection_confidence=conf,
                    min_hand_presence_confidence=conf,
                    min_tracking_confidence=conf,
                )
            )
        elif not want_hand and self._hand is not None:
            self._hand.close()
            self._hand = None

        if face and self._face is None:
            self._face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=self._base(FACE_MODEL),
                    running_mode=mode,
                    num_faces=cfg.max_faces,
                    min_face_detection_confidence=conf,
                    min_face_presence_confidence=conf,
                    min_tracking_confidence=conf,
                    output_face_blendshapes=cfg.send_blendshapes,
                )
            )
        elif not face and self._face is not None:
            self._face.close()
            self._face = None

    @property
    def pose_enabled(self) -> bool:
        return self._pose is not None

    @property
    def hands_enabled(self) -> bool:
        return self._hand is not None or self._gesture is not None

    @property
    def gesture_enabled(self) -> bool:
        return self._gesture is not None

    @property
    def face_enabled(self) -> bool:
        return self._face is not None

    @property
    def blobs_enabled(self) -> bool:
        return self._blobs is not None

    @property
    def blob_detector(self) -> BlobDetector | None:
        return self._blobs

    def _base(self, model_name: str) -> mp_python.BaseOptions:
        path = self.cfg.model_path(model_name)
        if not os.path.exists(path):
            raise RuntimeError(
                f"model file not found: {path}. Download it from "
                "https://developers.google.com/mediapipe/solutions or pass --models-dir."
            )
        return mp_python.BaseOptions(model_asset_path=path)

    def analyze(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Detections:
        """Run all enabled detectors on one RGB frame (height x width x 3, uint8).

        ``timestamp_ms`` must grow from call to call (VIDEO mode); equal or
        smaller values are bumped by one millisecond.
        """
        # VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        det = Detections()

        if self._blobs is not None:
            det.blobs = self._blobs.process(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))

        if self._pose is None and self._gesture is None and self._hand is None and self._face is None:
            return det
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._pose is not None:
            result = self._pose.detect_for_video(image, timestamp_ms)
            for lms, world in zip(result.pose_landmarks, result.pose_world_landmarks):
                det.poses.append(PoseDetection(
                    landmarks=_landmarks_to_array(lms, with_visibility=True),
                    world=_landmarks_to_array(world),
                ))

        if self._gesture is not None:
            result = self._gesture.recognize_for_video(image, timestamp_ms)
            for i, lms in enumerate(result.hand_landmarks):
                handed = result.handedness[i][0]
                gesture = result.gestures[i][0] if i < len(result.gestures) and result.gestures[i] else None
                det.hands.append(HandDetection(
                    landmarks=_landmarks_to_array(lms),
                    world=_landmarks_to_array(result.hand_world_landmarks[i]),
                    handedness=handed.category_name,
                    handedness_score=float(handed.score),
                    gesture=gesture.category_name if gesture else "",
                    gesture_score=float(gesture.score) if gesture else 0.0,
                ))
        elif self._hand is not None:
            result = self._hand.detect_for_video(image, timestamp_ms)
            for i, lms in enumerate(result.hand_landmarks):
                handed = result.handedness[i][0]
                det.hands.append(HandDetection(
                    landmarks=_landmarks_to_array(lms),
                    world=_landmarks_to_array(result.hand_world_landmarks[i]),
                    handedness=handed.category_name,
                    handedness_score=float(handed.score),
                ))

        if self._face is not None:
            result = self._face.detect_for_video(image, timestamp_ms)
            for i, lms in enumerate(result.face_landmarks):
                if result.face_blendshapes:
                    shapes = result.face_blendshapes[i]
                    blend = np.array([c.score for c in shapes], dtype=np.float32)
                    if len(BLENDSHAPE_NAMES) != len(shapes) or BLENDSHAPE_NAMES[1] != shapes[1].category_name:
                        BLENDSHAPE_NAMES[:] = [c.category_name for c in shapes]
                else:
                    blend = np.zeros(BLENDSHAPE_COUNT, dtype=np.float32)
                det.faces.append(FaceDetection(landmarks=_landmarks_to_array(lms), blendshapes=blend))

        return det

    def close(self) -> None:
        for d in (self._pose, self._hand, self._face, self._gesture):
            if d is not None:
                d.close()
