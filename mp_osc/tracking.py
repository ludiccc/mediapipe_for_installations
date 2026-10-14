"""Stable IDs, enter/leave events, velocity and smoothing for detections.

MediaPipe returns detections in an arbitrary order every frame and does not
tell you which person/hand/face is which. ``Tracker`` matches detections to
the tracks of the previous frame by nearest center, so an installation can
follow "person 3" over time, be told when somebody enters or leaves, and get a
velocity for each of them.

The matching is greedy nearest-neighbour with a distance threshold. It is
simple and predictable, which is what you want in an installation with a
handful of people; it is not a multi-object tracker for crowds.

How one ``update()`` goes:

1. compute the distance between every live track and every new detection
   (centres only, x and y);
2. walk the pairs from the closest up: a track and a detection that are both
   still free and closer than ``max_distance`` are matched;
3. unmatched detections become new tracks (``entered``); matched tracks get
   their centre, landmarks (optionally smoothed) and velocity updated;
4. tracks not seen for ``timeout`` seconds are dropped (``left``).

The tracker works for any detection object with ``center`` and ``landmarks``
(poses, hands, faces, blobs all use the same class).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Track:
    id: int
    center: np.ndarray                 # (x, y, z) normalized
    detection: Any                     # latest detection object
    first_seen: float
    last_seen: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    landmarks: np.ndarray | None = None   # smoothed landmarks (same shape as detection.landmarks)
    frames: int = 0

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class TrackerUpdate:
    active: list[Track]      # tracks seen in this frame, sorted by id
    entered: list[int]       # ids created in this frame
    left: list[int]          # ids removed in this frame (timeout)


class Tracker:
    def __init__(
        self,
        max_distance: float = 0.25,
        timeout: float = 0.5,
        smooth: float = 0.0,
        velocity_smooth: float = 0.6,
        first_id: int = 0,
    ):
        self.max_distance = float(max_distance)
        self.timeout = float(timeout)
        self.smooth = float(smooth)
        self.velocity_smooth = float(velocity_smooth)
        self._ids = itertools.count(first_id)
        self._tracks: dict[int, Track] = {}

    # ------------------------------------------------------------------
    @property
    def tracks(self) -> dict[int, Track]:
        """All live tracks (including ones not seen in the last frame)."""
        return self._tracks

    def reset(self) -> list[int]:
        """Drop every track; returns the ids that were alive (to send 'leave')."""
        ids = sorted(self._tracks)
        self._tracks.clear()
        return ids

    # ------------------------------------------------------------------
    def update(self, detections: list[Any], now: float) -> TrackerUpdate:
        """Match ``detections`` (objects with ``.center`` and ``.landmarks``) to tracks."""
        centers = [np.asarray(d.center, dtype=np.float32) for d in detections]
        assignment = self._match(centers)

        entered: list[int] = []
        active: list[Track] = []

        for det_idx, det in enumerate(detections):
            track_id = assignment.get(det_idx)
            if track_id is None:
                track_id = next(self._ids)
                track = Track(
                    id=track_id,
                    center=centers[det_idx],
                    detection=det,
                    first_seen=now,
                    last_seen=now,
                    landmarks=np.array(det.landmarks, dtype=np.float32),
                )
                self._tracks[track_id] = track
                entered.append(track_id)
            else:
                track = self._tracks[track_id]
                self._advance(track, det, centers[det_idx], now)
            track.frames += 1
            active.append(track)

        left = [tid for tid, t in self._tracks.items() if now - t.last_seen > self.timeout]
        for tid in left:
            del self._tracks[tid]

        active.sort(key=lambda t: t.id)
        return TrackerUpdate(active=active, entered=entered, left=sorted(left))

    # ------------------------------------------------------------------
    def _advance(self, track: Track, det: Any, center: np.ndarray, now: float) -> None:
        dt = now - track.last_seen
        if dt > 1e-4:
            instant = (center[:2] - track.center[:2]) / dt
            a = self.velocity_smooth
            track.velocity = (a * track.velocity + (1.0 - a) * instant).astype(np.float32)
        new_lms = np.asarray(det.landmarks, dtype=np.float32)
        if self.smooth > 0.0 and track.landmarks is not None and track.landmarks.shape == new_lms.shape:
            track.landmarks = (self.smooth * track.landmarks + (1.0 - self.smooth) * new_lms).astype(np.float32)
            smoothed_center = self.smooth * track.center + (1.0 - self.smooth) * center
            track.center = smoothed_center.astype(np.float32)
        else:
            track.landmarks = new_lms
            track.center = center
        track.detection = det
        track.last_seen = now

    def _match(self, centers: list[np.ndarray]) -> dict[int, int]:
        """Greedy nearest-neighbour matching: detection index -> track id."""
        if not centers or not self._tracks:
            return {}
        track_ids = list(self._tracks)
        track_centers = np.stack([self._tracks[t].center[:2] for t in track_ids])
        det_centers = np.stack([c[:2] for c in centers])
        # distance matrix (tracks x detections)
        dist = np.linalg.norm(track_centers[:, None, :] - det_centers[None, :, :], axis=2)

        pairs = sorted(
            ((dist[i, j], i, j) for i in range(len(track_ids)) for j in range(len(centers))),
            key=lambda p: p[0],
        )
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        assignment: dict[int, int] = {}
        for d, i, j in pairs:
            if d > self.max_distance:
                break
            if i in used_tracks or j in used_dets:
                continue
            used_tracks.add(i)
            used_dets.add(j)
            assignment[j] = track_ids[i]
        return assignment
