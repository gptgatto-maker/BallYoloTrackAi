"""One-ball constant-velocity Kalman filter with short occlusion handling."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Track:
    position: tuple | None
    detection: object | None
    new_track: bool = False


class BallTracker:
    def __init__(self, max_gap=0.4, gate=100.0, confirm_hits=2):
        self.max_gap = max_gap
        self.gate = gate
        self.kalman = None
        self.last_time = None
        self.last_seen = None
        self.confirm_hits = confirm_hits
        self.pending = None
        self.pending_hits = 0
        self.pending_time = None

    def reset(self):
        self.kalman = None
        self.last_time = None
        self.last_seen = None
        self.pending = None
        self.pending_hits = 0
        self.pending_time = None

    @property
    def active(self):
        return self.kalman is not None

    def expected_position(self, timestamp):
        """Estimate the next center without advancing the Kalman filter."""
        if self.kalman is None:
            return None
        state = self.kalman.statePost[:, 0]
        dt = max(timestamp - self.last_time, 0.0)
        return (float(state[0] + state[2] * dt),
                float(state[1] + state[3] * dt))

    def _start(self, detection, timestamp):
        k = cv2.KalmanFilter(4, 2)
        # State: x, y, horizontal velocity, vertical velocity (pixels/second).
        k.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        k.measurementNoiseCov = np.eye(2, dtype=np.float32) * 9
        k.errorCovPost = np.diag([25, 25, 10000, 10000]).astype(np.float32)
        k.statePost = np.array([*detection.center, 0, 0], np.float32).reshape(4, 1)
        self.kalman = k
        self.last_time = self.last_seen = timestamp
        return Track(detection.center, detection, new_track=True)

    def update(self, detections, timestamp):
        # Expire BEFORE association: never join a new ball to a stale trajectory.
        if self.kalman is not None and timestamp - self.last_seen > self.max_gap:
            self.reset()
        if self.kalman is None:
            if not detections:
                self.pending = None
                self.pending_hits = 0
                return Track(None, None)
            candidate = max(detections, key=lambda d: d.confidence)
            close = (self.pending is not None
                     and np.linalg.norm(np.array(candidate.center)
                                        - np.array(self.pending.center)) <= self.gate
                     and timestamp - self.pending_time <= 0.3)
            self.pending_hits = self.pending_hits + 1 if close else 1
            self.pending = candidate
            self.pending_time = timestamp
            if self.pending_hits < self.confirm_hits:
                return Track(None, None)
            return self._start(candidate, timestamp)

        dt = max(timestamp - self.last_time, 0.001)
        self.last_time = timestamp
        k = self.kalman
        k.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                                       [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # Acceleration uncertainty lets the filter adapt to changing ball motion.
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]], np.float32)
        k.processNoiseCov = (g @ g.T) * 800**2
        predicted = k.predict()
        center = predicted[:2, 0]
        nearest = min(detections, key=lambda d: np.linalg.norm(np.array(d.center) - center),
                      default=None)
        # Reject implausible jumps to other balls/false positives.
        if nearest is not None and np.linalg.norm(np.array(nearest.center) - center) <= self.gate:
            corrected = k.correct(np.array(nearest.center, np.float32).reshape(2, 1))
            self.last_seen = timestamp
            return Track(tuple(map(float, corrected[:2, 0])), nearest)
        return Track(tuple(map(float, center)), None)
