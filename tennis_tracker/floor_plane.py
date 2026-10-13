"""Experimental floor-plane prediction without measured floor dimensions.

Four corners describe an actual rectangle. An assumed pinhole lens supplies
orientation; a stationary tennis ball resting on it supplies scale. The known
ball size is a reference measurement, not the future landing location.
"""
from collections import deque
from dataclasses import replace

import cv2
import numpy as np

from tennis_tracker.calibration import CameraCalibration, solve_floor_pose
from tennis_tracker.metric_floor import MetricFloor
from tennis_tracker.physics import predict_physics


def assumed_camera(size, horizontal_fov):
    width, height = size
    focal = width / (2*np.tan(np.deg2rad(horizontal_fov)/2))
    matrix = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], float)
    return CameraCalibration(matrix, np.zeros(5), tuple(size), 0, "assumed-fov")


def unmeasured_rectangle_pose(camera, corners):
    """Infer rectangle aspect/orientation using the ASSUMED camera matrix."""
    unit = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], np.float32)
    homography = cv2.getPerspectiveTransform(unit, np.asarray(corners, np.float32))
    normalized = np.linalg.inv(camera.matrix) @ homography
    x_axis, y_axis = normalized[:, 0], normalized[:, 1]
    x_length, y_length = np.linalg.norm(x_axis), np.linalg.norm(y_axis)
    if min(x_length, y_length) < 1e-9:
        raise ValueError("Floor corners are too compressed; select a larger rectangle.")
    cosine = abs(float(x_axis@y_axis)/(x_length*y_length))
    aspect = y_length/x_length
    if cosine > 0.12 or not 0.1 < aspect < 15:
        raise ValueError("Rectangle/lens assumptions do not fit. Check corners or --plane-fov.")
    rotation, offset, error = solve_floor_pose(camera, corners, 1, aspect, min_distance=1e-4)
    return rotation, offset, float(aspect), error


class FloorPlane(MetricFloor):
    def __init__(self, fov=70, ball_diameter=0.067, setup_path=None, source_key=None):
        super().__init__(None, None, None, setup_path, source_key)
        self.fov = fov
        self.ball_diameter = ball_diameter
        self.reference_ready = False
        self.unit_offset = None
        self.unit_length = None
        self.reference_samples = deque(maxlen=15)
        self.status = "G: mark four corners of a real floor rectangle (no measurements)"

    @property
    def ready(self):
        return self.reference_ready and super().ready

    def _extra_setup_data(self):
        return {"prediction_mode": "plane", "fov": self.fov,
                "ball_diameter": self.ball_diameter}

    def _setup_compatible(self, data):
        return (data.get("prediction_mode") == "plane"
                and np.isclose(float(data.get("fov", 0)), self.fov)
                and np.isclose(float(data.get("ball_diameter", 0)), self.ball_diameter))

    def _pose_min_distance(self):
        return 0.005

    def configure_frame(self, width, height):
        size = (width, height)
        if self.size != size:
            changed = self.size is not None
            self.size = size
            self.camera = assumed_camera(size, self.fov)
            self.rotation = self.offset = None
            self.reference_ready = False
            self.unit_offset = None
            self.mode = None
            if changed:
                self.status = "Resolution changed; G to mark the floor in this view"
        if not self.setup_checked:
            self.setup_checked = True
            self._restore()
            if self.rotation is not None:
                self.reference_ready = True
                self.unit_offset = self.offset/self.width_m
                self.unit_length = self.length_m/self.width_m
                self.status = "Plane setup restored; keep camera fixed and check floor grid"

    def begin(self, mode="floor"):
        if self.size is None:
            self.status = "Wait for a camera frame first"
            return False
        self.mode = "floor"
        self.pending = []
        self.status = "Use a real rectangular floor patch, not arbitrary outline points"
        return True

    def click(self, x, y):
        if self.mode != "floor":
            return
        self.pending.append((x, y))
        if len(self.pending) < 4:
            return
        polygon = np.array(self.pending, np.int32)
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 500:
            self.pending = []
            self.status = "Invalid rectangle; retry the four corners in order"
            return
        try:
            rotation, offset, aspect, _ = unmeasured_rectangle_pose(self.camera, self.pending)
        except ValueError as error:
            self.status = str(error)
            self.pending = []
            return
        self.rotation, self.offset = rotation, offset
        self.unit_offset, self.unit_length = offset.copy(), aspect
        self.width_m, self.length_m = 1.0, aspect  # Temporary arbitrary units.
        self.corners = self.pending[:]
        self.reference_ready = False
        self.cancel()
        self.status = "Floor outlined. Rest ball INSIDE patch, press F, then let it settle"

    def begin_reference(self):
        if self.unit_offset is None or not self.corners:
            self.status = "Press G and finish marking the floor before F"
            return False
        self.mode = "reference"
        self.reference_samples.clear()
        self.status = "Ball must be RESTING on the floor, fully visible, not held in your hand"
        return True

    def observe_reference(self, track, timestamp):
        if self.mode != "reference":
            return False
        detection = track.detection
        if detection is None or detection.source != "YOLO":
            self.reference_samples.clear()
            self.status = "Reference: waiting for a clear YOLO ball detection inside the floor patch"
            return False
        x1, y1, x2, y2 = detection.box
        width, height = self.size
        box_width, box_height = x2-x1, y2-y1
        diameter = min(box_width, box_height)
        inside = cv2.pointPolygonTest(np.array(self.corners, np.float32),
                                     tuple(map(float, detection.center)), False) >= 0
        if (not inside or diameter < 8 or max(box_width, box_height)/max(diameter, 1) > 1.35
                or x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1):
            self.reference_samples.clear()
            self.status = "Reference needs a round, fully visible ball, at least 8px, inside the patch"
            return False
        self.reference_samples.append((timestamp, *detection.center, diameter))
        if len(self.reference_samples) < 12:
            self.status = f"Reference: keep ball resting still ({len(self.reference_samples)}/12)"
            return False
        samples = np.asarray(self.reference_samples)
        if np.ptp(samples[:, 0]) < 0.4:
            self.status = "Reference: collecting at least 0.4s of still observations"
            return False
        center = np.median(samples[:, 1:3], axis=0)
        diameter = float(np.median(samples[:, 3]))
        if (np.max(np.linalg.norm(samples[:, 1:3]-center, axis=1)) > max(3, diameter*0.1)
                or np.std(samples[:, 3])/diameter > 0.1):
            self.status = "Reference unstable: leave the ball still; use brighter lighting"
            return False
        # Approximate sphere depth: pixel diameter = focal_length * real_diameter / Z.
        depth = self.camera.matrix[0, 0]*self.ball_diameter/diameter
        ray = np.r_[self.camera.normalized([center])[0], 1.0]
        ball_center = ray*depth
        upward = self.rotation[:, 2]
        floor_distance = float(upward@ball_center-self.ball_diameter/2)
        unit_distance = float(upward@self.unit_offset)
        scale = floor_distance/unit_distance if abs(unit_distance) > 1e-9 else -1
        if not 0.005 < -floor_distance < 10 or not np.isfinite(scale) or scale <= 0:
            self.status = "Reference incompatible with the floor/lens assumptions; check outline and ball"
            return False
        self.offset = self.unit_offset*scale
        self.width_m, self.length_m = scale, self.unit_length*scale
        self.reference_ready = True
        self.cancel()
        self.status = "Approximate plane learned/saved. R before throw; keep camera fixed"
        self._save()
        return True

    def clear(self):
        self.reference_ready = False
        self.unit_offset = None
        self.reference_samples.clear()
        super().clear()
        self.status = "G: mark floor rectangle; F: resting-ball reference"

    def draw(self, frame):
        if self.rotation is not None:
            for fraction in np.linspace(0, 1, 5):
                x, y = fraction*self.width_m, fraction*self.length_m
                for points in ([(x, 0, 0), (x, self.length_m, 0)],
                               [(0, y, 0), (self.width_m, y, 0)]):
                    pixels = np.round(self.project(points)).astype(np.int32)
                    cv2.polylines(frame, [pixels], False, (170, 200, 110), 1, cv2.LINE_AA)
        for index, point in enumerate(self.pending):
            point = tuple(map(round, point))
            cv2.circle(frame, point, 5, (0, 255, 255), -1)
            cv2.putText(frame, str(index+1), (point[0]+8, point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        text = (f"Floor {len(self.pending)+1}/4: far-left, far-right, near-right, near-left"
                if self.mode == "floor" else "G: floor corners | F: resting-ball reference | R: fresh throw")
        cv2.putText(frame, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 150), 1)
        cv2.putText(frame, self.status, (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 150), 1)
        cv2.putText(frame, f"EXPERIMENTAL: assumed lens {self.fov:g}deg + ball size; no measured lens",
                    (10, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 200, 255), 1)


def predict_floor_plane(samples, now, floor, horizon=1.5,
                       motion_model="flight", size_observations=None):
    if not floor.ready:
        return None, "Waiting: G marks floor; F learns a resting-ball reference"
    landing, reason = predict_physics(samples, now, floor, horizon,
                                      radius=floor.ball_diameter/2,
                                      motion_model=motion_model, size_observations=size_observations)
    if landing is None:
        return None, reason
    normalized = (landing.floor_point[0]/floor.width_m,
                  landing.floor_point[1]/floor.length_m)
    return replace(landing, model="plane", floor_point=normalized), "Experimental floor-plane landing"
