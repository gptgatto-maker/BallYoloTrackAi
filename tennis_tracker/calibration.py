"""Lens calibration and projection helpers shared by setup and prediction."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    matrix: np.ndarray
    distortion: np.ndarray
    size: tuple
    rms: float = 0.0
    method: str = "checkerboard"

    @classmethod
    def load(cls, filename):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data):
        camera = cls(np.asarray(data["camera_matrix"], float),
                     np.asarray(data["distortion"], float),
                     tuple(data["image_size"]), float(data["rms_pixels"]),
                     str(data.get("method", "checkerboard")))
        if (camera.matrix.shape != (3, 3) or len(camera.size) != 2
                or camera.distortion.size not in (4, 5, 8, 12, 14)
                or not np.isfinite(camera.matrix).all()
                or not np.isfinite(camera.distortion).all()
                or not np.isfinite(camera.rms) or camera.rms < 0 or camera.rms > 1.5
                or any(not isinstance(n, int) or n <= 0 for n in camera.size)
                or camera.matrix[0, 0] <= 0 or camera.matrix[1, 1] <= 0
                or not np.allclose(camera.matrix[2], [0, 0, 1])):
            raise ValueError("Invalid or inaccurate camera calibration. Recalibrate the lens.")
        return camera

    def to_data(self):
        return {"camera_matrix": self.matrix.tolist(),
                "distortion": self.distortion.tolist(),
                "image_size": list(self.size), "rms_pixels": self.rms,
                "method": self.method}

    def save(self, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_data(), indent=2), encoding="utf-8")

    def normalized(self, image_points):
        return cv2.undistortPoints(np.asarray(image_points, float).reshape(-1, 1, 2),
                                   self.matrix, self.distortion).reshape(-1, 2)

    def camera_pixels(self, camera_points):
        points, _ = cv2.projectPoints(np.asarray(camera_points, float).reshape(-1, 3),
                                      np.zeros(3), np.zeros(3),
                                      self.matrix, self.distortion)
        return points.reshape(-1, 2)


def solve_floor_pose(camera, corners, floor_width, floor_length, min_distance=0.05):
    """Find camera pose from a measured floor rectangle, independent of landing."""
    # Origin is near-left. X goes right, Y goes away from the camera; this
    # ordering makes +Z point above the floor in a right-handed world frame.
    objects = np.array([(0, floor_length, 0), (floor_width, floor_length, 0),
                        (floor_width, 0, 0), (0, 0, 0)], float)
    images = np.asarray(corners, float)
    ok, rotations, translations, _ = cv2.solvePnPGeneric(
        objects, images, camera.matrix, camera.distortion, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("Cannot solve floor pose. Mark a larger floor rectangle.")
    candidates = []
    for rvec, tvec in zip(rotations, translations):
        rotation = cv2.Rodrigues(rvec)[0]
        offset = tvec.reshape(3)
        position = -rotation.T @ offset
        camera_points = objects @ rotation.T + offset
        if position[2] <= min_distance or np.any(camera_points[:, 2] <= min_distance):
            continue
        error = float(np.sqrt(np.mean(np.sum(
            (camera.camera_pixels(camera_points) - images)**2, axis=1))))
        candidates.append((error, rotation, offset))
    if not candidates:
        raise ValueError("Invalid floor orientation. Click far-left, far-right, near-right, near-left.")
    error, rotation, offset = min(candidates, key=lambda item: item[0])
    if error > 3:
        raise ValueError(f"Floor fit error {error:.1f}px: check dimensions and corner clicks.")
    return rotation, offset, error


def estimate_camera_from_rectangle(corners, size, floor_width, floor_length):
    """No-print approximate intrinsics from a known floor rectangle.

    Assumptions: square pixels, principal point at image center, zero distortion.
    Orthogonal equally scaled world axes constrain the focal length. A nearly
    front-on rectangle provides too little perspective information; reject it.
    """
    width, height = size
    world = np.array([(0, floor_length), (floor_width, floor_length),
                      (floor_width, 0), (0, 0)], np.float32)

    def focal_length(points):
        homography = cv2.getPerspectiveTransform(world, np.asarray(points, np.float32))
        h1, h2 = homography[:, 0], homography[:, 1]
        center = np.array([width/2, height/2])
        first = h1[:2] - center*h1[2]
        second = h2[:2] - center*h2[2]
        coefficients = np.array([first @ second,
                                 first @ first - second @ second])
        constant = np.array([h1[2]*h2[2], h1[2]**2 - h2[2]**2])
        denominator = coefficients @ coefficients
        if denominator < 1e-12:
            return None
        inverse_squared_focal = -float(coefficients @ constant) / denominator
        if inverse_squared_focal <= 0:
            return None
        focal = 1/np.sqrt(inverse_squared_focal)
        return float(focal) if 0.35*width < focal < 4*width else None

    focal = focal_length(corners)
    if focal is None:
        raise ValueError("Lens estimate failed: check rectangle dimensions and camera perspective.")
    # Estimate sensitivity to small click errors. This is not lens-distortion
    # validation; the centered, undistorted-camera assumptions still apply.
    random = np.random.default_rng(7)
    trials = [focal_length(np.asarray(corners) + random.normal(0, 1.5, (4, 2)))
              for _ in range(24)]
    good = [value for value in trials if value is not None]
    if len(good) < 20 or np.std(good)/focal > 0.15:
        raise ValueError("Lens estimate unstable: use a larger rectangle seen at an oblique angle.")
    matrix = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], float)
    return CameraCalibration(matrix, np.zeros(5), tuple(size), 0.0, "floor-estimated")
