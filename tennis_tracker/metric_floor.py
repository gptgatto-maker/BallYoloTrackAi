"""Measured ground plane for the gravity-based predictor; no contact guide."""
import json
from pathlib import Path

import cv2
import numpy as np

from tennis_tracker.calibration import (CameraCalibration, estimate_camera_from_rectangle,
                                        solve_floor_pose)


class MetricFloor:
    def __init__(self, camera, width_m, length_m, setup_path=None, source_key=None):
        self.camera = camera
        self.width_m = width_m
        self.length_m = length_m
        self.size = None
        self.rotation = None
        self.offset = None
        self.corners = []
        self.pending = []
        self.mode = None
        self.status = "G: mark measured floor rectangle"
        self.setup_path = Path(setup_path) if setup_path else None
        self.source_key = source_key
        self.setup_checked = False
        self.dimension_text = ""
        self.dimension_field = None

    @property
    def ready(self):
        return self.rotation is not None and self.mode is None

    def configure_frame(self, width, height):
        self.size = (width, height)
        if not self.setup_checked:
            self.setup_checked = True
            self._restore()
        if self.camera is not None and self.camera.size != self.size:
            self.rotation = None
            self.status = "Resolution changed; saved geometry cannot be reused at this size"
        elif (self.width_m is None or self.length_m is None) and self.mode != "dimensions":
            self.rotation = None
            self.status = "First setup: press G to enter floor dimensions, then mark corners"

    def _restore(self):
        """Restore the same source/resolution. Camera position must stay fixed."""
        if self.setup_path is None or not self.setup_path.exists():
            return
        try:
            data = json.loads(self.setup_path.read_text(encoding="utf-8"))
            if (data["version"] != 1 or data["source_key"] != self.source_key
                    or tuple(data["camera"]["image_size"]) != self.size
                    or not self._setup_compatible(data)):
                self.status = "Saved setup belongs to another source/resolution"
                return
            width, length = float(data["width_m"]), float(data["length_m"])
            if not np.isfinite([width, length]).all() or min(width, length) <= 0:
                raise ValueError("Invalid saved floor dimensions")
            if ((self.width_m is not None and not np.isclose(self.width_m, width))
                    or (self.length_m is not None and not np.isclose(self.length_m, length))):
                self.status = "Dimensions changed; G to mark the new rectangle"
                return
            saved_camera = CameraCalibration.from_data(data["camera"])
            if self.camera is not None and (not np.allclose(self.camera.matrix, saved_camera.matrix)
                    or self.camera.distortion.shape != saved_camera.distortion.shape
                    or not np.allclose(self.camera.distortion, saved_camera.distortion)):
                self.status = "Lens data changed; G to mark the floor again"
                return
            self.camera = saved_camera
            self.width_m, self.length_m = width, length
            if not data.get("valid", True):
                self.status = "Saved lens/dimensions loaded; G to mark the floor"
                return
            corners = np.asarray(data["corners"], float)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValueError("Invalid saved corners")
            rotation, offset, _ = solve_floor_pose(self.camera, corners, width, length,
                                                  min_distance=self._pose_min_distance())
            self.rotation, self.offset = rotation, offset
            self.corners = corners.tolist()
            self.status = "Saved setup loaded; keep camera fixed; check grid alignment"
            print(f"Restored floor setup from {self.setup_path}. Keep the camera in the same position.")
        except (ValueError, KeyError, TypeError, OSError, cv2.error) as error:
            self.rotation = None
            self.status = "Cannot load saved setup; G to mark the floor"
            print(f"Saved floor setup was not used: {error}")

    def _save(self, valid=True):
        if (self.setup_path is None or self.camera is None
                or self.width_m is None or self.length_m is None):
            return
        data = {"version": 1, "valid": valid, "source_key": self.source_key,
                "camera": self.camera.to_data(), "width_m": self.width_m,
                "length_m": self.length_m, "corners": self.corners}
        data.update(self._extra_setup_data())
        self.setup_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.setup_path.with_suffix(self.setup_path.suffix+".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.setup_path)
        print(f"Saved camera/floor setup to {self.setup_path}")

    def _extra_setup_data(self):
        return {}

    def _setup_compatible(self, data):
        return True

    def _pose_min_distance(self):
        return 0.05

    def begin(self, mode="floor"):
        if self.size is None:
            self.status = "Wait for the camera image before pressing G"
            print(self.status)
            return False
        if self.camera is not None and self.camera.size != self.size:
            self.status = "Cannot start: lens setup resolution differs from this stream"
            print(self.status)
            return False
        if self.width_m is None or self.length_m is None:
            self.mode = "dimensions"
            self.dimension_field = "width" if self.width_m is None else "length"
            self.dimension_text = ""
            self.pending = []
            self.status = "Measure a real floor rectangle; enter meters, not the landing location"
            print("First setup: type floor width/length in the video window, pressing Enter after each.")
            return True
        self.mode = "floor"
        self.pending = []
        self.status = "Click measured rectangle: far-left, far-right, near-right, near-left"
        return True

    def handle_dimension_key(self, key):
        """Collect one-time measurements without a blocking terminal prompt."""
        if self.mode != "dimensions" or key == 255:
            return
        if key == 27:
            self.cancel()
            self.status = "Setup cancelled; G to resume"
        elif key in (8, 127):
            self.dimension_text = self.dimension_text[:-1]
        elif key in (10, 13):
            try:
                value = float(self.dimension_text)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError
            except ValueError:
                self.status = "Enter a positive measurement in meters; Backspace edits"
                return
            if self.dimension_field == "width":
                self.width_m = value
                self.dimension_field = "length"
                self.dimension_text = ""
                self.status = "Now enter the rectangle length, near edge to far edge"
            else:
                self.length_m = value
                self.dimension_text = ""
                self.begin("floor")
        elif key in range(ord("0"), ord("9")+1) or key in (ord("."), ord(",")):
            self.dimension_text += "." if key == ord(",") else chr(key)

    def cancel(self):
        self.mode = None
        self.pending = []
        self.dimension_text = ""
        self.dimension_field = None

    def clear(self):
        self.cancel()
        self.rotation = self.offset = None
        self.corners = []
        self.status = "G: mark measured floor rectangle"
        self._save(valid=False)

    def click(self, x, y):
        if self.mode != "floor":
            return
        self.pending.append((x, y))
        if len(self.pending) != 4:
            return
        polygon = np.array(self.pending, np.int32)
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 500:
            self.status = "Invalid corners; retry far-left, far-right, near-right, near-left"
            self.pending = []
            return
        try:
            camera = self.camera
            if camera is None or camera.method == "floor-estimated":
                camera = estimate_camera_from_rectangle(
                    self.pending, self.size, self.width_m, self.length_m)
            rotation, offset, error = solve_floor_pose(
                camera, self.pending, self.width_m, self.length_m)
        except ValueError as error:
            self.status = str(error)
            self.pending = []
            return
        self.camera = camera
        self.rotation, self.offset = rotation, offset
        self.corners = self.pending[:]
        self.cancel()
        self.status = f"Setup saved ({error:.1f}px floor fit). R: reset before each throw"
        self._save()

    def project(self, world_points):
        camera_points = np.asarray(world_points, float) @ self.rotation.T + self.offset
        return self.camera.camera_pixels(camera_points)

    def draw(self, frame):
        if self.rotation is not None:
            # Curved polylines include the calibrated lens distortion.
            for fraction in np.linspace(0, 1, 5):
                x = fraction * self.width_m
                y = fraction * self.length_m
                for points in ([(x, t, 0) for t in np.linspace(0, self.length_m, 20)],
                               [(t, y, 0) for t in np.linspace(0, self.width_m, 20)]):
                    pixels = self.project(points)
                    cv2.polylines(frame, [np.round(pixels).astype(np.int32)], False,
                                  (170, 200, 110), 1, cv2.LINE_AA)
        for index, point in enumerate(self.pending):
            cv2.circle(frame, point, 5, (0, 255, 255), -1)
            cv2.putText(frame, str(index+1), (point[0]+8, point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self.mode == "dimensions":
            name = "WIDTH (left to right)" if self.dimension_field == "width" else "LENGTH (near to far)"
            instructions = f"Type {name} in meters: {self.dimension_text}_ | Enter confirms"
        elif self.mode == "floor":
            instructions = f"Click {len(self.pending)+1}/4: far-left, far-right, near-right, near-left"
        else:
            instructions = "G: calibrate floor | C: clear floor | R: fresh throw"
        cv2.putText(frame, instructions, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 220, 150), 1)
        cv2.putText(frame, self.status, (10, 102), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (255, 220, 150), 1)
        if self.camera is not None and self.camera.method == "floor-estimated":
            cv2.putText(frame, "Approximate lens: centered camera / zero distortion assumed",
                        (10, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 200, 255), 1)
