"""Replay a short drop with fast color frames, first-locking, and CSV decisions."""
from contextlib import ExitStack
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main as application
from tennis_tracker.detection import Detection
from tennis_tracker.floor_plane import FloorPlane
from tennis_tracker.prediction import draw_prediction
from test_floor_plane import fov_for, mark_plane, learn_plane
from test_physics import scene, GRAVITY


class DropApplicationTests(unittest.TestCase):
    def test_short_drop_locks_before_impact_and_persists_with_diagnostics(self):
        reference = scene()
        ground_target = reference.project([[1.1, 1, 0]])[0]
        impact = np.sqrt(2*(0.5-0.0335)/GRAVITY)
        rendered = []
        wait_count = 0
        current_frame = 0
        primary_calls = 0
        detections = []
        release_frame = 30  # Allow held-size agreement after R before a short fall.

        def detect(frame):
            nonlocal primary_calls
            primary_calls += 1
            return detections[current_frame]

        def show(window, frame):
            pass

        def wait(delay):
            nonlocal wait_count
            wait_count += 1
            return ord("r") if wait_count == 10 else -1

        def draw(frame, landing, reason, **kwargs):
            draw_prediction(frame, landing, reason, **kwargs)
            if landing is not None:
                rendered.append((current_frame, landing))

        with tempfile.TemporaryDirectory() as directory:
            video = str(Path(directory)/"drop.avi")
            profile = Path(directory)/"plane.json"
            log = Path(directory)/"drop-diagnostics.csv"
            key = hashlib.sha256(video.encode("utf-8")).hexdigest()[:16]
            plane = FloorPlane(fov_for(reference), setup_path=profile, source_key=key)
            mark_plane(reference, plane)
            learn_plane(reference, plane)
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
            self.assertTrue(writer.isOpened())
            for i in range(80):
                frame = np.zeros((720, 1280, 3), np.uint8)
                boxes = []
                if i < release_frame+10:
                    t = max(0, (i-release_frame)/30)
                    world = np.array([1.1, 1, 0.5-0.5*GRAVITY*t*t])
                    x, y = reference.project([world])[0]
                    camera_point = reference.rotation@world+reference.offset
                    diameter = reference.camera.matrix[0, 0]*0.067/camera_point[2]
                    boxes = [Detection((x-diameter/2, y-diameter/2, x+diameter/2, y+diameter/2), 0.9)]
                    cv2.circle(frame, (round(x), round(y)), round(diameter/2), (0, 255, 210), -1)
                detections.append(boxes)
                writer.write(frame)
            writer.release()
            argv = ["main.py", "--stage", "6", "--source", video, "--prediction-mode", "plane",
                    "--motion-model", "drop", "--plane-fov", str(fov_for(reference)),
                    "--setup-file", str(profile), "--gate", "250", "--detect-every", "3",
                    "--diagnostics", str(log)]
            original_read = application.VideoSource.read

            def read(source):
                nonlocal current_frame
                current_frame = source.index
                return original_read(source)

            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                detector = stack.enter_context(patch("tennis_tracker.detection.BallDetector"))
                detector.return_value.detect.side_effect = detect
                stack.enter_context(patch.object(application.VideoSource, "read", read))
                stack.enter_context(patch("tennis_tracker.prediction.draw_prediction", side_effect=draw))
                stack.enter_context(patch("cv2.namedWindow"))
                stack.enter_context(patch("cv2.setMouseCallback"))
                stack.enter_context(patch("cv2.imshow", side_effect=show))
                stack.enter_context(patch("cv2.waitKey", side_effect=wait))
                stack.enter_context(patch("cv2.getWindowProperty", return_value=1))
                stack.enter_context(patch("cv2.destroyAllWindows"))
                application.run(application.parse_args())
            with log.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        decisions = [(round(float(row["time_s"]), 2), row["detector"], row["candidate"],
                      row["candidate_span_s"], row["uncertainty_m"], row["uncertainty_px"], row["reason"])
                     for row in rows if (release_frame-2)/30 <= float(row["time_s"]) <= (release_frame+10)/30]
        self.assertTrue(rendered, f"Short drop never produced a locked target: {decisions}")
        first_frame, first = rendered[0]
        self.assertLess((first_frame-release_frame)/30, impact)
        self.assertTrue(first.first_locked)
        self.assertEqual(first.motion_model, "drop")
        self.assertLess(np.linalg.norm(np.array([first.x, first.y])-ground_target), 12)
        self.assertEqual((rendered[-1][1].x, rendered[-1][1].y), (first.x, first.y))
        self.assertEqual(rendered[-1][0], 79)
        self.assertLess(primary_calls, 60)
        self.assertTrue(any(row["detector"] == "color recovery" for row in rows))
        self.assertTrue(any(row["first_locked"] == "True" for row in rows))
