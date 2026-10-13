"""Check the full staged loop with file input, mocked detections, and floor clicks."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main as application
from tennis_tracker.detection import Detection
from tennis_tracker.prediction import draw_prediction
from test_physics import scene, observations


class PhysicsApplicationTests(unittest.TestCase):
    def test_calibration_mouse_events_lead_to_rendered_unknown_landing(self):
        floor = scene()
        samples, expected, _ = observations(floor, last=0.5, count=16)
        corners = floor.project([(0, 4, 0), (3, 4, 0), (3, 0, 0), (0, 0, 0)])
        detected = iter(samples)
        rendered = []
        callback = None
        wait_count = 0

        def detect(frame):
            _, x, y = next(detected)
            return [Detection((x-4, y-4, x+4, y+4), 0.9)]

        def register(window, handler):
            nonlocal callback
            callback = handler

        def wait(delay):
            nonlocal wait_count
            wait_count += 1
            if wait_count == 1:
                return ord("g")
            if wait_count == 2:
                for x, y in corners:
                    callback(cv2.EVENT_LBUTTONDOWN, float(x), float(y), 0, None)
            return -1

        def draw(frame, landing, reason, **kwargs):
            draw_prediction(frame, landing, reason, **kwargs)
            if landing is not None:
                rendered.append(landing)
                x, y = round(landing.x), round(landing.y)
                self.assertGreater(int(frame[y, x, 0]), int(frame[y, x, 2]))

        with tempfile.TemporaryDirectory() as folder:
            calibration_path = Path(folder)/"camera.json"
            floor.camera.save(calibration_path)
            video_path = str(Path(folder)/"flight.avi")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
            self.assertTrue(writer.isOpened())
            for _ in samples:
                writer.write(np.zeros((720, 1280, 3), np.uint8))
            writer.release()
            argv = ["main.py", "--stage", "6", "--prediction-mode", "physics", "--source", video_path,
                    "--camera-calibration", str(calibration_path),
                    "--setup-file", str(Path(folder)/"setup.json"),
                    "--floor-width", "3", "--floor-length", "4", "--gate", "250"]
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                detector = stack.enter_context(patch("tennis_tracker.detection.BallDetector"))
                detector.return_value.detect.side_effect = detect
                stack.enter_context(patch("tennis_tracker.prediction.draw_prediction", side_effect=draw))
                stack.enter_context(patch("cv2.namedWindow"))
                stack.enter_context(patch("cv2.setMouseCallback", side_effect=register))
                stack.enter_context(patch("cv2.imshow"))
                stack.enter_context(patch("cv2.waitKey", side_effect=wait))
                stack.enter_context(patch("cv2.getWindowProperty", return_value=1))
                stack.enter_context(patch("cv2.destroyAllWindows"))
                application.run(application.parse_args())
        self.assertTrue(rendered, "App never rendered the physics landing")
        self.assertEqual(rendered[-1].model, "physics")
        np.testing.assert_allclose(rendered[-1].floor_point, expected, atol=1e-4)
