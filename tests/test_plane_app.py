"""Exercise the unmeasured G/F workflow through the real staged capture loop."""
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
from test_floor_plane import fov_for, reference_detection
from test_physics import scene, observations


class PlaneApplicationTests(unittest.TestCase):
    def test_g_floor_f_reference_then_unknown_landing_is_rendered(self):
        reference = scene()
        samples, expected, impact_time = observations(reference, last=0.5, count=16)
        corners = reference.project([(0, 4, 0), (3, 4, 0), (3, 0, 0), (0, 0, 0)])
        resting = reference_detection(reference)
        index = 0
        wait_count = 0
        callback = None
        rendered = []

        def detect(frame):
            nonlocal index
            current = index
            index += 1
            if current < 17:
                return [resting]
            _, x, y = samples[current-17]
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
                return ord("f")
            if wait_count == 17:
                return ord("r")
            return -1

        def draw(frame, landing, reason, **kwargs):
            draw_prediction(frame, landing, reason, **kwargs)
            if landing is not None:
                rendered.append(landing)
                x, y = round(landing.x), round(landing.y)
                self.assertGreater(int(frame[y, x, 0]), int(frame[y, x, 2]))

        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"reference-throw.avi")
            profile = Path(folder)/"plane.json"
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
            self.assertTrue(writer.isOpened())
            for _ in range(33):
                writer.write(np.zeros((720, 1280, 3), np.uint8))
            writer.release()
            argv = ["main.py", "--stage", "6", "--source", path,
                    "--prediction-mode", "plane", "--plane-fov", str(fov_for(reference)),
                    "--setup-file", str(profile), "--gate", "250", "--no-color-recovery"]
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
                args = application.parse_args()
                self.assertIsNone(args.floor_width)
                self.assertIsNone(args.floor_length)
                application.run(args)
            self.assertTrue(profile.exists())
        self.assertTrue(rendered, "Plane workflow did not produce a bullseye")
        self.assertEqual(rendered[-1].model, "plane")
        np.testing.assert_allclose(rendered[-1].floor_point,
                                   expected/np.array([3, 4]), atol=1e-4)
        self.assertAlmostEqual(rendered[-1].seconds, impact_time-0.5, places=4)
