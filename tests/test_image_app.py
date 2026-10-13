"""Check default 2D mode through capture, tracking, prediction, and blue rendering."""
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


class ImageApplicationTests(unittest.TestCase):
    def test_default_2d_mode_predicts_known_crossing_and_handles_missing_frames(self):
        rendered = []
        index = 0

        def detect(frame):
            nonlocal index
            t = index/30
            missing = index in (12, 13) or index >= 19
            index += 1
            if missing:
                return []
            x, y = 100+50*t, 100+100*t*t
            return [Detection((x-4, y-4, x+4, y+4), 0.9)]

        def draw(frame, landing, reason, **kwargs):
            draw_prediction(frame, landing, reason, **kwargs)
            if landing is not None:
                rendered.append(landing)
                x, y = round(landing.x), round(landing.y)
                self.assertGreater(int(frame[y, x, 0]), int(frame[y, x, 2]))

        with tempfile.TemporaryDirectory() as folder:
            video = str(Path(folder)/"arc.avi")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
            self.assertTrue(writer.isOpened())
            for _ in range(100):
                writer.write(np.zeros((480, 640, 3), np.uint8))
            writer.release()
            argv = ["main.py", "--stage", "6", "--source", video, "--headless",
                    "--ground-y", str(200/479), "--no-color-recovery"]
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                detector = stack.enter_context(patch("tennis_tracker.detection.BallDetector"))
                detector.return_value.detect.side_effect = detect
                stack.enter_context(patch("tennis_tracker.prediction.draw_prediction", side_effect=draw))
                # Default mode must ignore stored metric calibration entirely.
                stack.enter_context(patch("tennis_tracker.calibration.CameraCalibration.load",
                                          side_effect=AssertionError("2D loaded calibration")))
                args = application.parse_args()
                self.assertEqual(args.prediction_mode, "image")
                application.run(args)
        self.assertTrue(rendered)
        self.assertTrue(any(point.held for point in rendered))
        self.assertGreater(len(rendered), 60)
        self.assertTrue(all(point.first_locked and point.retained for point in rendered))
        self.assertTrue(all(point.model == "image" for point in rendered))
        self.assertAlmostEqual(rendered[-1].x, 150)
        self.assertAlmostEqual(rendered[-1].y, 200)
        self.assertEqual(rendered[-1].seconds, 0)
