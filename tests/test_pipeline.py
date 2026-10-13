"""Exercise detection association -> history -> prediction -> rendered target."""
import unittest

import numpy as np

from tennis_tracker.detection import Detection
from tennis_tracker.prediction import predict_landing, draw_prediction
from tennis_tracker.tracking import BallTracker
from tennis_tracker.trajectory import Trajectory


class PipelineTests(unittest.TestCase):
    def test_observed_arc_with_missing_frames(self):
        tracker = BallTracker(confirm_hits=1)
        trajectory = Trajectory()
        for index in range(16):
            t = index / 30
            x, y = 100 + 50 * t, 100 + 100 * t * t
            detections = [] if index in (9, 10) else [
                Detection((x - 3, y - 3, x + 3, y + 3), 0.8)]
            track = tracker.update(detections, t)
            trajectory.update(track, t)
        landing, reason = predict_landing(trajectory.measurements(), t, 200, 640)
        self.assertIsNotNone(landing)
        self.assertAlmostEqual(landing.x, 150)
        frame = np.zeros((480, 640, 3), np.uint8)
        trajectory.draw(frame)
        draw_prediction(frame, landing, reason, 200)
        self.assertTrue(np.any(frame))
        trajectory.update(tracker.update([], 1.0), 1.0)
        self.assertEqual(trajectory.measurements(), [])
