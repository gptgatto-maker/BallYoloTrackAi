import unittest

from tennis_tracker.detection import Detection
from tennis_tracker.tracking import Track
from tennis_tracker.trajectory import Trajectory


class TrajectoryTests(unittest.TestCase):
    def test_coasting_does_not_add_measurements_and_history_expires(self):
        history = Trajectory(seconds=0.5)
        history.update(Track((10, 20), Detection((8, 18, 12, 22), 0.9), True), 0)
        history.update(Track((11, 20), None), 0.1)
        self.assertEqual(len(history.measurements()), 1)
        history.update(Track((12, 20), None), 0.6)
        self.assertEqual(history.measurements(), [])

    def test_new_track_clears_previous_flight(self):
        history = Trajectory()
        for t in [0, 0.1]:
            history.update(Track((10, 20), Detection((8, 18, 12, 22), 0.9), True), t)
        self.assertEqual(len(history.measurements()), 1)
        history.update(Track(None, None), 0.2)
        self.assertEqual(history.measurements(), [])
