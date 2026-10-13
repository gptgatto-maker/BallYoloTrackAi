import unittest

from tennis_tracker.detection import Detection
from tennis_tracker.tracking import BallTracker


def ball(x, y=100, confidence=0.9):
    return Detection((x - 3, y - 3, x + 3, y + 3), confidence)


class TrackingTests(unittest.TestCase):
    def test_occlusion_expiry_and_reacquisition(self):
        tracker = BallTracker(max_gap=0.4, confirm_hits=1)
        self.assertTrue(tracker.update([ball(100)], 0).new_track)
        self.assertIsNotNone(tracker.update([ball(105)], 0.05).detection)
        coast = tracker.update([], 0.1)
        self.assertIsNotNone(coast.position)
        self.assertIsNone(coast.detection)
        self.assertIsNone(tracker.update([], 0.6).position)
        self.assertTrue(tracker.update([ball(400)], 0.65).new_track)

    def test_association_rejects_distant_high_confidence_ball(self):
        tracker = BallTracker(gate=50, confirm_hits=1)
        tracker.update([ball(100)], 0)
        selected = tracker.update([ball(102, confidence=0.4), ball(400)], 0.03)
        self.assertEqual(selected.detection.center[0], 102)
        self.assertIsNone(tracker.update([ball(400)], 0.06).detection)

    def test_new_track_requires_two_consistent_detections(self):
        tracker = BallTracker(gate=50)
        self.assertIsNone(tracker.update([ball(100)], 0).position)
        self.assertIsNone(tracker.update([ball(300)], 0.05).position)
        self.assertTrue(tracker.update([ball(302)], 0.1).new_track)


if __name__ == "__main__":
    unittest.main()
