import unittest

import cv2
import numpy as np

from tennis_tracker.detection import TennisColorRecovery, looks_tennis_yellow


class ColorRecoveryTests(unittest.TestCase):
    def test_finds_yellow_blur_near_prediction(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.ellipse(frame, (170, 100), (18, 5), 15, 0, 360, (0, 255, 210), -1)
        found = TennisColorRecovery().detect_near(frame, (160, 100), 60)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].source, "color recovery")

    def test_ignores_same_color_outside_search_area(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(frame, (280, 200), 10, (0, 255, 210), -1)
        self.assertEqual(TennisColorRecovery().detect_near(
            frame, (40, 40), 50), [])

    def test_ignores_white_background(self):
        frame = np.full((240, 320, 3), 255, np.uint8)
        self.assertEqual(TennisColorRecovery().detect_near(
            frame, (160, 120), 100), [])

    def test_yellow_validation_rejects_wall_and_accepts_ball(self):
        frame = np.full((100, 100, 3), 220, np.uint8)
        self.assertFalse(looks_tennis_yellow(frame, (20, 20, 60, 60)))
        cv2.circle(frame, (40, 40), 12, (0, 255, 210), -1)
        self.assertTrue(looks_tennis_yellow(frame, (25, 25, 55, 55)))


if __name__ == "__main__":
    unittest.main()
