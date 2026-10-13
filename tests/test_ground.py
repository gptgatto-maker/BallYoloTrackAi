import unittest

import numpy as np

from tennis_tracker.ground import GroundMap
from tennis_tracker.prediction import predict_landing


def mapped_floor():
    ground = GroundMap()
    ground.begin("floor")
    for point in [(100, 180), (500, 160), (630, 450), (10, 470)]:
        ground.click(*point)
    return ground


class GroundTests(unittest.TestCase):
    def test_single_click_sets_a_2d_ground_row(self):
        ground = GroundMap()
        ground.begin("row")
        ground.click(250, 300)
        self.assertIsNone(ground.mode)
        self.assertEqual(ground.endpoints(640, 480), [(0.0, 300.0), (639.0, 300.0)])

    def test_floor_projects_to_corners_and_sloping_guide(self):
        ground = mapped_floor()
        np.testing.assert_allclose(ground.project([(0, 0), (1, 1)]),
                                   [(100, 180), (630, 450)], atol=1e-4)
        first, second = ground.endpoints(640, 480)
        self.assertNotAlmostEqual(first[1], second[1])
        self.assertTrue(ground.contains(320, 300))
        self.assertFalse(ground.contains(320, 40))

    def test_sloping_contact_intersection_uses_x_and_y(self):
        ground = mapped_floor()
        ground.begin("guide")
        ground.click(100, 200)
        ground.click(300, 300)  # y_ground(x) = 0.5*x + 150
        # x=100+50*t, y=100+100*t^2; compare the solved intersection.
        samples = [(float(t), 100+50*t, 100+100*t*t)
                   for t in np.linspace(0, 0.5, 16)]
        landing, _ = predict_landing(samples, 0.5, None, 640,
                                     ground=ground, height=480)
        self.assertIsNotNone(landing)
        self.assertAlmostEqual(landing.y, 0.5*landing.x+150)
        self.assertTrue(ground.contains(landing.x, landing.y))

    def test_rejects_contact_outside_mapped_region(self):
        ground = mapped_floor()
        ground.begin("guide")
        ground.click(0, 100)
        ground.click(639, 100)
        samples = [(float(t), 320, 20+100*t*t)
                   for t in np.linspace(0, 0.5, 16)]
        self.assertIsNone(predict_landing(samples, 0.5, None, 640,
                                         ground=ground, height=480)[0])

    def test_calibration_blocks_predictions_and_rejects_crossed_corners(self):
        ground = GroundMap()
        ground.begin("floor")
        for point in [(0, 0), (200, 200), (200, 0), (0, 200)]:
            ground.click(*point)
        self.assertEqual(ground.corners, [])
        samples = [(float(t), 100, 100+100*t*t)
                   for t in np.linspace(0, 0.5, 16)]
        self.assertIsNone(predict_landing(samples, 0.5, None, 640,
                                         ground=ground, height=480)[0])

    def test_depth_adjustment_moves_guide_and_clear_restores_default(self):
        ground = mapped_floor()
        before = ground.endpoints(640, 480)[0][1]
        ground.adjust_depth(0.1)
        self.assertGreater(ground.endpoints(640, 480)[0][1], before)
        ground.clear()
        self.assertAlmostEqual(ground.endpoints(640, 480)[0][1], round(0.85*479))
