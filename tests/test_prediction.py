import unittest

import numpy as np

from tennis_tracker.prediction import predict_landing, draw_prediction


def arc():
    # Ground y=200 is reached at absolute t=1: x=150.
    return [(float(t), 100 + 50 * t, 100 + 100 * t * t)
            for t in np.linspace(0, 0.5, 16)]


class PredictionTests(unittest.TestCase):
    def test_known_parabola_and_drawing(self):
        landing, _ = predict_landing(arc(), 0.5, 200, 640)
        self.assertIsNotNone(landing)
        self.assertAlmostEqual(landing.x, 150)
        self.assertAlmostEqual(landing.seconds, 0.5)
        frame = np.zeros((480, 640, 3), np.uint8)
        draw_prediction(frame, landing, "", 200)
        self.assertTrue(np.any(frame[190:211, 140:161]))

    def test_coasting_reduces_time_to_intersection(self):
        landing, _ = predict_landing(arc(), 0.6, 200, 640)
        self.assertAlmostEqual(landing.seconds, 0.4)
        self.assertAlmostEqual(landing.x, 150)

    def test_rejects_insufficient_stale_outside_and_far_predictions(self):
        cases = [(arc()[:2], 0.5, 200, 640, 1.5),
                 (arc(), 0.8, 200, 640, 1.5),
                 (arc(), 0.5, 200, 120, 1.5),
                 (arc(), 0.5, 200, 640, 0.1),
                 (arc(), 0.5, 110, 640, 1.5)]
        for args in cases:
            with self.subTest(args=args):
                self.assertIsNone(predict_landing(*args)[0])

    def test_rejects_bad_fit_and_upward_curvature(self):
        noisy = [(t, x + (40 if i % 2 else -40), y)
                 for i, (t, x, y) in enumerate(arc())]
        self.assertIsNone(predict_landing(noisy, 0.5, 200, 640)[0])
        upward = [(t, x, 150 - 100 * t * t) for t, x, y in arc()]
        self.assertIsNone(predict_landing(upward, 0.5, 200, 640)[0])

    def test_linear_motion_and_stationary_ball(self):
        linear = [(t, 100, 100 + 100 * t) for t, x, y in arc()]
        landing, _ = predict_landing(linear, 0.5, 200, 640)
        self.assertAlmostEqual(landing.seconds, 0.5)
        stationary = [(t, 100, 100) for t, x, y in arc()]
        self.assertIsNone(predict_landing(stationary, 0.5, 200, 640)[0])

    def test_duplicate_times_rejected(self):
        self.assertIsNone(predict_landing([(0, 1, 1)] * 8, 0, 200, 640)[0])

    def test_large_timestamp_does_not_change_prediction(self):
        shifted = [(t + 100000, x, y) for t, x, y in arc()]
        landing, _ = predict_landing(shifted, 100000.5, 200, 640)
        self.assertAlmostEqual(landing.x, 150, places=5)

    def test_three_points_are_insufficient_for_a_reliable_estimate(self):
        samples = [(0, 100, 100), (0.03, 103, 109), (0.06, 106, 118)]
        landing, _ = predict_landing(samples, 0.06, 200, 640)
        self.assertIsNone(landing)

    def test_three_stationary_points_do_not_predict(self):
        samples = [(0, 100, 100), (0.03, 100, 100), (0.06, 100, 100)]
        self.assertIsNone(predict_landing(samples, 0.06, 200, 640)[0])

    def test_bullseye_center_is_blue(self):
        landing, _ = predict_landing(arc(), 0.5, 200, 640)
        frame = np.zeros((480, 640, 3), np.uint8)
        draw_prediction(frame, landing, "", 200)
        blue, green, red = frame[200, 150]
        self.assertGreater(int(blue), int(red))
        self.assertGreater(int(blue), int(green))

    def test_noisy_arc_with_a_bad_center_recovers_known_crossing(self):
        random = np.random.default_rng(9)
        samples = np.array([(t, 100+50*t, 100+100*t*t)
                            for t in np.linspace(0, 0.55, 24)])
        samples[:, 1:] += random.normal(0, 0.6, (24, 2))
        samples[10, 1:] += [45, -50]
        samples = [tuple(point) for i, point in enumerate(samples) if i not in (5, 17)]
        landing, reason = predict_landing(samples, 0.55, 200, 640)
        self.assertIsNotNone(landing, reason)
        self.assertLess(abs(landing.x-150), 5)
        self.assertLess(abs(landing.seconds-0.45), 0.06)
        self.assertIsNotNone(landing.uncertainty_px)

    def test_ascent_then_apex_predicts_future_downward_crossing(self):
        samples = [(float(t), 200+40*t, 120-160*t+200*t*t)
                   for t in np.linspace(0, 0.4, 18)]
        landing, reason = predict_landing(samples, 0.4, 200, 640)
        self.assertIsNotNone(landing, reason)
        time = (160+np.sqrt(160**2+4*200*80))/(2*200)
        self.assertAlmostEqual(landing.seconds, time-0.4)
        self.assertAlmostEqual(landing.x, 200+40*time)

    def test_post_crossing_observation_does_not_show_an_old_target(self):
        samples = arc()+[(0.53, 126.5, 210)]
        self.assertIsNone(predict_landing(samples, 0.53, 200, 640)[0])

    def test_new_motion_is_not_treated_as_multiple_bad_centers(self):
        samples = arc()+[(0.53, 160, 60), (0.56, 200, 40)]
        self.assertIsNone(predict_landing(samples, 0.56, 200, 640)[0])

    def test_five_observation_linear_estimate_is_labeled_early(self):
        samples = [(float(t), 100+50*t, 100+300*t)
                   for t in np.linspace(0, 0.16, 5)]
        landing, reason = predict_landing(samples, 0.16, 200, 640)
        self.assertIsNotNone(landing, reason)
        self.assertTrue(landing.provisional)
        self.assertAlmostEqual(landing.x, 100+50/3)

    def test_uncertain_short_arc_does_not_extrapolate_far(self):
        samples = [(float(t), 100+50*t, 100+100*t*t)
                   for t in np.linspace(0, 0.12, 5)]
        self.assertIsNone(predict_landing(samples, 0.12, 400, 640)[0])
