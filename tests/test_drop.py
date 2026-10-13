"""Short vertical falls for which the general first-lock rules are too slow."""
import unittest

import numpy as np

from tennis_tracker.physics import GRAVITY, predict_physics
from tennis_tracker.floor_plane import FloorPlane, predict_floor_plane
from tennis_tracker.prediction_memory import PredictionMemory
from test_floor_plane import fov_for, mark_plane, learn_plane
from test_physics import scene


def drop_samples(floor, last=0.24, count=9, height=0.5, noise=0, size_noise=0,
                 sideways=0, seed=5):
    times = np.linspace(0, last, count)
    positions = np.column_stack([1.1+sideways*times, np.ones(count),
                                 height-0.5*GRAVITY*times**2])
    pixels = floor.project(positions)
    random = np.random.default_rng(seed)
    pixels += random.normal(0, noise, pixels.shape)
    camera_points = positions@floor.rotation.T+floor.offset
    diameter = floor.camera.matrix[0, 0]*0.067/camera_points[:, 2]
    diameter *= 1+random.normal(0, size_noise, count)
    samples, sizes = [], []
    for time, (x, y), size in zip(times, pixels, diameter):
        samples.append((float(time), float(x), float(y)))
        sizes.append((float(time), (x-size/2, y-size/2, x+size/2, y+size/2), "YOLO"))
    impact = np.sqrt(2*(height-0.0335)/GRAVITY)
    return samples, sizes, floor.project([[1.1, 1, 0]])[0], impact


class DropTests(unittest.TestCase):
    def test_vertical_drop_has_a_target_before_general_first_lock_can_finish(self):
        floor = scene()
        samples, sizes, expected, impact = drop_samples(floor)
        general = PredictionMemory(freeze_first=True)
        for index, point in enumerate(samples):
            candidate, _ = predict_physics(samples[:index+1], point[0], floor)
            self.assertIsNone(general.update(candidate, point[0], True))
        landing, reason = predict_physics(samples, 0.24, floor, motion_model="drop",
                                          size_observations=sizes)
        self.assertIsNotNone(landing, reason)
        np.testing.assert_allclose([landing.x, landing.y], expected, atol=0.001)
        self.assertAlmostEqual(landing.seconds, impact-0.24, places=4)
        self.assertEqual(landing.motion_model, "drop")

    def test_first_drop_target_locks_before_short_fall_after_held_warmup(self):
        reference = scene()
        floor = FloorPlane(fov_for(reference))
        mark_plane(reference, floor)
        learn_plane(reference, floor)
        samples, sizes, expected, impact = drop_samples(reference, last=0.30, count=11)
        held_sizes = [(float(time), sizes[0][1], "YOLO")
                      for time in np.linspace(-0.4, -0.03, 14)]
        memory = PredictionMemory(freeze_first=True)
        first = None
        for i in range(len(samples)):
            now = samples[i][0]
            candidate, reason = predict_floor_plane(samples[:i+1], now, floor,
                                                     motion_model="drop", size_observations=held_sizes+sizes[:i+1])
            shown = memory.update(candidate, now, True)
            if shown is not None and first is None:
                first = now
                np.testing.assert_allclose([shown.x, shown.y], expected, atol=0.01)
        self.assertIsNotNone(first, reason)
        self.assertLess(first, impact)
        saved = memory.update(None, 10, False)
        self.assertTrue(saved.first_locked)
        np.testing.assert_allclose([saved.x, saved.y], expected, atol=0.01)

    def test_drop_cues_tolerate_small_detection_and_size_noise(self):
        floor = scene()
        samples, sizes, expected, _ = drop_samples(floor, noise=0.3, size_noise=0.025)
        landing, reason = predict_physics(samples, 0.24, floor, motion_model="drop",
                                          size_observations=sizes)
        self.assertIsNotNone(landing, reason)
        self.assertLess(np.linalg.norm([landing.x-expected[0], landing.y-expected[1]]), 12)

    def test_drop_does_not_predict_stationary_or_missing_size_data(self):
        floor = scene()
        samples, sizes, _, _ = drop_samples(floor)
        self.assertIsNone(predict_physics(samples, 0.24, floor, motion_model="drop")[0])
        constant = [(time, 640, 300) for time, _, _ in samples]
        self.assertIsNone(predict_physics(constant, 0.24, floor, motion_model="drop",
                                          size_observations=sizes)[0])

    def test_drop_rejects_clear_sideways_motion(self):
        floor = scene()
        samples, sizes, _, _ = drop_samples(floor, sideways=5)
        self.assertIsNone(predict_physics(samples, 0.24, floor, motion_model="drop",
                                          size_observations=sizes)[0])
