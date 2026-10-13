"""Project known 3D throws into one camera, then recover unseen ground contact."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from tennis_tracker.calibration import (CameraCalibration, estimate_camera_from_rectangle,
                                        solve_floor_pose)
from tennis_tracker.metric_floor import MetricFloor
from tennis_tracker.physics import predict_physics, GRAVITY


def scene(distorted=False):
    matrix = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1]], float)
    distortion = np.array([-0.12, 0.035, 0.001, -0.001, 0]) if distorted else np.zeros(5)
    camera = CameraCalibration(matrix, distortion, (1280, 720), 0.2)
    center = np.array([1.1, -2.5, 1.6])
    forward = np.array([1.1, 1.5, 0.4])-center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    floor = MetricFloor(camera, 3.0, 4.0)
    floor.configure_frame(1280, 720)
    floor.rotation = rotation
    floor.offset = -rotation @ center
    return floor


def observations(floor, last=0.4, noise=0, seed=5, count=18, shift=0,
                 velocity=None):
    position = np.array([0.6, 0.5, 1.2])
    velocity = np.array([1.0, 1.5, 1.0]) if velocity is None else np.asarray(velocity)
    times = np.linspace(0, last, count)
    points = position + times[:, None]*velocity + 0.5*times[:, None]**2*[0, 0, -GRAVITY]
    pixels = floor.project(points)
    pixels += np.random.default_rng(seed).normal(0, noise, pixels.shape)
    samples = [(float(t+shift), float(x), float(y)) for t, (x, y) in zip(times, pixels)]
    impact_time = (velocity[2]+np.sqrt(velocity[2]**2+2*GRAVITY*(position[2]-0.0335)))/GRAVITY
    expected = position[:2] + velocity[:2]*impact_time
    return samples, expected, impact_time


class PhysicsTests(unittest.TestCase):
    def test_two_unknown_landing_depths_use_the_same_floor_calibration(self):
        floor = scene()
        contacts = []
        for velocity in ([1.0, 0.8, 1.0], [1.0, 3.0, 1.0]):
            samples, expected, _ = observations(floor, velocity=velocity)
            landing, reason = predict_physics(samples, 0.4, floor)
            self.assertIsNotNone(landing, reason)
            np.testing.assert_allclose(landing.floor_point, expected, atol=1e-5)
            contacts.append(landing.floor_point)
        self.assertGreater(contacts[1][1]-contacts[0][1], 1.0)

    def test_recovers_depth_and_floor_landing_without_contact_guide(self):
        floor = scene()
        samples, expected, impact_time = observations(floor)
        landing, reason = predict_physics(samples, 0.4, floor)
        self.assertIsNotNone(landing, reason)
        np.testing.assert_allclose(landing.floor_point, expected, atol=1e-5)
        self.assertAlmostEqual(landing.seconds, impact_time-0.4, places=5)
        np.testing.assert_allclose([landing.x, landing.y],
                                   floor.project([[*expected, 0]])[0], atol=1e-4)

    def test_distorted_lens_and_missing_observations(self):
        floor = scene(distorted=True)
        samples, expected, _ = observations(floor)
        samples = [sample for i, sample in enumerate(samples) if i not in (5, 8, 12)]
        landing, reason = predict_physics(samples, 0.4, floor)
        self.assertIsNotNone(landing, reason)
        np.testing.assert_allclose(landing.floor_point, expected, atol=0.005)

    def test_noisy_measurements_still_predict_with_bounded_error(self):
        floor = scene()
        samples, expected, _ = observations(floor, last=0.42, noise=0.4, count=24)
        landing, reason = predict_physics(samples, 0.42, floor)
        self.assertIsNotNone(landing, reason)
        self.assertLess(np.linalg.norm(np.array(landing.floor_point)-expected), 0.2)

    def test_no_calibration_stationary_stale_short_and_rank_deficient(self):
        floor = scene()
        samples, _, _ = observations(floor)
        floor.rotation = None
        self.assertIsNone(predict_physics(samples, 0.4, floor)[0])
        floor = scene()
        stationary = [(i*0.03, 640, 360) for i in range(18)]
        self.assertIsNone(predict_physics(stationary, 0.51, floor)[0])
        self.assertIsNone(predict_physics(samples, 0.8, floor)[0])
        short, _, _ = observations(floor, last=0.1)
        self.assertIsNone(predict_physics(short, 0.1, floor)[0])
        # Rays on one straight image line contain no independent depth curvature.
        straight = [(i*0.03, 640+i*3, 360+i*2) for i in range(18)]
        self.assertIsNone(predict_physics(straight, 0.51, floor)[0])

    def test_measured_floor_pose_and_corner_order(self):
        floor = scene()
        corners = [(0, 4, 0), (3, 4, 0), (3, 0, 0), (0, 0, 0)]
        pixels = floor.project(corners)
        rotation, offset, error = solve_floor_pose(floor.camera, pixels, 3, 4)
        np.testing.assert_allclose(rotation, floor.rotation, atol=1e-5)
        np.testing.assert_allclose(offset, floor.offset, atol=1e-5)
        self.assertLess(error, 1e-5)

    def test_floor_clicks_calibrate_then_clear_and_size_mismatch_invalidates(self):
        floor = scene()
        pixels = floor.project([(0, 4, 0), (3, 4, 0), (3, 0, 0), (0, 0, 0)])
        floor.clear()
        floor.begin()
        for x, y in pixels:
            floor.click(float(x), float(y))
        self.assertTrue(floor.ready, floor.status)
        floor.configure_frame(640, 480)
        self.assertFalse(floor.ready)

    def test_large_timestamps_and_horizon(self):
        floor = scene()
        samples, expected, _ = observations(floor, shift=100000)
        landing, reason = predict_physics(samples, 100000.4, floor)
        self.assertIsNotNone(landing, reason)
        np.testing.assert_allclose(landing.floor_point, expected, atol=1e-4)
        self.assertIsNone(predict_physics(samples, 100000.4, floor, horizon=0.05)[0])

    def test_calibration_round_trip(self):
        camera = scene().camera
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"camera.json"
            camera.save(path)
            loaded = CameraCalibration.load(path)
            np.testing.assert_equal(loaded.matrix, camera.matrix)
            self.assertEqual(loaded.size, camera.size)

    def test_no_print_lens_estimate_and_saved_setup(self):
        floor = scene()
        pixels = floor.project([(0, 4, 0), (3, 4, 0), (3, 0, 0), (0, 0, 0)])
        camera = estimate_camera_from_rectangle(pixels, (1280, 720), 3, 4)
        self.assertAlmostEqual(camera.matrix[0, 0], 900, places=2)
        self.assertEqual(camera.method, "floor-estimated")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"setup.json"
            initial = MetricFloor(None, 3, 4, path, "phone-1")
            initial.configure_frame(1280, 720)
            initial.begin()
            for point in pixels:
                initial.click(*map(float, point))
            self.assertTrue(initial.ready, initial.status)
            self.assertTrue(path.exists())
            restored = MetricFloor(None, None, None, path, "phone-1")
            restored.configure_frame(1280, 720)
            self.assertTrue(restored.ready, restored.status)
            self.assertEqual(restored.width_m, 3)
            samples, expected, _ = observations(floor)
            landing, reason = predict_physics(samples, 0.4, restored)
            self.assertIsNotNone(landing, reason)
            np.testing.assert_allclose(landing.floor_point, expected, atol=0.002)
            wrong_source = MetricFloor(None, None, None, path, "different-phone")
            wrong_source.configure_frame(1280, 720)
            self.assertFalse(wrong_source.ready)
            wrong_size = MetricFloor(None, None, None, path, "phone-1")
            wrong_size.configure_frame(640, 480)
            self.assertFalse(wrong_size.ready)
            restored.clear()
            cleared = MetricFloor(None, None, None, path, "phone-1")
            cleared.configure_frame(1280, 720)
            self.assertFalse(cleared.ready)
            self.assertEqual(cleared.width_m, 3)

    def test_rectangle_cannot_estimate_lens_from_front_parallel_view(self):
        with self.assertRaises(ValueError):
            estimate_camera_from_rectangle([(100, 100), (400, 100),
                                            (400, 500), (100, 500)], (1280, 720), 3, 4)

    def test_g_without_dimensions_prompts_then_allows_floor_marking(self):
        floor = MetricFloor(None, None, None)
        floor.configure_frame(1280, 720)
        self.assertTrue(floor.begin())
        self.assertEqual(floor.mode, "dimensions")
        # A mouse click during measurement entry must not become a floor corner.
        floor.click(100, 100)
        self.assertEqual(floor.pending, [])
        floor.handle_dimension_key(ord("0"))
        floor.handle_dimension_key(13)
        self.assertIsNone(floor.width_m)
        floor.handle_dimension_key(8)
        for key in [ord("3"), 13, ord("4"), 13]:
            floor.handle_dimension_key(key)
        self.assertEqual((floor.width_m, floor.length_m), (3, 4))
        self.assertEqual(floor.mode, "floor")
        reference = scene()
        corners = reference.project([(0, 4, 0), (3, 4, 0), (3, 0, 0), (0, 0, 0)])
        for point in corners:
            floor.click(*map(float, point))
        self.assertTrue(floor.ready, floor.status)


if __name__ == "__main__":
    unittest.main()
