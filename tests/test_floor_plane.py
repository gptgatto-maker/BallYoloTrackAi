"""Validate the plane experiment under its assumed-camera/size conditions."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_tracker.detection import Detection
from tennis_tracker.floor_plane import FloorPlane, predict_floor_plane
from tennis_tracker.tracking import Track
from test_physics import scene, observations


def fov_for(floor):
    return float(np.rad2deg(2*np.arctan(floor.size[0]/(2*floor.camera.matrix[0, 0]))))


def reference_detection(floor, point=(1.2, 1.2)):
    world = np.array([*point, 0.067/2])
    camera_point = floor.rotation@world+floor.offset
    x, y = floor.project([world])[0]
    diameter = floor.camera.matrix[0, 0]*0.067/camera_point[2]
    return Detection((x-diameter/2, y-diameter/2, x+diameter/2, y+diameter/2), 0.9)


def mark_plane(reference, plane):
    plane.configure_frame(*reference.size)
    plane.begin()
    for point in reference.project([(0, 4, 0), (3, 4, 0), (3, 0, 0), (0, 0, 0)]):
        plane.click(*point)


def learn_plane(reference, plane, point=(1.2, 1.2)):
    plane.begin_reference()
    detection = reference_detection(reference, point)
    for i in range(16):
        plane.observe_reference(Track(detection.center, detection), i/30)


class FloorPlaneTests(unittest.TestCase):
    def test_unmeasured_floor_and_known_ball_learn_geometry(self):
        reference = scene()
        plane = FloorPlane(fov_for(reference))
        mark_plane(reference, plane)
        self.assertFalse(plane.ready)
        self.assertIsNone(plane.mode)
        learn_plane(reference, plane)
        self.assertTrue(plane.ready, plane.status)
        self.assertAlmostEqual(plane.width_m, 3, places=4)
        self.assertAlmostEqual(plane.length_m, 4, places=4)
        np.testing.assert_allclose(plane.rotation, reference.rotation, atol=1e-5)
        np.testing.assert_allclose(plane.offset, reference.offset, atol=1e-4)

    def test_resting_ball_can_be_anywhere_and_unknown_landings_differ(self):
        reference = scene()
        targets = []
        for point, velocity in [((1.2, 1.2), [1, 0.8, 1]), ((2, 3), [1, 3.0, 1])]:
            plane = FloorPlane(fov_for(reference))
            mark_plane(reference, plane)
            learn_plane(reference, plane, point)
            samples, expected, _ = observations(reference, velocity=velocity)
            landing, reason = predict_floor_plane(samples, 0.4, plane)
            self.assertIsNotNone(landing, reason)
            self.assertEqual(landing.model, "plane")
            np.testing.assert_allclose(landing.floor_point,
                                       expected/np.array([3, 4]), atol=1e-4)
            np.testing.assert_allclose([landing.x, landing.y],
                                       reference.project([[*expected, 0]])[0], atol=0.01)
            targets.append(landing.y)
        self.assertGreater(abs(targets[1]-targets[0]), 30)

    def test_reference_requires_stability_and_usable_size(self):
        reference = scene()
        plane = FloorPlane(fov_for(reference))
        self.assertFalse(plane.begin_reference())
        mark_plane(reference, plane)
        plane.begin_reference()
        for i in range(20):
            detection = reference_detection(reference, (0.4+i*0.1, 1.2))
            plane.observe_reference(Track(detection.center, detection), i/30)
        self.assertFalse(plane.ready)
        plane.begin_reference()
        plane.observe_reference(Track((600, 500), Detection((599, 499, 601, 501), 0.9)), 0)
        self.assertEqual(len(plane.reference_samples), 0)
        fallback = reference_detection(reference)
        fallback.source = "color recovery"
        plane.observe_reference(Track(fallback.center, fallback), 0.03)
        self.assertEqual(len(plane.reference_samples), 0)

    def test_setup_restores_only_for_matching_source_size_and_assumptions(self):
        reference = scene()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plane.json"
            plane = FloorPlane(fov_for(reference), setup_path=path, source_key="phone")
            mark_plane(reference, plane)
            learn_plane(reference, plane)
            self.assertTrue(path.exists())
            restored = FloorPlane(fov_for(reference), setup_path=path, source_key="phone")
            restored.configure_frame(*reference.size)
            self.assertTrue(restored.ready, restored.status)
            samples, expected, _ = observations(reference)
            landing, reason = predict_floor_plane(samples, 0.4, restored)
            self.assertIsNotNone(landing, reason)
            changed_fov = FloorPlane(60, setup_path=path, source_key="phone")
            changed_fov.configure_frame(*reference.size)
            self.assertFalse(changed_fov.ready)
            restored.configure_frame(640, 480)
            self.assertFalse(restored.ready)

    def test_floor_overlap_without_reference_does_not_create_a_landing(self):
        reference = scene()
        plane = FloorPlane(fov_for(reference))
        mark_plane(reference, plane)
        samples, _, _ = observations(reference)
        landing, reason = predict_floor_plane(samples, 0.4, plane)
        self.assertIsNone(landing)
        self.assertIn("reference", reason)
