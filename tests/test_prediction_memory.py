"""Verify first-target locking, quality gates, persistence, and reset."""
import unittest

from tennis_tracker.prediction import Landing
from tennis_tracker.prediction_memory import PredictionMemory


def target(x=150, now=0, impact=1, **changes):
    fields = dict(x=x, y=200, seconds=impact-now, path=[(x, 100), (x, 200)],
                  uncertainty_px=4, observation_count=16, observation_span=0.4)
    fields.update(changes)
    return Landing(**fields)


class PredictionMemoryTests(unittest.TestCase):
    def test_first_stable_target_stays_fixed_through_revisions_and_long_loss(self):
        memory = PredictionMemory(freeze_first=True)
        self.assertIsNone(memory.update(target(now=0), 0, True))
        self.assertIsNone(memory.update(target(x=151, now=0.03), 0.03, True))
        locked = memory.update(target(x=150.5, now=0.06), 0.06, True)
        self.assertIsNotNone(locked)
        self.assertTrue(locked.first_locked)
        revised = memory.update(target(x=300, now=0.1), 0.1, True)
        self.assertEqual(revised.x, locked.x)
        lost = memory.update(None, 20, False)
        self.assertEqual((lost.x, lost.y), (locked.x, locked.y))
        self.assertEqual(lost.path, [])
        self.assertEqual(lost.seconds, 0)
        reacquired = memory.update(target(x=400, now=21, impact=22), 21, True, new_track=True)
        self.assertEqual(reacquired.x, locked.x)

    def test_insufficient_noisy_or_disagreeing_estimates_are_not_shown(self):
        for changes in (dict(observation_count=7), dict(observation_span=0.29),
                        dict(uncertainty_px=40)):
            memory = PredictionMemory(freeze_first=True)
            for now in (0, 0.03, 0.06, 0.09):
                self.assertIsNone(memory.update(target(now=now, **changes), now, True))
        memory = PredictionMemory(freeze_first=True)
        for now, x in [(0, 100), (0.03, 130), (0.06, 160)]:
            self.assertIsNone(memory.update(target(x=x, now=now), now, True))
        # Once the rejected proposals age out, three consistent proposals can lock.
        for now in (0.4, 0.43):
            self.assertIsNone(memory.update(target(x=150, now=now), now, True))
        self.assertIsNotNone(memory.update(target(now=0.46), 0.46, True))

    def test_reset_clears_both_target_and_pending_confirmation(self):
        memory = PredictionMemory(freeze_first=True)
        memory.update(target(), 0, True)
        memory.reset()
        self.assertIsNone(memory.update(target(now=0.03), 0.03, True))
        self.assertIsNone(memory.update(target(now=0.06), 0.06, True))
        self.assertIsNotNone(memory.update(target(now=0.09), 0.09, True))
        memory.reset()
        self.assertIsNone(memory.update(None, 100, False))

    def test_impact_time_disagreement_prevents_locking(self):
        memory = PredictionMemory(freeze_first=True)
        for now, impact in [(0, 0.5), (0.03, 1), (0.06, 1.5)]:
            self.assertIsNone(memory.update(target(now=now, impact=impact), now, True))

    def test_plane_first_prediction_requires_nonprovisional_low_sensitivity_fit(self):
        memory = PredictionMemory(freeze_first=True)
        fields = dict(model="plane", floor_point=(0.5, 0.5), uncertainty_m=0.1,
                      uncertainty_px=None, provisional=True)
        for now in (0, 0.03, 0.06):
            self.assertIsNone(memory.update(target(now=now, **fields), now, True))
        fields["provisional"] = False
        for now in (0.3, 0.33):
            self.assertIsNone(memory.update(target(now=now, **fields), now, True))
        self.assertIsNotNone(memory.update(target(now=0.36, **fields), 0.36, True))
