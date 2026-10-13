"""Keep one throw's blue target visible until an explicit reset."""
from dataclasses import replace
from collections import deque

import numpy as np


class PredictionMemory:
    def __init__(self, gap_seconds=0.7, freeze_first=False):
        self.gap_seconds = gap_seconds
        self.freeze_first = freeze_first
        self.reset()

    def reset(self):
        self.prediction = None
        self.impact_time = None
        self.last_update = None
        self.frozen = False
        self.pending = deque(maxlen=60)
        self.status = None

    def _first_is_consistent(self, candidate, now, observed):
        if candidate is None or not observed:
            self.pending.clear()
            return False
        dropping = candidate.motion_model == "drop"
        required_count, required_span = (5, 0.12) if dropping else (8, 0.30)
        if candidate.observation_count < required_count or candidate.observation_span < required_span:
            self.pending.clear()
            self.status = f"Waiting: first estimate needs {required_count} positions and {required_span:.2f}s"
            return False
        if dropping:
            # A saved IMAGE target needs bounded projected spread. A fixed meter
            # cutoff was too strict for short distant drops and too loose nearby.
            quality_ok = (not candidate.provisional and candidate.uncertainty_px is not None
                          and candidate.uncertainty_px <= 25)
        elif candidate.model == "image":
            quality_ok = candidate.uncertainty_px is not None and candidate.uncertainty_px <= 25
        else:
            quality_ok = (not candidate.provisional and candidate.uncertainty_m is not None
                          and candidate.uncertainty_m <= 0.35)
        if not quality_ok:
            self.pending.clear()
            self.status = "Waiting: first prediction is still too uncertain"
            return False
        self.pending.append((now, candidate.x, candidate.y, now+candidate.seconds))
        while self.pending and now-self.pending[0][0] > 0.20:
            self.pending.popleft()
        values = np.asarray(self.pending)
        self.status = "Waiting: checking first target consistency"
        if len(values) < (2 if dropping else 3) or np.ptp(values[:, 0]) < (0.03 if dropping else 0.05):
            return False
        # Repeated fits must agree on target and absolute impact time. Agreement
        # is a quality gate, not a guarantee against lens/floor/model bias.
        return (np.linalg.norm(np.ptp(values[:, 1:3], axis=0)) <= 12
                and np.ptp(values[:, 3]) <= 0.12)

    def update(self, candidate, now, observed, new_track=False):
        self.status = None
        if new_track and self.prediction is None:
            self.pending.clear()
        if self.prediction is not None:
            # A reacquired target could be a bounced ball or a different object.
            # Keep the previous throw until R, rather than silently replacing it.
            if (new_track or now >= self.impact_time
                    or (not observed and now-self.last_update > self.gap_seconds)):
                self.frozen = True

        if self.freeze_first and self.prediction is None:
            if not self._first_is_consistent(candidate, now, observed):
                return None

        if candidate is not None and not self.frozen and (observed or self.prediction is None):
            self.prediction = replace(candidate, held=not observed, retained=False)
            self.impact_time = now+candidate.seconds
            self.last_update = now
            self.frozen = self.freeze_first
            if not self.freeze_first:
                return self.prediction

        if self.prediction is None:
            return None
        # Only the bullseye remains: an old future path/countdown would look as
        # though the prediction were still live after the throw has finished.
        return replace(self.prediction, held=True, retained=True, path=[],
                       first_locked=self.freeze_first,
                       seconds=max(0.0, self.impact_time-now))
