"""Keep a short history; separate measured and smoothed coordinates."""
from collections import deque

import cv2


class Trajectory:
    def __init__(self, max_points=90, seconds=0.8):
        self.samples = deque(maxlen=max_points)
        self.sizes = deque(maxlen=max_points)
        self.seconds = seconds

    def clear(self):
        self.samples.clear()
        self.sizes.clear()

    def update(self, track, timestamp):
        if track.new_track or track.position is None:
            self.clear()
        if track.detection is not None:
            # Draw the smoothed centers, fit actual measurements. Coasting is
            # never added: otherwise invented observations reinforce the fit.
            self.samples.append((timestamp, track.detection.center, track.position))
            self.sizes.append((timestamp, track.detection.box, track.detection.source))
        while self.samples and timestamp - self.samples[0][0] > self.seconds:
            self.samples.popleft()
        while self.sizes and timestamp-self.sizes[0][0] > self.seconds:
            self.sizes.popleft()

    def measurements(self):
        return [(t, *measured) for t, measured, smoothed in self.samples]

    def size_observations(self):
        return list(self.sizes)

    def draw(self, frame):
        samples = list(self.samples)
        for previous, current in zip(samples, samples[1:]):
            # Leave a visible break for longer missed-detection intervals.
            if current[0] - previous[0] <= 0.15:
                cv2.line(frame, tuple(map(round, previous[2])),
                         tuple(map(round, current[2])), (255, 255, 0), 2)
