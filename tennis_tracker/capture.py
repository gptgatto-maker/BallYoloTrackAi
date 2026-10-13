"""Keep camera handling and video timing in one place."""
import math
import time

import cv2


class VideoSource:
    def __init__(self, source):
        self.is_camera = str(source).isdigit()
        self.capture = cv2.VideoCapture(int(source) if self.is_camera else source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open {source!r}. Check the camera index or video path.")
        if self.is_camera:
            # Best effort: some Windows camera backends ignore this setting.
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.index = 0
        self.start = time.perf_counter()

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            return None, None
        # File time must not depend on how slowly YOLO processes the video.
        timestamp = (time.perf_counter() - self.start if self.is_camera
                     else self.index / self.fps)
        self.index += 1
        return frame, timestamp

    def close(self):
        self.capture.release()
