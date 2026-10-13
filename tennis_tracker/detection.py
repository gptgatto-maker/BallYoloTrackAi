"""YOLO finds candidate balls; the tracker chooses one across frames."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple
    confidence: float
    source: str = "YOLO"

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class BallDetector:
    def __init__(self, weights, confidence=0.25, class_id=None, imgsz=640,
                 device="cpu", require_tennis_color=True):
        # Lazy import keeps the capture-only stage independent of YOLO.
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if class_id is None:
            names = self.model.names
            matches = [i for i, name in names.items()
                       if name.lower().replace("_", " ").replace("-", " ")
                       in ("sports ball", "tennis ball", "ball")]
            if len(matches) != 1:
                raise ValueError(f"Choose --class-id from model classes: {names}")
            class_id = matches[0]
        if class_id not in self.model.names:
            raise ValueError(f"Invalid --class-id. Classes: {self.model.names}")
        self.class_id = class_id
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.require_tennis_color = require_tennis_color
        print(f"Detecting class {class_id}: {self.model.names[class_id]}")

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.confidence, classes=[self.class_id],
                                    imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        detections = [Detection(tuple(map(float, box)), float(score))
                      for box, score in zip(boxes, scores)]
        if self.require_tennis_color:
            detections = [detection for detection in detections
                          if looks_tennis_yellow(frame, detection.box)]
        return detections


def looks_tennis_yellow(frame, box, minimum_pixels=8, minimum_fraction=0.03):
    """Reject YOLO sports-ball boxes that contain no tennis-yellow pixels."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    # Slight padding helps when YOLO's box is tight around a blurred streak.
    padding = max(2, round(max(x2 - x1, y2 - y1) * 0.08))
    x1, y1 = max(0, round(x1) - padding), max(0, round(y1) - padding)
    x2, y2 = min(width, round(x2) + padding), min(height, round(y2) + padding)
    if x2 <= x1 or y2 <= y1:
        return False
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((18, 55, 65)), np.array((48, 255, 255)))
    yellow_pixels = int(cv2.countNonZero(mask))
    return (yellow_pixels >= minimum_pixels
            and yellow_pixels / mask.size >= minimum_fraction)


class TennisColorRecovery:
    """Find tennis-yellow blobs near an existing track after YOLO misses.

    Color alone is too easy to fool, so this detector never starts a track. The
    main loop only calls it near the Kalman filter's expected ball position.
    """
    def __init__(self, minimum_area=8, maximum_area_fraction=0.04):
        self.minimum_area = minimum_area
        self.maximum_area_fraction = maximum_area_fraction

    def detect_near(self, frame, expected, radius):
        height, width = frame.shape[:2]
        cx, cy = expected
        x1, y1 = max(0, round(cx - radius)), max(0, round(cy - radius))
        x2, y2 = min(width, round(cx + radius)), min(height, round(cy + radius))
        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # This broad band includes chartreuse/yellow tennis balls under common
        # indoor lighting. Saturation excludes most white walls and reflections.
        mask = cv2.inRange(hsv, np.array((18, 55, 65)), np.array((48, 255, 255)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        maximum_area = width * height * self.maximum_area_fraction
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not self.minimum_area <= area <= maximum_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            center = np.array((x1 + bx + bw / 2, y1 + by + bh / 2))
            distance = float(np.linalg.norm(center - np.array(expected)))
            if distance > radius:
                continue
            # A low confidence keeps YOLO preferred if both sources see a ball.
            candidates.append(Detection((x1 + bx, y1 + by,
                                         x1 + bx + bw, y1 + by + bh),
                                        0.05, "color recovery"))
        return candidates
