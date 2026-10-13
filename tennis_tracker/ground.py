"""User-marked floor region and a perspective-aware assumed contact guide.

The homography maps FLOOR points only. It never converts an airborne ball into
3D. The guide's depth remains an assumption you can adjust for the experiment.
"""
import cv2
import numpy as np


class GroundMap:
    def __init__(self, default_fraction=0.85):
        self.default_fraction = default_fraction
        self.corners = []
        self.guide = []
        self.pending = []
        self.mode = None
        self.depth = 0.5
        self.revision = 0
        self.row = None
        self.image_height = None

    def begin(self, mode):
        self.mode = mode
        self.pending = []
        self.revision += 1

    def cancel(self):
        self.mode = None
        self.pending = []
        self.revision += 1

    def click(self, x, y):
        if self.mode is None:
            return
        if self.mode == "row":
            self.row = float(y)
            self.corners = []
            self.guide = []
            self.cancel()
            return
        self.pending.append((x, y))
        count = 4 if self.mode == "floor" else 2
        if len(self.pending) < count:
            return
        if self.mode == "floor":
            polygon = np.array(self.pending, np.int32)
            # Reject crossed, concave, or nearly collinear corner selections.
            if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 100:
                self.pending = []
                return
            self.corners = self.pending[:]
            self.guide = []
        else:
            if abs(self.pending[1][0] - self.pending[0][0]) < 5:
                self.pending = []
                return
            self.guide = self.pending[:]
            self.row = None
        self.cancel()

    def adjust_depth(self, step):
        if not self.corners:
            if self.row is not None and self.image_height is not None:
                self.row = float(np.clip(self.row+step*(self.image_height-1), 0, self.image_height-1))
            else:
                self.default_fraction = float(np.clip(self.default_fraction+step, 0.05, 0.95))
            self.guide = []
            self.revision += 1
            return
        self.depth = float(np.clip(self.depth + step, 0.05, 0.95))
        self.guide = []
        self.revision += 1

    def clear(self):
        self.corners = []
        self.guide = []
        self.row = None
        self.cancel()

    def project(self, points):
        # Click order: far-left, far-right, near-right, near-left.
        unit = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], np.float32)
        matrix = cv2.getPerspectiveTransform(unit, np.array(self.corners, np.float32))
        return cv2.perspectiveTransform(np.array([points], np.float32), matrix)[0]

    def endpoints(self, width, height):
        if self.guide:
            points = self.guide
        elif self.corners:
            points = self.project([(0, self.depth), (1, self.depth)])
        else:
            row = self.row if self.row is not None else round(self.default_fraction * (height - 1))
            points = [(0, row), (width - 1, row)]
        return sorted([tuple(map(float, p)) for p in points], key=lambda p: p[0])

    def contains(self, x, y):
        return (not self.corners or cv2.pointPolygonTest(
            np.array(self.corners, np.float32), (float(x), float(y)), False) >= 0)

    def draw(self, frame):
        height, width = frame.shape[:2]
        self.image_height = height
        if self.corners:
            polygon = np.array(self.corners, np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (80, 130, 40))
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, dst=frame)
            cv2.polylines(frame, [polygon], True, (160, 210, 110), 2)
            for depth in (0.25, 0.5, 0.75):
                first, second = self.project([(0, depth), (1, depth)])
                cv2.line(frame, tuple(map(round, first)), tuple(map(round, second)),
                         (140, 170, 110), 1)
        first, second = self.endpoints(width, height)
        cv2.line(frame, tuple(map(round, first)), tuple(map(round, second)), (255, 190, 60), 2)
        for point in self.pending:
            cv2.circle(frame, point, 5, (0, 255, 255), -1)
        if self.mode == "row":
            instruction = "Click once to place the assumed 2D ground line"
        elif self.mode == "floor":
            instruction = f"Floor {len(self.pending)+1}/4: far-left, far-right, near-right, near-left"
        elif self.mode == "guide":
            instruction = f"Contact guide {len(self.pending)+1}/2: click two ball-center contact locations"
        else:
            instruction = "G: ground row (1 click) | L: sloped line (2 clicks) | [ / ]: move row"
        cv2.putText(frame, instruction, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 220, 150), 1)
        if self.mode is None:
            label = "Approximate 2D prediction: crossing the selected line; depth unknown"
            cv2.putText(frame, label, (10, 103), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 220, 150), 1)
