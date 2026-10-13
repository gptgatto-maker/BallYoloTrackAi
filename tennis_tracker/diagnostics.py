"""Optional CSV decisions for debugging real drops without recording video."""
import csv
from pathlib import Path
import time


class PredictionDiagnostics:
    columns = ["session", "time_s", "mode", "motion_model", "fps", "detector",
               "x", "y", "box_width", "box_height", "confidence", "positions",
               "history_s", "candidate", "candidate_points", "candidate_span_s",
               "uncertainty_m", "uncertainty_px", "displayed", "first_locked", "target_x", "target_y", "reason"]

    def __init__(self, filename=None):
        self.file = None
        if filename:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            header_needed = not path.exists() or path.stat().st_size == 0
            fields = self.columns
            if not header_needed:
                with path.open(newline="", encoding="utf-8") as previous:
                    fields = next(csv.reader(previous))
                if "session" not in fields or "time_s" not in fields:
                    raise ValueError("Diagnostics file is not a compatible CSV; choose a new filename.")
            self.file = path.open("a", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=fields, extrasaction="ignore")
            self.session = time.time_ns()
            if header_needed:
                self.writer.writeheader()
            print(f"Prediction diagnostics are appended to {path}")

    def record(self, timestamp, mode, motion_model, fps, track, points, candidate, displayed, reason):
        if self.file is None:
            return
        row = dict(session=self.session, time_s=timestamp, mode=mode, motion_model=motion_model,
                   fps=round(fps, 1), positions=len(points),
                   history_s=points[-1][0]-points[0][0] if len(points)>1 else 0,
                   candidate=candidate is not None, displayed=displayed is not None,
                   first_locked=bool(displayed and displayed.first_locked), reason=reason)
        if track.detection is not None:
            detection = track.detection
            row.update(detector=detection.source, x=detection.center[0], y=detection.center[1],
                       box_width=detection.box[2]-detection.box[0],
                       box_height=detection.box[3]-detection.box[1], confidence=detection.confidence)
        if candidate is not None:
            row.update(candidate_points=candidate.observation_count,
                       candidate_span_s=candidate.observation_span,
                       uncertainty_m=candidate.uncertainty_m, uncertainty_px=candidate.uncertainty_px)
        if displayed is not None:
            row.update(target_x=displayed.x, target_y=displayed.y)
        self.writer.writerow(row)

    def close(self):
        if self.file is not None:
            self.file.close()
