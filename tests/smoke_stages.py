"""Generate a temporary blank video and check staged application execution.

This verifies plumbing, not tennis-ball detection accuracy. Stage 3+ runs real
YOLO inference and may download weights the first time.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, default=6)
    parser.add_argument("--only-stage", type=int, choices=range(2, 7))
    parser.add_argument("--prediction-mode", choices=("image", "plane", "physics"), default="image")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "smoke.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
        if not writer.isOpened():
            raise RuntimeError("Cannot create smoke test video.")
        for _ in range(8):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        stages = [args.only_stage] if args.only_stage is not None else range(2, args.through+1)
        for stage in stages:
            subprocess.run([sys.executable, str(root / "main.py"), "--stage", str(stage),
                            "--source", path, "--headless", "--imgsz", "320",
                            "--prediction-mode", args.prediction_mode],
                           cwd=root, check=True)
            print(f"Stage {stage} smoke check passed", flush=True)


if __name__ == "__main__":
    main()
