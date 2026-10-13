"""One-time lens setup: SPACE captures a board pose, C solves/saves, Q quits."""
import argparse
import sys

import cv2
import numpy as np

from tennis_tracker.calibration import CameraCalibration
from tennis_tracker.capture import VideoSource


def run(args):
    source = VideoSource(args.source)
    pattern = (args.columns, args.rows)
    objects = np.zeros((args.columns * args.rows, 3), np.float32)
    objects[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square_m
    image_samples = []
    size = None
    try:
        while True:
            frame, _ = source.read()
            if frame is None:
                break
            current_size = (frame.shape[1], frame.shape[0])
            if size is not None and size != current_size:
                raise ValueError("Stream resolution changed during calibration.")
            size = current_size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(gray, pattern)
            if found:
                cv2.drawChessboardCorners(frame, pattern, corners, True)
            cv2.putText(frame, f"Poses: {len(image_samples)} | SPACE: capture | C: solve | Q: quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Lens calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(
                    "Lens calibration", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                if not found:
                    print("Entire checkerboard must be visible and detected.")
                elif any(np.sqrt(np.mean((corners-old)**2)) < 12 for old in image_samples):
                    print("Move/tilt the board more before capturing another pose.")
                else:
                    image_samples.append(corners.copy())
                    print(f"Captured pose {len(image_samples)}. Vary angle, distance, and image location.")
            if key in (ord("c"), ord("C")):
                if len(image_samples) < 12:
                    print("Collect at least 12 varied board poses; 20 is recommended.")
                    continue
                coverage = cv2.contourArea(cv2.convexHull(
                    np.concatenate(image_samples).astype(np.float32))) / (size[0]*size[1])
                if coverage < 0.25:
                    print("Board samples cover too little of the image. Capture near the edges too.")
                    continue
                rms, matrix, distortion, _, _ = cv2.calibrateCamera(
                    [objects] * len(image_samples), image_samples, size, None, None)
                if not np.isfinite(rms) or rms > 1.5:
                    print(f"Calibration error {rms:.2f}px is too high. Restart with sharper, varied views.")
                    continue
                CameraCalibration(matrix, distortion, size, rms).save(args.output)
                print(f"Saved {args.output}; RMS {rms:.2f}px. Keep the same lens, zoom, and resolution.")
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the camera with a printed checkerboard")
    parser.add_argument("--source", default="1")
    parser.add_argument("--columns", type=int, default=9, help="INNER corner columns")
    parser.add_argument("--rows", type=int, default=6, help="INNER corner rows")
    parser.add_argument("--square-m", type=float, default=0.024)
    parser.add_argument("--output", default="calibration/camera.json")
    options = parser.parse_args()
    if options.columns < 3 or options.rows < 3 or not np.isfinite(options.square_m) or options.square_m <= 0:
        parser.error("Use at least 3 rows/columns of inner corners and a positive square size.")
    try:
        run(options)
    except KeyboardInterrupt:
        print("Stopped.")
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
