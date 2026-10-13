"""Run stage 2 first: python main.py --stage 2 --source 0."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

import cv2

from tennis_tracker.capture import VideoSource


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental tennis ball tracker")
    parser.add_argument("--stage", type=int, choices=range(2, 7), default=2)
    parser.add_argument("--source", default="0", help="Camera number or video file path")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device such as 0")
    parser.add_argument("--max-gap", type=float, default=0.4, help="Seconds to coast without detection")
    parser.add_argument("--gate", type=float, default=100, help="Maximum association distance in pixels")
    parser.add_argument("--headless", action="store_true", help="Process without a window (testing)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means unlimited")
    parser.add_argument("--history-seconds", type=float, default=0.8)
    parser.add_argument("--ground-y", type=float, default=0.85, help="Ground row as fraction of height")
    parser.add_argument("--horizon", type=float, default=1.5, help="Maximum prediction seconds ahead")
    parser.add_argument("--no-color-recovery", action="store_true",
                        help="Disable tennis-yellow recovery after a YOLO miss")
    parser.add_argument("--allow-any-ball", action="store_true",
                        help="Allow non-yellow YOLO sports-ball detections")
    parser.add_argument("--prediction-hold", type=float, default=0.7,
                        help="Missing-frame grace for live updates; saved targets always remain until R")
    parser.add_argument("--update-prediction", action="store_true",
                        help="Allow live revisions; default locks the first stable target until R")
    parser.add_argument("--prediction-mode", choices=("image", "plane", "physics"), default="image",
                        help="2D line (default), experimental unmeasured plane, or calibrated physics")
    parser.add_argument("--camera-calibration", default="calibration/camera.json")
    parser.add_argument("--floor-width", type=float, help="Measured far-left to far-right distance in meters")
    parser.add_argument("--floor-length", type=float, help="Measured far edge to near edge distance in meters")
    parser.add_argument("--setup-file", help="Saved floor/lens setup; default is per-source in calibration/")
    parser.add_argument("--plane-fov", type=float, default=70,
                        help="Assumed horizontal lens angle in degrees for plane mode")
    parser.add_argument("--ball-diameter", type=float, default=0.067,
                        help="Ball diameter in meters for plane reference (tennis default 0.067)")
    parser.add_argument("--motion-model", choices=("flight", "drop"), default="flight",
                        help="Drop constrains sideways velocity to zero and uses ball-size cues")
    parser.add_argument("--detect-every", type=int, default=1,
                        help="Run YOLO every N frames; yellow recovery handles frames between")
    parser.add_argument("--diagnostics", help="Append detection/prediction decisions to a CSV file")
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.confidence, args.gate, args.max_gap,
                                         args.history_seconds, args.ground_y, args.horizon)):
        parser.error("Numeric settings must be finite.")
    if not 0 < args.confidence <= 1 or args.gate <= 0 or args.max_gap <= 0:
        parser.error("Confidence must be in (0, 1]; gate and max-gap must be positive.")
    if args.imgsz < 32 or args.max_frames < 0:
        parser.error("imgsz must be >= 32 and max-frames must be >= 0.")
    if args.history_seconds <= 0:
        parser.error("history-seconds must be positive.")
    if not 0 < args.ground_y < 1 or args.horizon <= 0:
        parser.error("ground-y must be between 0 and 1; horizon must be positive.")
    if not math.isfinite(args.prediction_hold) or args.prediction_hold < 0:
        parser.error("prediction-hold must be finite and non-negative.")
    for dimension in (args.floor_width, args.floor_length):
        if dimension is not None and (not math.isfinite(dimension) or dimension <= 0):
            parser.error("Floor dimensions must be finite positive numbers in meters.")
    if (args.floor_width is None) != (args.floor_length is None):
        parser.error("Supply both --floor-width and --floor-length.")
    if not math.isfinite(args.plane_fov) or not 30 <= args.plane_fov <= 110:
        parser.error("plane-fov must be between 30 and 110 degrees.")
    if not math.isfinite(args.ball_diameter) or not 0.02 <= args.ball_diameter <= 0.15:
        parser.error("ball-diameter must be between 0.02 and 0.15 meters.")
    if args.detect_every < 1:
        parser.error("detect-every must be at least 1.")
    if args.stage >= 6 and args.motion_model == "drop" and args.prediction_mode == "image":
        parser.error("Drop mode needs --prediction-mode plane or physics.")
    if args.headless and args.source.isdigit() and not args.max_frames:
        parser.error("Use --max-frames with a headless webcam run.")
    return args


def run(args):
    detector = None
    color_recovery = None
    if args.stage >= 3:
        from tennis_tracker.detection import BallDetector, TennisColorRecovery
        detector = BallDetector(args.weights, args.confidence, args.class_id,
                                args.imgsz, args.device,
                                require_tennis_color=not args.allow_any_ball)
        if args.stage >= 4 and not args.no_color_recovery:
            color_recovery = TennisColorRecovery()
    tracker = None
    if args.stage >= 4:
        from tennis_tracker.tracking import BallTracker
        tracker = BallTracker(args.max_gap, args.gate)
    trajectory = None
    if args.stage >= 5:
        from tennis_tracker.trajectory import Trajectory
        trajectory = Trajectory(seconds=args.history_seconds)
    if args.stage >= 6:
        from tennis_tracker.prediction import predict_landing, draw_prediction
        if args.prediction_mode == "physics":
            from tennis_tracker.calibration import CameraCalibration
            from tennis_tracker.metric_floor import MetricFloor
            from tennis_tracker.physics import predict_physics
            calibration_path = Path(args.camera_calibration)
            camera = CameraCalibration.load(calibration_path) if calibration_path.exists() else None
            # Hash source identity instead of storing possible stream passwords.
            source_key = hashlib.sha256(args.source.encode("utf-8")).hexdigest()[:16]
            setup_path = args.setup_file or f"calibration/setup-{source_key}.json"
            ground = MetricFloor(camera, args.floor_width, args.floor_length,
                                 setup_path=setup_path, source_key=source_key)
        elif args.prediction_mode == "plane":
            from tennis_tracker.floor_plane import FloorPlane, predict_floor_plane
            source_key = hashlib.sha256(args.source.encode("utf-8")).hexdigest()[:16]
            setup_path = args.setup_file or f"calibration/plane-{source_key}.json"
            ground = FloorPlane(args.plane_fov, args.ball_diameter, setup_path, source_key)
        else:
            from tennis_tracker.ground import GroundMap
            ground = GroundMap(args.ground_y)
    else:
        ground = None
    from tennis_tracker.prediction_memory import PredictionMemory
    prediction_memory = PredictionMemory(args.prediction_hold, freeze_first=not args.update_prediction)

    def on_mouse(event, x, y, flags, parameter):
        if (event == cv2.EVENT_LBUTTONDOWN and ground is not None
                and ground.mode in ("row", "floor", "guide")):
            ground.click(x, y)
            prediction_memory.reset()
            # Never fit a throw observed before the camera/floor setup changed.
            if trajectory is not None:
                trajectory.clear()

    source = VideoSource(args.source)
    diagnostics = None
    previous_start = None
    frame_period = None
    try:
        from tennis_tracker.diagnostics import PredictionDiagnostics
        diagnostics = PredictionDiagnostics(args.diagnostics)
        if not args.headless:
            cv2.namedWindow("Tennis tracker", cv2.WINDOW_AUTOSIZE)
            if args.stage >= 6:
                cv2.setMouseCallback("Tennis tracker", on_mouse)
        while True:
            frame_start = time.perf_counter()
            if previous_start is not None:
                elapsed = frame_start-previous_start
                frame_period = elapsed if frame_period is None else 0.9*frame_period+0.1*elapsed
            previous_start = frame_start
            fps = 1/frame_period if frame_period and frame_period > 0 else 0
            frame, timestamp = source.read()
            if frame is None:
                print("Camera stopped returning frames." if source.is_camera else "End of video.")
                break
            if args.stage >= 6 and args.prediction_mode in ("physics", "plane"):
                ground.configure_frame(frame.shape[1], frame.shape[0])
            should_detect = (detector is not None and
                             (tracker is None or not tracker.active
                              or (source.index-1) % args.detect_every == 0
                              or (ground is not None and ground.mode == "reference")))
            detections = detector.detect(frame) if should_detect else []
            # YOLO must establish the track. Once it has, tennis-yellow blobs
            # near the expected position can bridge motion-blurred YOLO misses.
            if color_recovery and tracker.active and not detections:
                expected = tracker.expected_position(timestamp)
                detections.extend(color_recovery.detect_near(
                    frame, expected, radius=args.gate * 1.5))
            track = tracker.update(detections, timestamp) if tracker else None
            if args.stage >= 6 and args.prediction_mode == "plane" and ground.mode == "reference":
                if ground.observe_reference(track, timestamp):
                    trajectory.clear()
                    prediction_memory.reset()
            if trajectory is not None:
                trajectory.update(track, timestamp)
                trajectory.draw(frame)
            shown = ([track.detection] if track.detection else []) if track else detections
            for detection in shown:
                x1, y1, x2, y2 = map(round, detection.box)
                box_color = (0, 255, 0) if detection.source == "YOLO" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            if track and track.position:
                color = (0, 255, 0) if track.detection else (0, 165, 255)
                cv2.circle(frame, tuple(map(round, track.position)), 6, color, 2)
            if args.stage >= 6:
                measured = trajectory.measurements()
                if args.prediction_mode == "physics":
                    landing, reason = predict_physics(measured, timestamp, ground, args.horizon,
                                                      motion_model=args.motion_model,
                                                      size_observations=trajectory.size_observations())
                elif args.prediction_mode == "plane":
                    landing, reason = predict_floor_plane(measured, timestamp, ground, args.horizon,
                                                          motion_model=args.motion_model,
                                                          size_observations=trajectory.size_observations())
                else:
                    landing, reason = predict_landing(measured, timestamp,
                                                      None, frame.shape[1], args.horizon,
                                                      ground=ground, height=frame.shape[0])
                geometry_ready = ground.mode is None and (args.prediction_mode == "image" or ground.ready)
                candidate = landing
                if geometry_ready:
                    has_candidate = landing is not None
                    landing = prediction_memory.update(landing, timestamp,
                                                        observed=track.detection is not None,
                                                        new_track=track.new_track)
                    if has_candidate and landing is None and prediction_memory.status:
                        reason = prediction_memory.status
                else:
                    prediction_memory.reset()
                draw_prediction(frame, landing, reason, ground=ground)
                span = measured[-1][0]-measured[0][0] if len(measured)>1 else 0
                cv2.putText(frame, f"Input: {len(measured)} positions / {span:.2f}s | processing {fps:.1f} FPS",
                            (15, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                diagnostics.record(timestamp, args.prediction_mode, args.motion_model,
                                   fps, track, measured, candidate, landing, reason)
            cv2.putText(frame, f"Stage {args.stage} | Q: quit | R: reset", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if tracker:
                status = ((f"Tracking ({track.detection.source})") if track.detection else
                          "Coasting (no detection)" if track.position else "Searching for ball")
                cv2.putText(frame, status, (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 1)
            if args.max_frames and source.index >= args.max_frames:
                break
            if not args.headless:
                cv2.imshow("Tennis tracker", frame)
                # Avoid adding a full frame delay on top of inference time.
                elapsed = time.perf_counter() - frame_start
                delay = 1 if source.is_camera else max(1, round(1000 * (1 / source.fps - elapsed)))
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty("Tennis tracker", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if (ground is not None and args.prediction_mode == "physics"
                        and ground.mode == "dimensions"):
                    ground.handle_dimension_key(key)
                    # Keep typing in the video window; setup must not block the
                    # capture loop or accidentally trigger other keyboard actions.
                    continue
                if ground is not None:
                    if key in (ord("g"), ord("G")):
                        ground.begin("row" if args.prediction_mode == "image" else "floor")
                    elif key in (ord("f"), ord("F")) and args.prediction_mode == "plane":
                        if ground.begin_reference():
                            tracker.reset()
                    elif key in (ord("l"), ord("L")) and args.prediction_mode == "image":
                        ground.begin("guide")
                    elif key in (ord("c"), ord("C")):
                        ground.clear()
                    elif key == ord("[") and args.prediction_mode == "image":
                        ground.adjust_depth(-0.05)
                    elif key == ord("]") and args.prediction_mode == "image":
                        ground.adjust_depth(0.05)
                    elif key == 27:
                        ground.cancel()
                    if key in (ord("g"), ord("G"), ord("f"), ord("F"), ord("l"), ord("L"),
                               ord("c"), ord("C"), ord("["), ord("]"), 27):
                        prediction_memory.reset()
                        if trajectory is not None:
                            trajectory.clear()
                if key in (ord("r"), ord("R")):
                    if tracker:
                        tracker.reset()
                    if trajectory is not None:
                        trajectory.clear()
                    prediction_memory.reset()
    finally:
        if diagnostics is not None:
            diagnostics.close()
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
