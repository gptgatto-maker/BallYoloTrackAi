# Tennis ball tracker

A staged Python/OpenCV project with YOLO detection, Kalman tracking, recent
trajectory drawing, and a blue landing bullseye.

**Stage 6 defaults to approximate 2D prediction.** It fits recent image positions
and draws a blue bullseye where the future path crosses the selected ground line.
No printing, measured floor dimensions, or lens setup is needed. Actual landing
depth cannot be established by this image-coordinate predictor.

An **experimental floor-plane mode** is now available. It uses an unmeasured
rectangle, an assumed lens, and a resting tennis ball as a scale reference to
estimate contact anywhere on that patch. It requires no printing, tape-measured
dimensions, or selected landing line. See [FLOOR_PLANE.md](FLOOR_PLANE.md).

## Ready to run in this workspace

Python and dependencies are already installed in `.runtime`; YOLO starter weights
are downloaded. Open **Terminal → New Terminal** in VS Code and run each line once:

```powershell
cd "C:\Users\enzop\OneDrive\Desktop\Ai Tennis ball tracking"
.\.runtime\python.exe main.py --stage 2 --source 1
```

The output is a window named **Tennis tracker**. This version does not save
annotated videos. Click that window and press Q to quit before running another
camera command. Ctrl+C in the terminal stops the app or cancels an unfinished
command. Select `.runtime\python.exe` as the VS Code interpreter.

## Windows + VS Code installation on another computer

Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/, enabling
the launcher and "Add python.exe to PATH." Install VS Code and Microsoft's Python
extension. Open this folder and run these PowerShell commands separately:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` with **Ctrl+Shift+P → Python: Select Interpreter**.
For the remaining commands on that machine, replace `.\.runtime\python.exe` with
`.\.venv\Scripts\python.exe`. No environment activation or PowerShell policy change
is needed. Reopen VS Code after installing Python if the launcher is unavailable.

## Incremental stages

Stage 1 is the project structure. Close each stage with Q before running the next:

```powershell
.\.runtime\python.exe main.py --stage 2 --source 1
.\.runtime\python.exe main.py --stage 3 --source 1
.\.runtime\python.exe main.py --stage 4 --source 1
.\.runtime\python.exe main.py --stage 5 --source 1
```

| Stage | What to check |
| --- | --- |
| 2 | Live camera opens; Q exits. |
| 3 | Green boxes show YOLO detections. |
| 4 | Filtered marker follows the ball; orange means coasting through a missed detection. |
| 5 | Cyan trail follows accepted observations. |
| 6 | A blue target marks a plausible future crossing of the selected 2D ground line. |

The default is stage 2. Camera 0 is often the built-in webcam; use the index that
actually shows your phone/capture card. New installations download the YOLO model
on the first detection run. CPU is the default; actual speed depends on hardware.

## Stage 6: simple 2D prediction

```powershell
.\.runtime\python.exe main.py --stage 6 --source 1 --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --history-seconds 0.8 --horizon 1.5 --ground-y 0.85
```

Click the video for keyboard focus. Controls:

- **G**, then one click: place the assumed horizontal ground line at that row.
- **L**, then two clicks: select a sloping line if desired.
- **[ / ]**: move the horizontal line farther up/down the image.
- **R**: clear tracking and prediction before a fresh throw or after a bounce/hit.
- **C**: reset the selected line; **Esc**: cancel marking; **Q**: quit.

`--ground-y 0.85` defaults to 85% of image height. For a fixed side-on test,
select the row where the ball's center would touch the floor. The blue bullseye
is an approximate image-coordinate crossing, not a measured 3D landing location.
A single row cannot represent every depth on the floor; the line is an assumption.
The saved metric floor/lens profiles are not loaded in the default 2D mode.

The predictor fits X linearly and Y quadratically, downweights isolated bad
detections, and tries recent shorter windows to exclude holding the ball before
release. It needs at least five accepted positions spanning 0.12s and visible
vertical motion. When curvature is unresolved but a clean descending straight
segment is available, a provisional linear fit can extrapolate up to 0.75s.
Otherwise it uses the quadratic and solves the future downward line crossing.

**The first displayed prediction is locked until R.** In the general flight model, before showing it the app
requires at least eight accepted observations over 0.30 seconds, bounded fit
uncertainty, and at least three agreeing predictions over 0.05 seconds. Plane and
physics modes also require a non-provisional fit. The confirming targets must
agree within 12 image pixels and their predicted impact times within 0.12 seconds.
This can delay the first marker but avoids pinning a very early unstable estimate.
The dedicated drop model uses its shorter, size-assisted checks described in
[DROP_TEST.md](DROP_TEST.md).
Consistency does not guarantee accuracy when geometry or motion assumptions are
wrong. The display says **First prediction locked** and retains the blue target
after the ball disappears. R clears it for another throw; changing ground setup
also clears predictions whose coordinates would no longer be valid.

Estimates are rejected when the trajectory is too noisy, observations are stale,
the crossing is behind the current ball or off-screen, or extrapolation is highly
sensitive to observation noise. After a target is locked, later motion and new
tracks do not replace it. This lets you compare that first prediction with where
the ball actually lands. Until a first target is accepted, the waiting message
explains why it is withheld.
The displayed fit spread describes detector-noise sensitivity in pixels, not a
guaranteed landing accuracy interval. Perspective, spin, bounces, and false
associations can still invalidate the model.

For a visual accuracy check, use a fixed camera and a throw across the image in a
narrow depth range. Record several throws and compare the predicted bullseye with
the measured crossing of the selected line. A mostly stationary ball should not
produce a landing target; a missing target should have an on-screen waiting reason.

## Optional calibrated physics experiment

For a straight-down drop that finishes before the general flight model can lock,
see [DROP_TEST.md](DROP_TEST.md). It uses `--motion-model drop`, approximate ball-size
depth cues, faster local tracking between YOLO frames, and a shorter first-lock
check under a zero-sideways-velocity assumption. It preserves the first target
until R and can append CSV decisions for diagnosing real failures.

For the unmeasured floor-plane attempt, use:

```powershell
.\.runtime\python.exe main.py --stage 6 --source 1 --prediction-mode plane --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --horizon 2.0
```

In this mode G marks four corners and F learns the floor from a resting ball.
Its lens geometry is assumed and accuracy is unverified. Follow
[FLOOR_PLANE.md](FLOOR_PLANE.md) for setup and assumptions.

The earlier measured-floor experiment is retained behind `--prediction-mode physics`.
It is not needed for 2D tracking. See [PHYSICS_SETUP.md](PHYSICS_SETUP.md) if you
choose to revisit it. Its saved setup files are preserved.

## Detection, tuning, and sources

The YOLO11 nano starter detects COCO's **sports ball** class, not tennis balls
specifically. Tiny, fast, or blurred balls may be missed. Supply a specialized
Ultralytics-compatible detector with `--weights "C:\Models\tennis-best.pt" --class-id 0`,
using your model's actual class ID. The default recognizes a class named sports
ball, tennis ball, or ball.

YOLO boxes must contain tennis-yellow pixels unless `--allow-any-ball` is set.
Two consistent frames establish a track. A local yellow search assists after a
YOLO miss; a yellow box and `color recovery` identify it. `--no-color-recovery`
disables this fallback. Other yellow objects or balls can fool it.

Only one ball is tracked. `--gate` sets the maximum association distance in pixels;
larger values permit faster motion but increase target swaps. `--max-gap` sets
seconds of coasting before reset. Coasting positions never become fit measurements.
A larger `--imgsz` may help small-ball detection but slows processing. Bright
lighting reduces blur. Press R after a hit or bounce to start fresh motion history.

Recorded constant-frame-rate video uses frame index/FPS for fitting time:

```powershell
.\.runtime\python.exe main.py --stage 5 --source "C:\Videos\tennis.mp4"
```

For physics stage 6, the video must match the calibrated camera, lens, crop, and
resolution, with a floor rectangle whose dimensions you know. Mark that rectangle
in the footage. Playback may be slower than real time; fitting uses video FPS.
Convert variable-frame-rate footage to constant frame rate first.

OpenCV-compatible direct HTTP/RTSP streams are accepted as sources. This capture
class treats non-numeric sources as files and uses reported FPS, so network
timing/dropped frames cannot be validated. Use a local webcam device or constant
frame-rate recording for evaluating physical landing predictions. Webcam receipt
timestamps can also be biased by buffering and varying stream delay.

## Files

```text
main.py                           Staged application and controls
calibrate_camera.py               Lens setup helper
calibration/checkerboard.svg       Printable pattern
tennis_tracker/capture.py          Capture and timestamps
tennis_tracker/detection.py        YOLO and local yellow recovery
tennis_tracker/tracking.py         Kalman smoothing and association
tennis_tracker/trajectory.py       Recent observations and trail
tennis_tracker/calibration.py      Lens data and metric floor pose
tennis_tracker/metric_floor.py     Measured floor setup and grid
tennis_tracker/physics.py          Calibrated ballistic landing fit
tennis_tracker/floor_plane.py      Experimental unmeasured plane + ball-size reference
tennis_tracker/ground.py           Original image-guide floor demo
tennis_tracker/prediction.py       Original 2D fit and blue rendering
tennis_tracker/prediction_memory.py First-target confirmation and retention until R
tennis_tracker/drop_fit.py         Constrained drop fit with size/held-position cues
tennis_tracker/diagnostics.py      Optional numerical decision log
PHYSICS_SETUP.md                   Full calibrated prediction guide
FLOOR_PLANE.md                     Unmeasured floor-plane experiment guide
DROP_TEST.md                       Short vertical-drop test and troubleshooting
requirements.txt                  Dependencies
tests/                            Synthetic checks and smoke test
```

## Verification and troubleshooting

```powershell
.\.runtime\python.exe -m unittest discover -s tests -v
.\.runtime\python.exe tests\smoke_stages.py --through 6
```

Smoke checks use temporary blank video and real YOLO inference without a GUI.
Synthetic checks recover known projected flights and check ambiguity, distortion,
noise, tracking, and rendering. They do not establish real landing accuracy.
See [VALIDATION.md](VALIDATION.md).

If the camera will not open, close other camera apps, try another index, and
enable camera access for desktop apps in Windows Settings. For windows install
`opencv-python`, not the headless package. Read the on-screen waiting reason if
there is no landing marker. Physics estimates assume correct calibration and
continuous gravity-dominated flight; spin, drag, hits, camera motion, and false
detections can invalidate them. This version is unsuitable for line calls.

References: [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/),
[OpenCV calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).
#   B a l l Y o l o T r a c k A i  
 