# Predict a landing without choosing where it will land

Stage 6 defaults to 2D mode. This guide covers the optional **experimental calibrated
flight model**, selected with `--prediction-mode physics`. Its blue
bullseye estimates a contact point anywhere inside a measured floor rectangle.
There is no contact guide in this mode. The original 2D teaching demo remains
available with `--prediction-mode image`.

The new model needs lens geometry and the location/dimensions of the floor.
The default **no-print setup** estimates both from four corners of a measured
floor rectangle, with assumptions about the lens. It saves the setup and loads
it on later runs with the same source and resolution. The model then estimates
ball position and velocity from the flight using gravity; you never select the
landing point or depth.

## Default: no printing, setup once, then reuse

You can enter the initial dimensions directly in the video window. Start with
the normal camera command, click the video to give it keyboard focus, and press G:

```powershell
.\.runtime\python.exe main.py --stage 6 --prediction-mode physics --source 1 --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --horizon 2.0
```

Type the measured rectangle width in meters, press Enter, then type its length
in meters and press Enter. Backspace edits; Esc cancels. Next click the four
corners in the displayed order. The dimensions and successful floor/lens setup
are saved automatically. You do not need to restart to add command-line options.
The command-line route below also works if you prefer entering measurements there.

Mark a real rectangular patch of floor with tape, or use a known rectangular mat
or tile rectangle. Measure its width and length. If it is actually 2m by 3m:

```powershell
.\.runtime\python.exe main.py --stage 6 --prediction-mode physics --source 1 --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --horizon 2.0 --floor-width 2.0 --floor-length 3.0
```

Use your actual dimensions. Press G and click far-left, far-right, near-right,
near-left. The program estimates focal length from the rectangle's perspective,
solves the floor pose, and saves `calibration/setup-<source-id>.json` automatically.
It assumes a centered lens, square pixels, and no lens distortion. The display
labels this **Approximate lens**. Strong lens distortion, digital cropping, bad
dimensions/clicks, or insufficient perspective can make it inaccurate or fail.
The program rejects estimates sensitive to small corner-click errors.

On future runs use the same source, with no dimensions or clicks required:

```powershell
.\.runtime\python.exe main.py --stage 6 --prediction-mode physics --source 1 --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --horizon 2.0
```

**Keep the physical camera position, lens, zoom, and stream settings unchanged.**
Check that the restored grid still aligns with the floor. The app does not yet
track camera movement or verify physical camera identity; a source number is not
a hardware identifier. If you move the phone or change its view, the saved floor
pose becomes invalid. C clears the saved pose while remembering dimensions/lens;
G marks a new pose. Saving a setup is not automatic recalibration of a moving phone.

For zero manual floor setup, the program would need another source of camera/floor
geometry, such as synchronized phone AR metadata. Its current OpenCV webcam feed
contains only images and provides no such data. This no-print route still has one
initial measured rectangle; it does not secretly guess a landing depth.

The checkerboard method below is optional if you want measured lens distortion
instead of the approximate lens model. It is not required for the no-print route.

## Optional: measure the phone/webcam lens

Open [calibration/checkerboard.svg](calibration/checkerboard.svg) in a browser.
Print it in landscape at **100% / actual size** on A4 or larger paper, with browser
headers/footers disabled. It has 10 by 7 squares and **9 by 6 inner corners**.
Measure a square: it should be 24mm wide. Attach the print to a rigid, flat board.
A screen showing the pattern can cause reflections or refresh artifacts; a print
is preferred. Keep a white border around the pattern.

Start this from the VS Code PowerShell terminal:

```powershell
cd "C:\Users\enzop\OneDrive\Desktop\Ai Tennis ball tracking"
.\.runtime\python.exe calibrate_camera.py --source 1
```

Use the source number that actually shows your phone. In the calibration window:

1. Hold the whole board in view until colored corner markers appear.
2. Press **Space** to capture it. Move/tilt the board between captures.
3. Collect around **20 different poses**: left/right/top/bottom of the image,
   closer/farther, and tilted in several directions. Avoid blur and reflections.
4. Press **C** to compute and save `calibration/camera.json`. The program requires
   at least 12 different poses and rejects poor fits or inadequate image coverage.
5. **Q** quits without saving if you want to restart.

If your squares have a different size, use meters; for 20mm squares:

```powershell
.\.runtime\python.exe calibrate_camera.py --source 1 --square-m 0.020
```

The saved lens calibration can be reused with the same physical camera, phone
lens, resolution, zoom, crop, and stabilization settings. The stream must have
the exact calibrated resolution; this version deliberately refuses to guess
when sizes differ. Keep digital zoom and automatic cropping/stabilization off
when possible. Changing lens or stream settings needs new lens calibration.

## 2. Mark a measured rectangle on the floor

Put four small pieces of tape on a flat floor to form a real rectangle. Measure
its width left-to-right and length near-to-far in meters. The four sides must
form a rectangle in the real world, even though it appears as a quadrilateral
in the camera image. Choose a patch large enough to contain likely landings;
its location does not predict which part the ball will land on.

For example, if your measured patch is **2 meters wide by 3 meters long**, run:

```powershell
.\.runtime\python.exe main.py --stage 6 --prediction-mode physics --source 1 --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --history-seconds 0.8 --horizon 2.0 --floor-width 2.0 --floor-length 3.0
```

Replace `2.0` and `3.0` with **your actual measurements**. Guessing these dimensions
changes the geometry and can bias the estimate. No extra dependencies are needed.

Keep the camera fixed. Click the video window and press **G**, then click the tape
markers in this order:

1. Far-left
2. Far-right
3. Near-right
4. Near-left

The projected floor grid should line up with the real rectangle. Misalignment
means the dimensions, clicks, lens calibration, or stream settings need attention.
The setup is accepted only when the camera is above the floor and the corner
reprojection error is small. Four clicks alone are not a guarantee of accuracy.

Floor coordinates use near-left as the origin: X goes right and Y goes toward
the far edge. This setup describes the floor for every subsequent throw while
the camera stays fixed. It is saved and restored after restarting with the same
source/resolution, including the lens and measured dimensions.
After moving the camera, press **C** to clear the old floor and **G** to mark it
again. The lens calibration can stay if lens/zoom/resolution did not change.

## 3. Throw and observe the blue bullseye

Press **R** immediately before each fresh throw to clear the previous flight.
Use a gentle toss with visible curvature, across the view at an angle. Show as
much of the flight as possible; throws directly toward the lens can leave depth
very poorly constrained. Bright lighting reduces motion blur.

The model uses at least six detections spanning 0.18 seconds. It may need more
observations when the arc is noisy or the viewing geometry is weak. It tries
recent shorter windows to discard samples from holding the ball before release.
The first target is now shown only after a non-provisional fit and consistency
confirmation. It is labeled **First prediction locked**, stays fixed through later
updates or disappearance, and remains until R (or a ground setup change). A stable
fit does not guarantee accuracy under incorrect geometry or motion assumptions.

The displayed X/Y coordinates are the estimated floor contact relative to your
tape rectangle. A bullseye is drawn only when the estimated landing is inside
that rectangle and in the camera view. The blue path is projected from the fitted
3D flight; the bullseye is on the actual floor plane, accounting for ball radius.

Messages explain withheld predictions:

- **Calibrate lens and measured floor first:** finish both setup steps.
- **Depth uncertain:** longer/clearer curved flight or a different camera angle
  is needed. A tiny straight-looking image segment does not determine depth.
- **Does not fit free flight:** the ball may still be held, bounced, hit, or the
  detector followed another object. R starts a fresh fit.
- **Outside measured floor patch / camera view:** enlarge the measured patch
  or reposition the camera before recalibrating the floor.

## What this model can and cannot establish

For measured timestamps it fits camera-space position and velocity to
`P(t) = P0 + V*t + 0.5*G*t^2`, with gravity `9.81 m/s^2` pointing down relative
to the calibrated floor. It fits the full perspective projection instead of
assuming image Y is physical height. It finds the future time when the center
height equals the tennis-ball radius (33.5mm), projects that floor contact back
into the image, and draws the blue target.

The six unknown position/velocity components are first solved with least squares
from viewing-ray equations, then refined using image reprojection residuals.
Checks reject ambiguous viewing geometry, implausible heights/speeds, large fit
errors, stationary targets, and high estimated sensitivity to detector noise.
The on-screen **noise sensitivity** is twice the largest directional standard
deviation propagated from the fit. It is not a guaranteed accuracy interval and
does not include errors in calibration (especially the estimated lens), timestamps,
spin/drag, or the detector. A stable estimate under detector noise can still be
wrong when the no-print lens assumptions do not match the physical camera.

This is a physics-constrained estimate from a single camera, not measured 3D
ground truth. Its assumptions are a fixed camera, correctly calibrated lens and
flat ground, accurate frame times, one continuous free flight, and negligible
drag/spin for the gentle indoor toss. A low fit residual alone does not prove a
correct landing. Real fast tennis shots need a stronger flight model and detector;
two synchronized calibrated cameras provide more direct depth measurements.

Live streams may buffer frames. Webcam timestamps here are receipt times, so
buffering, jitter, variable latency, or dropped frames can bias the gravity fit.
For evaluation use constant-frame-rate recorded video with known FPS. Do not
claim physical accuracy from smoke checks alone.

## Verification

```powershell
.\.runtime\python.exe -m unittest discover -s tests -v
.\.runtime\python.exe tests\smoke_stages.py --through 6
```

Synthetic tests project known 3D flights into a calibrated camera and recover
their unknown landing depths. They cover lens distortion, missed observations,
detector noise, floor pose, and ambiguous viewing geometry. These demonstrate
the math under controlled assumptions. Your physical setup and detection accuracy
still require comparison between predicted and actual bounces on recorded throws.

References: [OpenCV lens calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html),
[OpenCV camera pose](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
