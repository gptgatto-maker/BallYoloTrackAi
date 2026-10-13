# Unmeasured floor-plane experiment

This tries predicting contact anywhere on a floor patch, without selecting a
landing line, entering floor dimensions, or printing anything. It is separate
from the default 2D line predictor. The on-screen bullseye remains blue.

## Run and set up once

Close the old video window and run this one command:

```powershell
.\.runtime\python.exe main.py --stage 6 --source 1 --prediction-mode plane --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --history-seconds 0.8 --horizon 2.0
```

Use the source index that shows your camera. Keep the phone fixed and use its
normal lens/zoom settings. Make sure a reasonably large floor patch is visible.

1. Click inside the video window for keyboard focus, then press **G**.
2. Click the corners of a patch that is **rectangular on the real floor**:
   far-left, far-right, near-right, near-left. Existing tile corners or a
   rectangular mat are useful. No dimensions are entered. An arbitrary screen
   quadrilateral is not automatically a physical rectangle.
3. Put a tennis ball **resting on the floor**, anywhere well inside that patch.
   Keep hands clear. Press **F**, then leave it still until the status says the
   plane was learned/saved. This location is a reference, not a requested future
   landing location. It needs a clear round YOLO box, at least 8 pixels across,
   stable for 12 observations spanning at least 0.4 seconds.
4. Pick up the same ball, press **R**, and make a gentle toss with a visible arc.
   The blue point estimates its future floor contact inside the patch.

The **first displayed target stays fixed until R**, even after the throw finishes,
the ball disappears, or tracking finds it again. Before locking, it waits for a
non-provisional fit with at least eight observations over 0.30 seconds, low fitting
noise sensitivity, and three agreeing targets spanning at least 0.05 seconds.
Expect the display to say **First prediction locked**. This checks consistency;
it does not remove errors from the assumed lens or resting-ball reference.
Press R before each next throw. Altering floor setup also clears the target.

For short straight-down drops, use the dedicated model in [DROP_TEST.md](DROP_TEST.md).
Its shorter first-lock rule is paired with an explicit zero-sideways assumption
and ball-size depth cues; the general throw model keeps its stricter checks.

The completed setup is saved automatically in `calibration/plane-<source-id>.json`
and restored when you run the same command again. You do not need to repeat G/F
while camera position, view, lens, source, resolution, and assumptions remain the
same. **Moving the camera invalidates the saved plane.** The app does not yet
track camera motion. Check that the restored grid aligns with the floor.

**C** clears the setup; **Esc** cancels marking/reference collection; **Q** exits.
**F** can relearn the resting-ball scale after the floor is outlined.

## What provides the missing information

An image floor outline alone does not provide an airborne ball's height.
This experiment adds two explicit assumptions and one reference:

- A centered pinhole camera, square pixels, zero distortion, and a **70-degree
  horizontal field of view** by default. This is an assumed lens, not a measured
  calibration of your phone. The display keeps that assumption visible.
- Four corners of a real rectangle. Its image homography, together with the
  assumed lens, estimates floor orientation and its aspect ratio in arbitrary
  units. Poorly conditioned or incompatible corners are rejected.
- The known approximate tennis-ball diameter, **67mm**, plus its observed size
  when resting on the floor. Approximate depth is `focal_pixels * real_diameter /
  pixel_diameter`. The ball center is one radius above the ground. This supplies
  floor distance/scale without using a tape measure.

The flight then uses the existing gravity-based least-squares model with observed
2D centers and timestamps. It estimates a spatial trajectory, finds future
intersection with the inferred plane (accounting for the ball radius), and
projects that contact into the video. You are not asked to choose its landing
depth. Ball diameter is used during the stable floor reference, not measured from
a blurred moving ball during every frame.

This is still a **one-time reference setup**, not a fully automatic scene-depth
measurement. Floor geometry and scale are inferred under assumptions. It would
be incorrect to claim precise calibrated 3D coordinates or guaranteed contact
accuracy from it.

## Limits and troubleshooting

- A box can include background, cut off part of the ball, or vary with lighting.
  Stable does not necessarily mean its diameter is correct; that biases the floor.
- A ball held above the floor during F teaches the wrong ground plane. The
  software cannot verify that it is physically resting just from its pixels.
- Wide-angle lenses, distortion, zoom, cropping, and wrong lens assumptions can
  produce a plausible-looking but incorrect target. The displayed motion-fit
  checks do not bound all such systematic errors.
- If your camera's actual horizontal field of view is known, provide it, for
  example `--plane-fov 60`. Changing this assumption requires a new G/F setup.
  Do not tune it solely to make one throw's prediction look correct.
- Multiple balls, false detections, insufficient visible curvature, hits/bounces,
  drag/spin, bad timestamps, and buffering can invalidate the flight model.
- If F waits for detection, use better light or place the ball closer while
  keeping it fully within the outlined patch. The reference requires YOLO;
  color-recovery detections are intentionally not used to establish scale.
- A target outside the floor patch/view, a stationary ball, or an ambiguous flight
  is withheld with an on-screen reason. R starts a fresh motion fit.

Compare predictions with actual bounces across recorded throws before judging
accuracy. For a meaningful first test, use gentle throws and a fixed camera with
a clear view of the curved flight and floor. This is unsuitable for line calls.

## Checks

Synthetic checks recover an unmeasured floor from an ideal camera and ball-size
reference, predict throws at different unknown landing depths, reject unstable
references, restore saved setups, and exercise the full G/F/R app workflow.
They use generated detections and do not establish real camera/detector accuracy.

```powershell
.\.runtime\python.exe -m unittest discover -s tests -v
.\.runtime\python.exe tests\smoke_stages.py --only-stage 6 --prediction-mode plane
```

The ball-size cue is inspired by the geometric principle described in
[3D Ball Localization From a Single Calibrated Image](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Van_Zandycke_3D_Ball_Localization_From_a_Single_Calibrated_Image_CVPRW_2022_paper.html).
That work uses calibrated cameras and a learned diameter estimator. This project
uses an assumed lens and the smaller YOLO box dimension for the reference, so it
does not inherit the paper's reported accuracy.
