# Validation performed

Verified in this Windows workspace with a project-local Python 3.12.10 runtime.

- Stage 1: created package, dependency list, and VS Code configuration.
- Stages 2–4: syntax checks, then temporary-video smoke checks with real YOLO
  inference; two tracking tests passed before adding trajectory history.
- Stage 5: smoke checks through stage 5 and four tracking/history tests passed
  before adding the landing predictor.
- Stage 6: smoke checks through all stages and all 12 synthetic tests passed.
- `pip check`: no broken requirements.
- Python compilation of application and tests passed; command-line help ran.

Latest floor-mapping update: 25 synthetic tests pass. Added checks for perspective
projection, a sloping guide's analytically solved intersection, rejecting contact
outside the floor region, rejecting crossed calibration corners, changing assumed
depth, three-observation provisional estimates, stationary-ball rejection, and
the blue bullseye color. Manual marking and accuracy still require a live test.

The current gravity-based update has 35 passing tests. Synthetic cases project
known 3D flights into one calibrated camera and recover withheld contact points.
Two throws at different landing depths use the same floor/camera setup without a
contact guide. Cases include distortion, missed observations, 0.4-pixel detection
noise, floor-pose recovery, changed resolution, stationary/short/straight motion,
stale observations, and the time horizon. Python compilation also passed.

The application integration check reads an actual temporary video, uses known
synthetic detections, simulates G and the four floor clicks, runs tracking and
physics, and checks the rendered blue target and recovered floor coordinates.
GUI calls and detections are mocked for this check. All stages also passed the
blank-video smoke checks with real YOLO inference. Stage 6 correctly withholds
physics predictions when no lens calibration is supplied.

No physical lens calibration or measured floor dimensions for the user's camera
were provided. The calibration GUI, real flight-model errors, live stream timing,
and real landing accuracy remain to be evaluated on the physical setup.

No-print/persistence update: 37 tests pass. Added recovery of an ideal centered
camera's focal length from a measured rectangle, automatic setup save/restore,
landing prediction with a restored setup, source/resolution mismatch rejection,
clear-pose persistence, and rejection of a front-parallel rectangle that contains
insufficient perspective. Estimated-lens assumptions and real accuracy are unverified.

G-key setup update: all 38 tests pass. A regression check starts setup without
command-line dimensions, rejects zero input, ignores clicks during dimension
entry, accepts width/length, then marks corners and obtains a valid floor pose.
Syntax compilation also passes. Keyboard focus and live GUI behavior still need
the user's camera-window check.

Current 2D-default update: all 46 tests pass, including the default application
loop with file input, mocked detections, missed frames, and blue rendering. The
2D loop is checked not to load metric camera calibration. The mathematical checks
recover a known parabolic line crossing, a flight through its apex, a noisy arc
with an isolated bad center and missing observations (within 5 pixels and 0.06s
of its known crossing), and a short linear arc explicitly labeled early. They
reject insufficient history, new incompatible motion, already-crossed targets,
and uncertain extrapolation. Single-click row placement is checked. All runnable
stages pass blank-video smoke checks with real YOLO inference; compilation passes.
These checks establish model behavior on controlled inputs, not real-world
landing accuracy or unknown depth recovery in 2D.

Floor-plane experiment: all 52 tests pass. New cases recover plane geometry
without entered dimensions, use reference balls at different floor locations,
predict different unknown landing depths, reject moving/tiny/color-fallback
references, and check saved setup compatibility. A full application test reads
temporary video, mocks detections/GUI calls, simulates G/F/R, saves the learned
plane, and verifies its blue target against a known synthetic contact. Lens
assumptions match the generated camera for the exact-recovery checks. These tests
do not validate actual YOLO box diameter, lens field of view, floor reference
placement, real flight timing, or landing accuracy on the user's camera.

First-prediction locking: all 57 tests pass. New checks cover rejecting short or
noisy candidates, target/impact-time disagreement, confirmation of three stable
estimates, ignoring later revisions and reacquired tracks, retaining the target
for 20 seconds without observations, and explicit reset. The default 2D app test
now runs a 100-frame video and verifies that its first blue target stays displayed
well past tracking expiration and the predicted crossing time. Compilation passes.
These are consistency/lifecycle checks, not a guarantee of physical landing accuracy.

Short-drop update: all 63 tests pass. The new tests reproduce a vertical fall that cannot lock under
the old time requirements, recover its known contact with the constrained drop
model, and test small center/size noise, stationary targets, missing size cues,
and clear sideways-motion rejection. The complete application replay uses a saved
plane, a held warm-up, real colored frames between mocked YOLO detections, CSV
logging, and a roughly 0.3s fall. It verifies locking before impact, less than
12-pixel error against the synthetic contact, and persistence after disappearance.
This does not establish accuracy on the user's real camera; recorded diagnostics
are available to inspect actual detections and rejection reasons.

The synthetic integration test follows a known parabola with two missing frames,
checks the expected ground intersection, draws the trajectory and target, and
checks that a lost track clears its history. Prediction tests also cover linear
motion, a stationary ball, insufficient history, stale observations, duplicate
timestamps, noisy fits, upward curvature, off-screen predictions, and horizon limits.

Repeat using the runtime already present here:

```powershell
.\.runtime\python.exe -m unittest discover -s tests -v
.\.runtime\python.exe tests\smoke_stages.py --through 6
.\.runtime\python.exe -m pip check
```

Smoke checks use an eight-frame blank MJPEG video. They exercise input, YOLO
inference, tracking calls, overlays, and clean video completion without opening
a GUI. They do not establish tennis-ball detection recall or landing accuracy.
No real tennis footage was provided, and live webcam capture, visual GUI behavior,
Q/R input, and mouse placement have not been manually verified. Use the README's
stage-by-stage visual checklist on your camera before relying on the display.
