# Test a straight-down drop

The general flight model can be poorly conditioned during a short straight-looking
fall, and its first-lock check previously needed 0.30–0.35 seconds plus confirmation.
A ball dropped about 0.5m can reach the floor in roughly 0.3 seconds, so even good
detections could fail to produce a target before contact.

Drop mode makes an explicit stronger assumption: **zero sideways velocity**.
It fits camera-space position and vertical velocity under gravity, adds approximate
ball-size depth cues, and uses recent held-ball observations to stabilize the
horizontal floor position without treating the held period as free fall. It also
aligns color-recovery box sizes to nearby stationary YOLO boxes when enough matched
observations exist. This is an experimental model, not direct depth measurement.

Close the old video window with Q. Run this one line:

```powershell
.\.runtime\python.exe main.py --stage 6 --source 1 --prediction-mode plane --motion-model drop --detect-every 3 --confidence 0.10 --gate 250 --max-gap 0.8 --imgsz 640 --horizon 2.0 --diagnostics drop-diagnostics.csv
```

Your saved plane setup can be reused if the camera/view/source/resolution are the
same. The program does not alter that saved setup when running this test.

1. Hold the tennis ball still and fully visible over the mapped floor patch.
2. Click the video window and press R. Wait about one second, until its marker
   follows steadily. This allows tracking and detector-size alignment to warm up.
3. Release it straight down without a sideways push. Do not use drop mode for
   ordinary sideways throws; remove `--motion-model drop` for those.
4. The first accepted blue target locks and stays until R. It needs five measured
   positions spanning 0.12s, a non-provisional constrained fit, projected noise
   sensitivity at most 25 pixels, and two consistent candidates over 0.03s.

YOLO runs every third frame while a track is active; local yellow recovery handles
the intervening frames. Acquisition and F floor-reference collection still run
YOLO on every frame. A one-frame camera buffer is requested where supported, but
Windows backends may ignore that setting.

The on-screen **Input** line shows accepted position count, recent history duration,
and processing FPS. The bottom waiting message distinguishes insufficient data,
unusable size cues, incompatible motion, high uncertainty, and incomplete setup.
If the ball marker vanishes, the issue is detection/capture; if it stays visible
but no target appears, inspect the waiting reason. A low frame rate can still leave
too few observations during a short fall.

The command appends numerical decisions to `drop-diagnostics.csv` in this folder.
It records detector source, centers, box sizes, candidate count/span, estimated
noise sensitivity, first-lock state, and waiting reason. It does not record video.
Press Q to close and finish the log. Each run has a separate session ID and earlier
rows are preserved. This lets us diagnose an actual failure rather than relying
only on ideal synthetic tests.

The assumed lens, resting-ball floor scale, live frame timestamps, blur, occlusion,
and box-size bias still affect real landing accuracy. Noise sensitivity does not
bound all such errors. The zero-sideways assumption may fit a real drop but must
not be applied as a general throw model. Incorrect floor geometry is not repaired
by this mode.
